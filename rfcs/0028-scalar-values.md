- Start Date: 2026-05-04
- Authors: @ngates
- RFC PR: [vortex-data/rfcs#28](https://github.com/vortex-data/rfcs/pull/28)

# Scalar Values and Complex Constants

## Summary

Vortex should keep `Scalar` as a small, host-resident, context-free value representation, and stop
using it as the primary execution representation for complex values. Complex constants should be
represented in the array world as length-1 arrays wrapped by `ConstantArray`, and complex expression
literals should serialize as singleton arrays instead of recursively nested scalar values.

This proposal introduces a scalar-or-row-backed constant representation:

```rust
pub enum ConstantValue {
    Scalar(Scalar),
    Row(ArrayRef), // invariant: len == 1, dtype == the ConstantArray dtype
}
```

Scalar-backed constants remain the fast path for nulls, booleans, primitives, decimals, UTF-8, and
binary values. Row-backed constants become the representation for non-null list, fixed-size-list,
struct, variant, and other complex values where nested scalar materialization is expensive or
requires array-level storage.

## Motivation

Vortex currently uses `ScalarValue::Tuple(Vec<Option<ScalarValue>>)` for list, fixed-size-list, and
struct scalars. That is convenient for simple expression literals, but it is a poor representation
for execution and serialization of list-like values.

The main problems are:

- Nested scalar values duplicate structure that Vortex already represents efficiently as arrays.
- Constructing a list scalar from an array row requires recursively calling `execute_scalar` for
  every nested element.
- Serializing a complex literal as protobuf recursively expands the value tree instead of preserving
  buffers, offsets, validity, and existing encodings.
- `ConstantArray` currently stores only a `Scalar`, so array kernels that detect constants often
  accidentally force complex data back through scalar form.
- Moving scalar values toward array-backed storage would make `Scalar::try_new` depend on execution
  context, buffer residency, and possibly device synchronization, which would make `Scalar` much less
  useful as a simple host literal.

This has shown up most clearly in list-oriented expressions such as `list_contains`, where a constant
list literal should be cheap to carry around, serialize, and compare against, but currently becomes an
expensive nested scalar object.

## Goals

- Keep `Scalar` simple, host-resident, context-free, and cheap for primitive-style values.
- Let execution represent complex constants as arrays, preserving array buffers and encodings.
- Let expression literals carry complex values without recursively serializing nested scalar trees.
- Avoid requiring an `ExecutionCtx` to construct or validate a `Scalar`.
- Allow in-memory complex constants to hold device-resident array buffers without copying them into
  host scalar values.
- Preserve compatibility with existing scalar literal and constant-array encodings.

## Non-Goals

- This RFC does not remove `ScalarValue::Tuple` immediately.
- This RFC does not require every scalar-like API to move to arrays in one change.
- This RFC does not define a device-resident `Scalar`.
- This RFC does not require canonicalizing complex constants during expression deserialization.

## Design

### Scalar remains a host literal

`Scalar` should remain:

```rust
pub struct Scalar {
    dtype: DType,
    value: Option<ScalarValue>,
}
```

Its contract should be narrowed and documented:

- `Scalar` is a host value.
- `Scalar::try_new` validates only dtype/value compatibility.
- `Scalar::try_new` never needs an `ExecutionCtx`.
- `Scalar` never stores `ArrayRef`, `BufferHandle`, or device buffers.
- `Scalar` is appropriate for expression literals, stats, FFI values, Python values, display, tests,
  and scalar-at results where the caller explicitly asked for a scalar.

`ScalarValue::Tuple` remains valid for compatibility and for small host values. It should no longer
be the default representation for complex constants inside array execution.

### ConstantArray stores either a Scalar or a singleton row

`ConstantData` should be changed from:

```rust
pub struct ConstantData {
    scalar: Scalar,
}
```

to:

```rust
pub struct ConstantData {
    value: ConstantValue,
}

pub enum ConstantValue {
    Scalar(Scalar),
    Row(ArrayRef),
}
```

The row variant has these invariants:

- `row.len() == 1`
- `row.dtype() == constant_array.dtype()`
- the outer `ConstantArray` length is independent of the row length

Recommended constructors:

```rust
impl ConstantArray {
    pub fn new<S: Into<Scalar>>(scalar: S, len: usize) -> Self;

    pub fn try_new_value(value: ConstantValue, len: usize) -> VortexResult<Self>;

    pub fn try_new_row(row: ArrayRef, len: usize) -> VortexResult<Self>;

    pub fn constant_value(&self) -> ConstantValueRef<'_>;

    pub fn scalar(&self) -> Option<&Scalar>;

    pub fn row(&self) -> Option<&ArrayRef>;
}
```

The exact public API names can be adjusted, but the important distinction is that callers must be
able to ask whether a constant is scalar-backed or row-backed without forcing materialization.

`ArrayRef::as_constant()` should keep its current semantics as a scalar-only helper. It should return
`Some(Scalar)` only for scalar-backed constants. New helpers should be added for callers that can
handle complex constants:

```rust
impl ArrayRef {
    pub fn as_constant_scalar(&self) -> Option<Scalar>;
    pub fn as_constant_row(&self) -> Option<ArrayRef>;
    pub fn as_constant_value(&self) -> Option<ConstantValueRef<'_>>;
}
```

This is intentionally conservative. Existing kernels that call `as_constant()` generally expect a
`Scalar` and should not silently get a value that may require execution, allocation, or device reads.

### Choosing the representation

New constants should use scalar-backed representation when the value is naturally scalar:

- `Null`
- `Bool`
- `Primitive`
- `Decimal`
- `Utf8`
- `Binary`

New constants should use row-backed representation when the value is complex and non-null:

- `List`
- `FixedSizeList`
- `Struct`
- `Variant`
- `Extension` values whose storage dtype is complex

Null complex values may remain scalar-backed. A null complex scalar is compact and does not contain
nested values, so there is no benefit to constructing a singleton array just to represent absence.

The boundary should be pragmatic rather than philosophical. If a future scalar representation becomes
bad for large binary values, a threshold can move those to row-backed constants as well.

### Scalar extraction

`ConstantArray::execute_scalar(index, ctx)` should behave as follows:

- scalar-backed: clone and return the scalar
- row-backed: return `row.execute_scalar(0, ctx)`

This preserves the public scalar-at contract, but makes scalar materialization explicit and lazy. Code
that only needs to move, serialize, compare, or execute a complex constant can stay in array form.

### Validity

For scalar-backed constants, validity is unchanged:

- non-nullable dtype: `Validity::NonNullable`
- nullable and non-null scalar: `Validity::AllValid`
- nullable and null scalar: `Validity::AllInvalid`

For row-backed constants, validity should broadcast the singleton row validity:

- if the row is non-nullable: `Validity::NonNullable`
- if row 0 is valid: `Validity::AllValid`
- if row 0 is invalid: `Validity::AllInvalid`
- if determining row validity requires an array value, represent the validity as a constant boolean
  array instead of converting the row to a scalar

The last case is important for device and deferred execution. It should not force a host scalar read
just to determine whether a row-backed constant is valid.

### Canonicalization and execution

Scalar-backed constants keep the existing optimized canonicalization paths.

Row-backed constants should canonicalize by broadcasting the singleton row structurally:

- Struct constants produce a `StructArray` whose fields are `ConstantArray`s wrapping the singleton
  field rows.
- List constants produce a list-view-style canonical array whose offsets and sizes are constant and
  whose elements are the singleton row's element slice.
- Fixed-size-list constants may need to materialize repeated elements when a canonical
  `FixedSizeListArray` is requested, because that canonical layout requires `len * list_size`
  elements. This is acceptable because canonicalization is an explicit execution boundary.
- Primitive, decimal, UTF-8, binary, and bool row-backed constants are allowed but should usually be
  normalized to scalar-backed constants.

The key rule is that row-backed constants should not be converted into recursive `ScalarValue::Tuple`
except when an API explicitly asks for a `Scalar`.

### Serialization of ConstantArray

The existing serialized form for `vortex.constant` should remain readable:

- metadata: empty
- buffers: one protobuf-encoded `ScalarValue`
- children: none

Add a row-backed serialized form:

- metadata: empty, or a small version/kind marker if desired
- buffers: none
- children: one array node, decoded with the same dtype and length `1`

Deserialization should accept both forms:

```text
buffers.len() == 1 && children.len() == 0 => Scalar-backed legacy constant
buffers.len() == 0 && children.len() == 1 => Row-backed constant
otherwise => error
```

This reuses the existing array serialization machinery, including buffers, offsets, validity, and
encoding trees. It also means complex constants can preserve specialized encodings instead of being
flattened into nested protobuf scalar values.

New writers should use the row-backed form for complex non-null constants. Writers that need to
target older readers can keep an option to force the legacy scalar form or canonicalize complex
constants before writing.

### Expression literals

Expression literals should no longer be restricted to `Scalar`.

Introduce:

```rust
pub enum LiteralValue {
    Scalar(Scalar),
    Row(ArrayRef), // len == 1
}
```

`Literal` then becomes:

```rust
impl ScalarFnVTable for Literal {
    type Options = LiteralValue;
}
```

Execution is straightforward:

```rust
match literal {
    LiteralValue::Scalar(s) => ConstantArray::new(s.clone(), row_count),
    LiteralValue::Row(row) => ConstantArray::try_new_row(row.clone(), row_count)?,
}
```

Recommended expression constructors:

```rust
pub fn lit(value: impl Into<Scalar>) -> Expression;

pub fn lit_row(row: ArrayRef) -> VortexResult<Expression>;

pub fn lit_value(value: LiteralValue) -> VortexResult<Expression>;
```

`lit(value: impl Into<Scalar>)` remains for compatibility and simple values. Integrations that need
to pass list-like or struct-like values into expressions should use `lit_row`.

In a later migration, `lit(Scalar::list(...))` may choose to build a singleton array internally, but
that is not required by this RFC.

### Expression literal serialization

The current protobuf literal options are:

```proto
message LiteralOpts {
  vortex.scalar.Scalar value = 1;
}
```

Replace this with a oneof:

```proto
message LiteralOpts {
  oneof value {
    vortex.scalar.Scalar scalar = 1;
    ArrayLiteral array = 2;
  }
}

message ArrayLiteral {
  repeated string encoding_ids = 1;
  vortex.dtype.DType dtype = 2;
  uint64 len = 3;
  bytes serialized_array = 4;
}
```

For this RFC, `ArrayLiteral.len` must be `1`. It is included so the serialized array can be decoded
using the existing array deserialization API, which needs dtype and length from the parent context.

The `encoding_ids` field carries the array serialization context. Existing array serialization stores
encoding IDs as indices in the flatbuffer; expression literals do not have the file footer's array
context, so they must carry their own context.

This requires adding a session-aware expression serialization path:

```rust
pub struct ExprSerializeOptions<'a> {
    pub array_ctx: &'a ArrayContext,
    pub session: &'a VortexSession,
}

pub trait ExprSerializeProtoExt {
    fn serialize_proto(&self) -> VortexResult<pb::Expr>;
    fn serialize_proto_with_options(&self, options: &ExprSerializeOptions<'_>) -> VortexResult<pb::Expr>;
}
```

The old `serialize_proto` can continue to work for scalar-only expressions. It should return an error
if asked to serialize a row-backed literal without the context required to serialize arrays.

Deserialization already receives a `VortexSession`, so it can decode `ArrayLiteral` by constructing a
`ReadContext` from `encoding_ids`, decoding `serialized_array` with `dtype` and `len`, and validating
that `len == 1`.

### DType validation

`Scalar::try_new` should continue to run dtype validation for scalar values. This validation is
purely structural and should not require an `ExecutionCtx`.

Row-backed constants and row-backed literals validate through array invariants:

- the singleton row array must be a valid `ArrayRef`
- its dtype must match the literal or constant dtype
- its length must be 1

This cleanly separates scalar validation from array validation. There is no need for an execution
context to construct a scalar, and there is no need for scalar validation to understand array buffers
or device residency.

### Device buffers

Device buffers should not be allowed inside `Scalar`.

Device buffers should be allowed inside row-backed constants and row-backed literals because those
are arrays. In-memory execution can preserve device residency. Scalar extraction from a row-backed
constant may require execution or host transfer depending on the underlying array and execution
context, but that cost is paid only when the caller explicitly requests a scalar.

Portable expression serialization should copy buffers to host in the same way array serialization
does today. A future device-aware expression transport can carry `BufferHandle`s or external device
segments, but that is out of scope for this RFC.

### Statistics

For scalar-backed constants, statistics remain unchanged.

For row-backed constants:

- `Stat::IsConstant` is exactly true.
- `Stat::NullCount` can be derived from row validity and outer length.
- `Stat::Min` and `Stat::Max` may be derived lazily by extracting row 0 as a scalar when the dtype
  supports scalar ordering and the value is non-null.
- If extracting a scalar would require undesirable execution, implementations may leave min/max
  absent unless a compute path explicitly requests them.

This preserves correctness while avoiding accidental scalarization during cheap metadata operations.

## Compatibility

Existing scalar-backed constants and scalar literals remain readable.

Existing readers will not understand the new row-backed `vortex.constant` serialized form if it is
encoded under the same encoding ID. They will fail because the constant encoding has zero buffers and
one child instead of one scalar buffer. This is a forward-compatibility limitation, not silent data
corruption.

Writers should expose a compatibility option:

- modern mode: write row-backed complex constants
- legacy mode: write scalar-backed complex constants, or canonicalize complex constants before writing

The expression protobuf change is backward-compatible for readers that accept both `scalar` and
`array` literal variants. Older readers will not understand `array` literals.

Public Rust API compatibility should be managed in phases:

1. Add new `ConstantValue` and literal APIs.
2. Keep existing scalar-only helpers for existing callers.
3. Migrate internal kernels that can benefit from row-backed constants.
4. Deprecate ambiguous APIs such as `ConstantArray::scalar()` if they cannot represent row-backed
   constants safely.
5. Consider breaking API cleanup only after downstream integrations have a migration path.

## Drawbacks

This adds a second representation for constants, and kernels must be explicit about whether they
need scalar constants or can operate on row-backed constants.

Expression serialization becomes more complex because array literals need an array serialization
context. The existing scalar-only expression serialization path is simpler, but it is also the source
of the current inefficiency for complex values.

Some code that currently assumes every constant has a `Scalar` will need to be audited. The upside is
that this audit makes accidental scalarization visible instead of hiding it behind `as_constant()`.

Row-backed constants do not remove all materialization costs. If a caller asks for canonical
fixed-size-list arrays, scalar extraction, or legacy serialization, Vortex may still need to build
repeated values. The important change is that these costs move to explicit boundaries.

## Alternatives

### Make Scalar array-backed

We could add `ScalarValue::Array(ArrayRef)` and represent complex scalars directly as singleton
arrays.

This is rejected because it makes `Scalar` no longer context-free. Scalar equality, hashing,
display, validation, and serialization would all need to handle arrays, and arrays may require
execution or device-to-host transfer. That would make `Scalar::try_new` and scalar literals depend on
execution context, which is exactly the direction we want to avoid.

### Replace Scalar with length-1 arrays everywhere

This is conceptually clean, but too disruptive. `Scalar` is still useful for stats, display, FFI,
Python interop, expression literals, and primitive constants. Replacing it everywhere would force
array execution into places that need a cheap value object.

This RFC takes the smaller step: arrays are used for complex execution constants, while `Scalar`
remains the host literal representation.

### Always canonicalize complex literals at deserialization

This avoids carrying arbitrary encodings inside expression literals, but it throws away information
and can eagerly allocate. If a literal was serialized as a specialized array, deserialization should
not immediately flatten it unless execution demands that.

### Add a separate `vortex.constant.row` encoding

A new encoding ID would make forward incompatibility clearer for old readers. It would also avoid
changing the shape of `vortex.constant`.

This is a reasonable fallback, but the in-memory model should still be a single logical
`ConstantArray` with scalar-backed and row-backed variants. Most users and kernels should not care
which wire-level encoding was used.

### Keep nested scalar values and optimize hot paths

We could add specialized list-scalar and struct-scalar storage to reduce allocation. That may help
some cases, but it still duplicates the array system and still does not solve device residency or
array literal serialization.

## Prior Art

Apache Arrow distinguishes between scalars and arrays, but computation generally operates over
arrays and treats scalar inputs as broadcast values. That is the model this RFC follows: scalar values
remain useful, but execution should be able to represent a broadcast value as array data.

Database vector engines commonly distinguish constant vectors from flat vectors. A constant vector
does not necessarily mean "store a recursive scalar object"; it means "the logical row value is the
same for every row." For nested values, the payload can still be represented by child vectors.

Vortex already has the pieces of this model: `ArrayRef`, `ConstantArray`, array serialization,
validity, and singleton rows. The missing piece is allowing constants and literals to carry singleton
arrays directly.

## Unresolved Questions

- Should row-backed constants use the existing `vortex.constant` encoding ID with a new child-based
  shape, or should the wire format use a separate `vortex.constant.row` encoding ID?
- Should `lit(Scalar::list(...))` continue to produce a scalar-backed literal, or should it eagerly
  build a singleton array for complex scalar values?
- What is the exact public API migration for `ConstantArray::scalar()` and `ArrayRef::as_constant()`?
- Should large UTF-8 or binary values ever become row-backed constants based on size?
- How much min/max statistic support should row-backed constants provide without explicit execution?

## Future Possibilities

Once row-backed constants exist, Vortex can add more array-native literal construction APIs in Python,
Java, C++, DuckDB, and DataFusion integrations.

Expression serialization could eventually use a general "literal payload" abstraction that supports
host arrays, device buffers, and external buffer references. That would allow complex literals to be
transported without copying through a monolithic protobuf payload.

The same singleton-row mechanism may also help with dictionary values, sparse fill values, and other
places where Vortex currently stores a complex repeated value as a scalar.
