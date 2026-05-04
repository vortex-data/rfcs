- Start Date: 2026-05-04
- Authors: @gatesn
- RFC PR: [vortex-data/rfcs#56](https://github.com/vortex-data/rfcs/pull/56)

# Scalar Values and Complex Constants

## Summary

Vortex should stop using recursive scalar objects as the row representation for arrays.

The new split is:

- `Scalar` remains a small host value for scalar-compatible values: null, bool, primitive, decimal,
  UTF-8, binary, and optionally small structs whose fields are scalar-compatible.
- Array row access becomes array-shaped. A row is a length-1 `ArrayRef`, not a type-erased `Scalar`.
- `ConstantArray` becomes the only execution representation for broadcast values of any dtype. It is
  always a wrapper around a length-1 child array.
- Complex expression literals are represented by a new `ArrayLiteral` expression that holds an
  `ArrayRef` directly.

There is no scalar-backed vs row-backed `ConstantArray` variant. The bet in this RFC is that the
uniform representation is worth paying for primitive constants too: even `ConstantArray(42, N)` is
represented as a length-1 primitive array wrapped by `ConstantArray`.

## Motivation

Today's `Scalar` has two jobs:

- it is a compact host value for simple values such as `i32`, `bool`, and `utf8`
- it is also a recursive container for nested values such as lists and structs

The first job is useful. Small struct-like host records can also be useful, for example expression
literals or metadata records. The second job duplicates the array system and makes list-like values
expensive.

Problems with the current design:

- List-like scalars recursively allocate and walk `ScalarValue::Tuple`.
- Extracting a nested scalar from an array row requires recursive `execute_scalar` calls.
- Complex expression literals serialize as recursive protobuf scalar values instead of preserving
  buffers, offsets, validity, and encodings.
- `ConstantArray` stores a `Scalar`, so complex constants are forced through nested scalar form even
  though execution already has an array representation.
- There is no clean public way to embed an `ArrayRef` in an expression. The codebase already has a
  private `ArrayExpr` for validity expressions, backed by fake equality and hashing. That is a sign
  the expression model is missing an explicit array literal node.

The desired model is simpler:

- scalar-compatible host values use `Scalar`
- array rows use arrays
- all broadcast execution uses `ConstantArray`
- expressions can carry either a host scalar literal or an array literal

## Goals

- Remove recursive scalar values from array row access, constants, and aggregate partial transport.
- Keep `Scalar` small, host-resident, and context-free.
- Allow restricted struct-like host scalars when every field is scalar-compatible.
- Remove type-erased `scalar_at` / `execute_scalar` as the universal array row API.
- Represent every `ConstantArray` as a broadcast of a length-1 child array.
- Add a first-class `ArrayLiteral` expression for embedding `ArrayRef` values in expressions.
- Avoid requiring content equality or hashing for embedded array literals.
- Preserve scalar function execution over `ArrayRef` inputs.
- Preserve a staged migration path from today's scalar APIs.

## Non-Goals

- This RFC does not require renaming every public API in one PR.
- This RFC does not require changing scalar function input semantics.
- This RFC does not define a device-resident `Scalar`.
- This RFC does not make `ArrayLiteral` content-equal. Array literal equality is pointer identity by
  design.

## Design

### Scalar means host value

The intended `Scalar` contract is:

- `Scalar` is a host value.
- `Scalar::try_new` validates dtype/value compatibility without an `ExecutionCtx`.
- `Scalar` never stores `ArrayRef`, `BufferHandle`, or device buffers.
- `Scalar` is valid for dtypes whose value fits in a direct host payload: null, bool, primitive,
  decimal, UTF-8, and binary.
- `Scalar` may also support a restricted struct payload, preferably named `ScalarValue::Struct`
  rather than generic `Tuple`. Every field dtype must itself be scalar-compatible. This gives Vortex
  a cheap host record without preserving list-like scalar trees.
- `Scalar` is appropriate for expression literals, scalar-compatible statistics, FFI values, Python
  values, display, tests, and APIs where the caller explicitly asks for a host value.

These dtypes do not have scalar values in the long-term model:

- list
- fixed-size-list
- variant
- extension values whose storage dtype is nested
- structs containing any field that is not scalar-compatible

Those values are represented as arrays. A single such value is a length-1 array.

The restricted struct scalar is not an array row representation. It is a host value for places where
building arrays would be needless overhead. Array rows, constants, builders, and aggregate partial
exchange still use arrays.

### ConstantArray is always a broadcast child

`ConstantArray` should store its value as a length-1 child array:

```rust
pub struct ConstantData;

impl ConstantArray {
    pub fn with_child(child: ArrayRef, broadcast_len: usize) -> VortexResult<Self>;
}
```

`ConstantArray::with_child` has these invariants:

- `child.len() == 1`
- `child.dtype() == constant.dtype()`
- the outer constant length is `broadcast_len`
- the outer constant has no separate validity
- if `child` is itself a `ConstantArray`, construction normalizes by unwrapping to the inner child

The value of `ConstantArray` at any logical row `i` is `child[0]`.

This is the only in-memory representation. There is no `ConstantValue::Scalar` variant and no
`ConstantValue::Row` variant. The child slot is the source of truth for all dtypes, including
primitive dtypes.

### Primitive constants pay the wrapper cost

This RFC intentionally chooses the uniform representation even for primitive constants.

For example:

```rust
ConstantArray::new(42i32, 1_000_000)
```

is represented as:

```text
ConstantArray {
    len: 1_000_000,
    child: PrimitiveArray<i32>([42]),
}
```

The expected costs are:

- one length-1 child array allocation for primitive constants
- one child-slot hop in constant fast paths
- updated constant kernels that read from the child instead of an inline scalar

The expected benefits are:

- one constant representation for every dtype
- no recursive scalar values for nested constants
- constant serialization reuses array serialization uniformly
- validity, stats, hashing, equality, slicing, and casting all flow through the array machinery

Hot kernels should still have ergonomic helpers, but those helpers must not introduce a second
storage representation. For example, an accessor like this is acceptable:

```rust
impl ConstantArray {
    pub fn scalar_value(&self, ctx: &mut ExecutionCtx) -> VortexResult<Option<Scalar>>;
}
```

or, for scalar-compatible fast paths:

```rust
impl ConstantArray {
    pub fn primitive_value<T>(&self) -> Option<T>;
}
```

These are read helpers over the length-1 child. They are not alternate storage.

### Equality, hashing, validity, and stats

`ConstantArray` equality and hashing should be content-based through the child slot. Two constants
with independently constructed but equal length-1 children should compare equal and hash equal.

Validity is owned by the child:

- if the child row is valid, the constant is all-valid
- if the child row is invalid, the constant is all-invalid
- if the child dtype is non-nullable, the constant is non-nullable

Statistics derive from the child:

- rank-style stats such as min, max, is_constant, and is_sorted pass through when available
- count-style stats such as null_count and true_count scale by `broadcast_len`
- nested stats should avoid constructing recursive scalar values

### Canonicalization and execution

Canonicalization should broadcast the length-1 child structurally:

- primitive, bool, decimal, UTF-8, and binary constants can materialize repeated canonical values
- struct constants can produce a `StructArray` whose fields are constant arrays over each child field
  row
- list constants can produce a list-view-style canonical array whose offsets and sizes repeat the
  singleton row shape
- fixed-size-list constants may need to materialize repeated elements when a canonical
  fixed-size-list array is requested

The key rule is that canonicalization may allocate at an explicit boundary, but normal constant
transport should not convert nested values into recursive scalar values.

### Scalar functions and broadcasting

Scalar functions already take `ArrayRef` inputs. This RFC keeps that interface unchanged.

The calling convention remains:

- every input array has logical length `args.row_count()`
- broadcasting is represented by `ConstantArray(len = args.row_count(), child = length_1_array)`
- naked length-1 arrays are not normal scalar function inputs, except for private sub-executions
  such as evaluating an all-constant expression once

Scalar functions may still use constant fast paths by checking whether an input is a `ConstantArray`.
This RFC only changes what a constant contains. It does not require changing the scalar-function
execution API, adding argument materialization helpers, or changing `Columnar`.

### ArrayLiteral expression

Expression literals should split into two expression nodes:

- scalar literal expression: stores a host `Scalar`
- array literal expression: stores an `ArrayRef`

Add a first-class `ArrayLiteral` expression:

```rust
pub struct ArrayLiteral;

pub struct ArrayLiteralOptions {
    array: ArrayRef,
}
```

`ArrayLiteralOptions` implements `PartialEq`, `Eq`, and `Hash` by array pointer identity, not by
array contents.

That means:

- two `ArrayLiteral`s wrapping the same `ArrayRef` allocation compare equal
- two independently constructed arrays with equal contents compare unequal
- hashing does not walk buffers, execute arrays, inspect device memory, or depend on array contents

This is the right tradeoff for expression identity. Embedded arrays can be large, compressed,
deferred, or device-resident. Expression equality should not accidentally become array equality.

`ArrayLiteral` has arity 0. Its return dtype is `array.dtype()`.

Execution:

```rust
impl ScalarFnVTable for ArrayLiteral {
    type Options = ArrayLiteralOptions;

    fn execute(
        &self,
        options: &ArrayLiteralOptions,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let array = options.array.clone();

        if array.len() == args.row_count() {
            return array.execute(ctx);
        }

        if array.len() == 1 {
            return Ok(ConstantArray::with_child(array, args.row_count())?.into_array());
        }

        vortex_bail!(
            "ArrayLiteral length {} cannot be used in execution scope of length {}",
            array.len(),
            args.row_count(),
        );
    }
}
```

Nested expression literal helpers should normally construct length-1 arrays and wrap them in
`ArrayLiteral`. Supporting `len == row_count` is useful for internal expression construction and can
replace the private `ArrayExpr` currently used by scalar-function validity expressions.

Recommended constructors:

```rust
pub fn lit(value: impl Into<Scalar>) -> Expression;

pub fn lit_array(array: ArrayRef) -> Expression;

pub fn lit_list(elements: ArrayRef) -> VortexResult<Expression>;

pub fn lit_fixed_size_list(elements: ArrayRef) -> VortexResult<Expression>;

pub fn lit_struct(
    dtype: StructDType,
    fields: impl IntoIterator<Item = (FieldName, ArrayRef)>,
) -> VortexResult<Expression>;

pub fn lit_variant(value: ArrayRef) -> VortexResult<Expression>;
```

The nested helpers build a length-1 array of the requested dtype and return `lit_array(row)`.
Scalar-compatible struct records may also use `lit(Scalar::struct(...))`; execution still
materializes them through a child-backed `ConstantArray`.

### Per-row access

Per-row access is fundamentally an array operation. The universal API should return a one-row array:

```rust
impl ArrayRef {
    pub fn row(&self, index: usize) -> VortexResult<ArrayRef> {
        self.slice(index..index + 1)
    }
}
```

The semantic fallback for host value extraction starts from that row array. The caller canonicalizes
the length-1 array and uses the typed canonical API to read the value:

```rust
let row = array.row(index)?;
let canonical = row.to_canonical(ctx)?;
let value = canonical.as_primitive::<i32>()?.value(0);
```

That fallback is correct, but it is not the required fast path. Many callers need repeated point
reads from scalar-compatible arrays, and forcing every read through singleton materialization would
regress compressed and wrapper encodings.

Add a fallible, stateful probe API for those callers:

```rust
pub struct ProbeOptions {
    // Exact fields intentionally left unspecified by this RFC.
}

pub trait ScalarProbe {
    fn dtype(&self) -> &DType;
    fn get(&mut self, index: usize) -> VortexResult<Scalar>;
}

impl ArrayRef {
    pub fn scalar_probe(
        &self,
        options: ProbeOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Box<dyn ScalarProbe>>;
}
```

Probe construction can fail for dtypes that are not scalar-compatible. It can also fail when an
encoding chooses not to support scalar probing directly, in which case the caller can use the
semantic row/canonical fallback if that cost is acceptable.

`ProbeOptions` is deliberately left undefined here. It should describe the caller's expected access
shape, not prescribe an implementation. For example, it may include an estimate of how many values
will be read, whether access is single, random, monotonic, or dense, and an approximate cache budget.
Encodings and wrapper arrays can use those hints to choose between lazy point access, page/chunk
caching, dictionary caching, cursor state, or canonicalizing a larger region up front.

The important distinction is that scalar probing is an optional scalar-compatible capability, not
the array row protocol. It preserves the current fast path where it matters while preventing
list-like and nested rows from re-entering the system as recursive scalar values.

### Serialization

`ConstantArray` serialization should serialize the length-1 child as a normal child array:

```text
new: buffers = [], children = [length-1 array]
```

New readers should continue to accept the legacy scalar-backed form during migration:

```text
legacy: buffers = [scalar_value], children = []
```

Deserializing the legacy form constructs a length-1 child array and then constructs the
`ConstantArray` wrapper. New writers should prefer the child-backed form for all constants,
including primitive constants. Compatibility writers may continue emitting the legacy scalar form for
old readers.

`ArrayLiteral` serialization requires array serialization. Portable expression serialization should
add an array-literal payload:

```proto
message ArrayLiteral {
  repeated string encoding_ids = 1;
  vortex.dtype.DType dtype = 2;
  uint64 len = 3;
  bytes serialized_array = 4;
}
```

On deserialization, Vortex decodes the array and constructs a new pointer-backed `ArrayLiteral`.
Pointer identity is process-local, so a deserialized expression is not pointer-equal to the original
in-memory expression even if it has the same contents.

### Device buffers

Device buffers should not be allowed inside `Scalar`.

Device buffers are allowed inside `ConstantArray` children and `ArrayLiteral` payloads because those
are arrays. In-memory execution can preserve device residency. Portable serialization copies buffers
to host in the same way array serialization does today.

### Impact on existing code

The current codebase uses `Scalar` for several different jobs. This RFC keeps the cheap host-value
jobs and removes the array-row jobs.

#### Row access and assertions

`OperationsVTable::scalar_at`, `ArrayRef::execute_scalar`, and Python/FFI `scalar_at` are the main
APIs that accidentally turn array rows into nested scalars. They should be deprecated and removed as
type-erased array operations.

Callers split into two groups:

- Callers that need a row use `row(index)` or `slice(index..index + 1)` and keep an `ArrayRef`.
  This includes constant compression, scalar-fn row execution, `case_when`, nested display,
  parquet-variant rows, nested tests, and fuzz/conformance row oracles.
- Callers that need repeated point reads from scalar-compatible arrays construct a `ScalarProbe`.
  This includes run-end ends and offsets, validity booleans, patch indices, encoded primitive
  values, decimal byte-part values, FSST codes, search/sort paths, and scalar-compatible test values.
  For one-off or unsupported reads, callers can fall back to length-1 row canonicalization and typed
  canonical extraction.

Tests comparing nested rows should use array equality on length-1 arrays. Tests comparing
scalar-compatible host values can compare values read through a probe or typed canonical arrays.

#### Constants and literals

`ConstantArray::new(value, len)` remains as an ergonomic constructor for host `Scalar` values, but it
immediately builds a length-1 child array and calls `ConstantArray::with_child`.

`ConstantArray::scalar()` and `ArrayRef::as_constant()` should not be the generic constant API.
Replacement APIs are:

- `ConstantArray::child()` for the source-of-truth length-1 value.
- typed helpers such as `primitive_value<T>()` for hot scalar-compatible kernels.
- array helpers for kernels that can operate on the length-1 child directly.

Expression literals split the same way:

- `lit(value)` stores a host `Scalar`, including restricted scalar-compatible struct literals.
- `lit_array(array)` stores an `ArrayRef` by pointer identity.
- list, fixed-size-list, variant, and nested struct literals use `ArrayLiteral`.

#### Scalar functions and compute kernels

Scalar functions continue to receive `ArrayRef` inputs of logical length `args.row_count()`.
Broadcasting is represented by `ConstantArray`, not by passing naked length-1 arrays into kernels.

Kernels that currently branch on `as_constant()` should inspect the constant child or use typed
helpers over that child. List-oriented functions such as `list_contains` should accept constant list
values through `ArrayLiteral` / `ConstantArray` and avoid list scalar extraction.

Missing intermediate constant folding is a separate optimization. The scalar-function API does not
need to become iterator-based to remove nested scalars.

#### Type-erased builders

Type-erased builders should be array-oriented:

```rust
trait ArrayBuilder {
    fn dtype(&self) -> &DType;
    fn extend(&mut self, values: ArrayRef) -> VortexResult<()>;
    fn append_nulls(&mut self, len: usize) -> VortexResult<()>;
    fn finish(self) -> VortexResult<ArrayRef>;
}
```

`append_row(row)` is `extend(row)` with `row.len() == 1`. Typed builders may keep host-value
conveniences such as `append_i32`, `append_utf8`, or `append_scalar(&Scalar)` for scalar-compatible
values, including restricted struct scalars where useful. The erased nested-builder protocol should
not depend on nested scalar values.

#### Aggregate partials

Aggregate partials should not be represented as one length-1 array per group. That would preserve
the array model but make hot aggregation state unnecessarily expensive.

Instead:

- in-memory aggregate partials are native Rust structs such as `MeanPartial { sum, count }` or
  `MinMaxPartial { min, max }`.
- batched partial exchange and serialization are array-shaped, for example a `StructArray` with one
  row per partial.
- combining partials reads typed child arrays from the partial array, not nested scalar rows.
- final aggregate output is an array; a one-group result is a length-1 array. Host scalar extraction
  from that result is a caller convenience, not the aggregate protocol.

Small struct `Scalar` values remain useful for expression literals, metadata, and compatibility, but
aggregate partial transport should not use `Scalar::struct_`.

#### Stats, encodings, and bindings

Scalar-valued stats remain `Scalar` when the stat value is scalar-compatible: min, max, sum,
counts, sortedness, and constantness. Nested stats should use child-array stats or array metadata,
not recursive tuple scalars.

Encoding metadata can keep `Scalar` when the metadata dtype is scalar-compatible, for example
FastLanes frame-of-reference references, decimal-byte-parts compare values, datetime-parts scalar
values, and FSST string/binary compare values. Encodings that carry nested fill values or nested row
values store one-row arrays.

FFI and Python scalar APIs remain for scalar-compatible values. Nested Python lists/dicts should
lower to arrays or `ArrayLiteral`, and nested row access should return a one-row array object.

## Migration Plan

### Phase 1: Add the new shapes

- Add `ConstantArray::with_child`.
- Add `ArrayLiteral` with pointer equality and hashing.
- Add `lit_array` and nested literal helpers.
- Add `ArrayRef::row(index)` as the universal row-access helper.
- Keep existing scalar APIs as compatibility wrappers where needed.

### Phase 2: Move expression literals

- Keep `lit(value: impl Into<Scalar>)` for scalar-compatible host values.
- Move list, fixed-size-list, variant, nested struct, and nested extension literals to
  `ArrayLiteral`.
- Replace private `ArrayExpr` with public `ArrayLiteral`.
- Add expression protobuf support for array literals.

### Phase 3: Move ConstantArray storage

- Change `ConstantArray` to store one length-1 child slot.
- Read both legacy scalar-backed and new child-backed constant encodings.
- Write child-backed constants by default.
- Audit constant compute kernels and preserve hot-path complexity.

### Phase 4: Remove nested scalar dependencies

- Deprecate and remove type-erased array `scalar_at` / `execute_scalar`.
- Move row call sites to one-row arrays.
- Move scalar-compatible point-read call sites to `ScalarProbe`, with typed canonical extraction as
  the fallback.
- Move nested stats away from recursive scalar values.
- Replace nested builder scalar append with array-oriented `extend`.
- Move aggregate partial exchange to arrays.
- Remove list, fixed-size-list, variant, and nested-extension scalar constructors.

## Compatibility

Existing scalar-backed constants and scalar literals remain readable during migration.

Existing readers will not understand the new child-backed `vortex.constant` serialized form if it is
encoded under the same encoding ID. They will fail because the constant encoding has zero buffers and
one child instead of one scalar buffer. This is a forward-compatibility limitation, not silent data
corruption.

Writers should expose a compatibility option:

- modern mode: write child-backed constants for every dtype
- legacy mode: write scalar-backed constants where possible, or canonicalize unsupported constants

Expression array literals require new expression protobuf support. Older readers will not understand
array literals.

Public Rust API compatibility should be managed in phases:

1. Add child-backed `ConstantArray` construction and `ArrayLiteral`.
2. Keep scalar-only helpers for existing callers.
3. Migrate internal kernels and expression constructors.
4. Deprecate ambiguous APIs such as `ConstantArray::scalar()`.
5. Remove list-like scalar constructors after downstream integrations have a migration path.

## Drawbacks

Primitive constants become slightly heavier. They now carry a length-1 child array instead of an
inline scalar. Hot constant kernels must be audited and benchmarked before this lands.

Pointer-based `ArrayLiteral` equality is intentionally weaker than value equality. It avoids
accidental expensive comparisons, but optimizers cannot assume two independently constructed
content-equal array literals are the same expression.

Removing type-erased array `scalar_at` requires a replacement fast path for primitive random-access
workloads. `ScalarProbe` fills that role, but it adds another capability API that encodings and
wrappers must implement or delegate carefully.

## Alternatives

### Scalar-or-row ConstantArray

We could represent constants as:

```rust
enum ConstantValue {
    Scalar(Scalar),
    Row(ArrayRef),
}
```

This reduces the cost of primitive constants, but it keeps two constant representations and preserves
branching pressure in kernels. This RFC chooses a single representation instead.

### Make Scalar array-backed

We could add `ScalarValue::Array(ArrayRef)` and represent complex scalars directly as singleton
arrays.

This makes `Scalar` no longer context-free. Scalar equality, hashing, display, validation, and
serialization would all need to understand arrays, and arrays may require execution or
device-to-host transfer.

### Remove struct scalars too

We could make `Scalar` valid only for null, bool, primitive, decimal, UTF-8, and binary values.

This is cleaner, but it makes small host records more expensive than necessary. Struct-like values
show up in expression literals, metadata, and compatibility APIs where a handful of host fields are
cheaper and clearer than a one-row `StructArray`. This RFC keeps restricted struct scalars while
forbidding their use as array rows, constant storage, erased builder input, or aggregate partial
transport.

### Only add ArrayLiteral

Adding `ArrayLiteral` alone would fix expression embedding, but `ConstantArray` would still force
nested values through recursive scalar representation. This would leave the execution problem mostly
unsolved.

### Keep unrestricted nested scalar values and optimize hot paths

We could add specialized list-scalar and struct-scalar storage to reduce allocation.

This may help some local cases, but it keeps recursive array-row materialization alive and still does
not solve device residency or array literal serialization. The useful subset is restricted struct
host records; list-like and variant values should use arrays.

## Prior Art

Apache Arrow distinguishes between scalars and arrays, and compute generally operates over arrays.
Broadcast scalar inputs are an execution concern, not a reason to represent nested arrays as
recursive scalar objects.

Database vector engines commonly represent constants as constant vectors. For nested values, a
constant vector can still point at child vectors rather than storing a recursive scalar tree.

Vortex already has most of this model: `ArrayRef`, `ConstantArray`, array serialization, validity,
and expression trees. The missing pieces are a child-only constant representation and a first-class
array literal expression.

## Unresolved Questions

- What benchmark thresholds should gate the primitive-constant wrapper cost?
- Should public `ArrayLiteral` construction allow row-count-length arrays, or should that be an
  internal-only constructor?
- How long should legacy nested `Scalar` constructors remain available?
- Should extension values with scalar-compatible storage be allowed as `Scalar`, or should all
  extension values stay array-backed for consistency?
- What is the exact public API migration for `ConstantArray::scalar()` and `ArrayRef::as_constant()`?
- What API should replace Python/FFI `scalar_at` for nested rows?

## Future Possibilities

Once nested values are array-backed everywhere, Vortex can add richer nested literal constructors in
Python, Java, C++, DuckDB, and DataFusion without routing through recursive scalar values.

`ArrayLiteral` can become the common mechanism for expression-local array payloads such as validity
arrays, dictionary values, compact lookup tables, and literal nested values.

A future device-aware expression transport could serialize `ArrayLiteral` payloads through external
buffer references instead of copying them into a portable byte payload.
