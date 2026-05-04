- Start Date: 2026-05-04
- Authors: @gatesn
- RFC PR: TBD

# Extension DTypes and Arrays

## Summary

This RFC proposes a redesign of Vortex extension dtypes and extension arrays. Extension arrays should remain a fully type-erased semantic wrapper around a storage array, but their array encoding id should be the extension dtype id rather than the generic `vortex.ext` id. Scalar-function behavior for extensions should be implemented through session-registered `execute_parent` kernels, with helper APIs for common storage-delegation behavior, instead of ad hoc hooks on `ExtVTable` or special cases in every builtin scalar function.

The proposal does not require a structural wire-format break. New readers should continue reading old `vortex.ext` arrays and should also read extension arrays encoded under their extension dtype id. A compatibility plugin should deserialize both forms into the same in-memory extension array representation.

## Motivation

Extensions are currently represented by a generic `vortex.ext` array encoding. That makes the extension wrapper itself the canonical array, but it hides the extension identity from array-level dispatch. This creates several problems:

- All extension arrays have the same array id, so parent kernels and scalar-function kernels cannot dispatch on the concrete extension family without inspecting dtype metadata manually.
- Common functions such as comparison, equality, cast, min/max, and hashing are tempted to delegate directly to storage, even when storage semantics are not necessarily extension semantics.
- Adding a hook such as `ExtVTable::cast` or `ExtVTable::compare` would not scale. Every new scalar function would need a new hook, or the dtype vtable would become a compute engine.
- Requiring every builtin scalar function to check an extension-storage flag would spread extension logic throughout scalar-function implementations.
- The current generic extension wrapper makes it difficult to distinguish custom extension semantics from physical storage optimization.

The design goal is to make extension types first-class semantic wrappers while preserving Vortex's plugin model:

- Extension dtypes describe identity, metadata, storage dtype, validation, and whether the extension is a nominal newtype or storage-preserving refinement.
- Extension arrays wrap storage arrays and expose the extension id as their array encoding id.
- Scalar-function behavior is provided by session-registered `execute_parent` kernels.
- Storage delegation is implemented as a reusable kernel helper, not as a required check in every scalar function.

## Design

### Type-Erased Extension Array

Vortex should keep one erased extension array type:

```rust
pub struct Extension {
    id: ArrayId,
}

pub type ExtensionArray = Array<Extension>;
```

There should not be an `ExtensionArray<E: ExtDType>`. The array remains fully type-erased and carries a `DType::Extension(ExtDTypeRef)`.

An extension array has exactly one child slot:

```text
ExtensionArray
└── storage: ArrayRef
```

The array dtype is the extension dtype:

```text
DType::Extension(ext_dtype)
```

The storage child dtype must match `ext_dtype.storage_dtype()`, modulo any existing nullability rules.

The important behavioral change is that the array encoding id is the extension dtype id:

```rust
array.encoding_id() == ext_dtype.id()
```

This gives the execution engine and kernel registry a concrete semantic id such as `vortex.uuid`, `vortex.date`, or `vortex.json`, while still using one erased Rust array implementation.

### Extension Array Construction

The public constructors should be split by validation level:

```rust
impl ExtensionArray {
    pub fn try_new(ext_dtype: ExtDTypeRef, storage: ArrayRef) -> VortexResult<Self>;

    pub fn try_new_validated(
        ext_dtype: ExtDTypeRef,
        storage: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self>;

    pub unsafe fn new_unchecked(ext_dtype: ExtDTypeRef, storage: ArrayRef) -> Self;
}
```

`try_new` performs structural validation only:

- dtype is `DType::Extension(ext_dtype)`;
- storage dtype matches `ext_dtype.storage_dtype()`;
- length matches;
- extension metadata and storage dtype are valid for the extension type.

`try_new_validated` performs structural validation and value validation. It may require an `ExecutionCtx` because validating values may require execution, canonicalization, decoding, scalar functions, or other runtime behavior.

`new_unchecked` is reserved for trusted internal paths that have already established the invariants.

### Extension Kind

Extension dtype vtables should expose a coarse classification:

```rust
pub enum ExtensionKind {
    Newtype,
    Refinement,
}

pub trait ExtVTable {
    fn kind(&self) -> ExtensionKind {
        ExtensionKind::Newtype
    }
}
```

This is policy metadata, not a custom execution mechanism. It gives Vortex a conservative default for generated storage-delegate kernels.

`Newtype` means a nominal semantic type over storage. UUID over fixed bytes and UserId over `u64` are examples. The default policy is conservative: do not assume storage operations have extension semantics. Newtypes must register session `execute_parent` kernels or explicit storage-delegate kernels for operations they support.

`Refinement` means the extension represents a subset or refinement of the storage type where storage equality and value identity are still the extension's equality and value identity. Utf8-over-Binary, non-empty-Utf8, and fixed-size-list-as-list are examples.

Refinements may get default generated storage-delegate kernels for operations that only observe or preserve existing values. Equality, inequality, hash, filter, take, slice, dictionary decode, and min/max are candidates when the storage operation has the same semantics. Transforming operations, such as arithmetic, casts into the refinement, string transforms, parsing, or functions that construct new values, still need explicit kernels or validation-aware wrapping.

The kind should not replace explicit session kernels. It is a default-policy input for the storage-delegate helper. If an extension's semantics differ from storage for a particular operation, the extension should be a `Newtype` or should avoid registering that default delegate.

### Session Execute-Parent Kernels

Vortex should move `execute_parent` kernels into the session. This is the extension point for extension-authored scalar-function behavior and storage delegation.

This is not a new scalar-function execution path. Today many scalar functions already have operation-specific kernels, such as `CastKernel`, `CompareKernel`, `LikeKernel`, and `FillNullKernel`, that are adapted into `ExecuteParentKernel` so a child encoding can execute its `ScalarFnArray` parent. This RFC proposes moving those `execute_parent` kernels from static child-vtable registration into a session registry.

The registry should be keyed by parent id and child id:

```rust
pub type ParentKernelKey = (Id, ArrayId);

pub trait SessionExecuteParentKernel: Send + Sync {
    fn parent_id(&self) -> Id;
    fn child_id(&self) -> ArrayId;

    fn execute_parent(
        &self,
        child: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}
```

The exact Rust signature can be refined during implementation. The important point is that the session stores erased `execute_parent` kernels. Existing typed `ExecuteParentKernel<V>` implementations can remain as an implementation convenience and be adapted into the erased session form.

Parent id lookup should follow these rules:

- For ordinary array parents, `parent_id = parent.encoding_id()`.
- For `ScalarFnArray` parents, `parent_id = parent.scalar_fn().id()`, not the generic scalar-function array id.
- The child id is always `child.encoding_id()`.

After this RFC, an extension array's child id is its extension dtype id. That means scalar-function extension behavior can be registered as ordinary parent kernels:

```text
(parent_id = vortex.binary, child_id = vortex.uuid)
(parent_id = vortex.cast, child_id = vortex.timestamp)
(parent_id = vortex.get_item, child_id = vortex.json)
```

Execution order should be:

```text
1. For each child slot, try matching session execute_parent kernels.
2. During migration, fall back to the child's static execute_parent implementation.
3. If no parent kernel applies, execute the parent normally.
```

This centralizes extension dispatch in the existing parent-kernel mechanism. Individual builtin scalar functions do not all need to remember to check extension-specific flags.

### Custom Extension Kernels

Extensions that need custom semantics register session `execute_parent` kernels during plugin initialization or default-session construction.

Examples:

```text
(vortex.binary, vortex.uuid) -> UUID equality/comparison behavior
(vortex.cast, vortex.timestamp) -> timestamp unit/timezone cast behavior
(vortex.get_item, vortex.json) -> JSON path extraction behavior
```

The kernel owns the full execution plan. It may inspect scalar-function options, argument dtypes, extension metadata, and storage dtypes. It may choose to execute directly, rewrite to another scalar function, unwrap storage, wrap results, validate results, or decline by returning `Ok(None)`.

This avoids putting compute behavior on `ExtVTable`.

### Storage-Delegate Kernel Helper

Many extension functions only need to delegate to storage. This should be easy to register, but still implemented as ordinary session `execute_parent` kernels.

Vortex should provide a helper/builder that creates session `execute_parent` kernels:

```rust
session.execute_parent_kernels().register(
    StorageDelegateExecuteParentKernel::new(Binary.id())
        .for_extension(Uuid.id())
        .when_options(|options| matches_binary_operator(options, [Eq, NotEq]))
        .unwrap_args([0, 1])
        .result(ResultPlan::KeepStorageResult),
);
```

For an extension author, this should be a small amount of code:

```rust
register_storage_delegate(
    Binary.id(),
    Uuid.id(),
    StorageDelegatePlan::new()
        .operators([Operator::Eq, Operator::NotEq])
        .unwrap_all_extension_args()
        .keep_result(),
);
```

The exact API can be refined during implementation. The important properties are:

- it registers an `execute_parent` kernel in the session;
- it is not a method on `ExtVTable`;
- it does not require every scalar function to check a flag;
- it can express argument unwrapping, output wrapping, validation, and option matching.

Storage delegation plans should support at least:

```rust
pub enum ResultPlan {
    KeepStorageResult,
    WrapAsInputExtension { input: usize, validation: ValidationPolicy },
    WrapAsTargetExtension { validation: ValidationPolicy },
}

pub enum ValidationPolicy {
    None,
    Structural,
    Full,
}
```

Examples:

```text
UUID equality:
  unwrap UUID args -> compare storage -> Bool

Date ordering:
  unwrap Date args -> compare integer storage -> Bool

fill_null(UUID, UUID):
  unwrap both args -> fill_null storage -> wrap as UUID

cast(UUID -> storage):
  unwrap source -> return storage or casted storage

cast(storage -> UUID):
  cast to UUID storage dtype -> wrap as UUID with validation
```

### What ExtVTable Owns

`ExtVTable` should remain a type-description interface:

```rust
pub trait ExtVTable {
    type Metadata;

    fn id(&self) -> ExtId;
    fn kind(&self) -> ExtensionKind;

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>>;
    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata>;

    fn validate_dtype(dtype: &ExtDType<Self>) -> VortexResult<()>;
    fn validate_scalar_value(dtype: &ExtDType<Self>, storage_value: &ScalarValue)
        -> VortexResult<()>;

    fn can_coerce_from(dtype: &ExtDType<Self>, other: &DType) -> bool;
    fn can_coerce_to(dtype: &ExtDType<Self>, other: &DType) -> bool;
    fn least_supertype(dtype: &ExtDType<Self>, other: &DType) -> Option<DType>;
}
```

`ExtVTable` should not contain:

```rust
fn delegate_scalar_fn(...)
fn execute_scalar_fn(...)
fn register_storage_delegates(...)
```

Registration should happen through plugin initialization or default-session construction. This keeps dtype descriptors separate from compute behavior.

### Wrap and Unwrap Scalar Functions

Vortex should add explicit scalar functions:

```text
extension_unwrap(x) -> storage
extension_wrap(storage, ext_dtype, validation_policy) -> extension
```

`extension_unwrap` strips extension semantics and returns the storage array. This is an explicit escape hatch for users, kernels, and optimizers.

`extension_wrap` constructs an extension array from storage. It should support structural-only and full validation modes. Internal callers may use unchecked construction after proving invariants.

Useful identities:

```text
extension_wrap(extension_unwrap(x), dtype(x)) == x
extension_unwrap(extension_wrap(s, E)) == s
```

The first identity assumes the original extension dtype is preserved. The second assumes validation succeeds.

### Serde and Compatibility

The wire format does not need a structural change.

Today, an extension array may be serialized as:

```text
encoding_id = vortex.ext
dtype = Extension(vortex.uuid, storage_dtype, metadata)
children[0] = storage array
```

New writers may serialize the same logical array as:

```text
encoding_id = vortex.uuid
dtype = Extension(vortex.uuid, storage_dtype, metadata)
children[0] = storage array
```

Both forms should deserialize into the same in-memory `ExtensionArray`, whose runtime `encoding_id()` is `vortex.uuid`.

Implementation requirements:

- Register an extension array plugin under `vortex.ext` for old files.
- Register the same extension array plugin under each known extension dtype id.
- During deserialization, if the array encoding id is unknown but the dtype is `DType::Extension(ext)` and `encoding_id == ext.id()`, use the extension array plugin before falling back to a foreign array.
- After deserialization, the decoded array may have a different encoding id from the serialized node when reading legacy `vortex.ext`. The compatibility plugin must declare that this is supported.

If old-reader compatibility is required for newly written files, add a writer option:

```rust
pub enum ExtensionEncodingMode {
    ExtensionId,
    LegacyVortexExt,
}
```

The default should be `ExtensionId` once the new readers are available.

### Canonicalization

An extension array should not be treated as canonical merely because the wrapper itself has no buffers. The storage child is the physical representation.

Execution of an extension array should require the storage slot to be executed/canonicalized, then rewrap the executed storage child with the same extension dtype. This is slot-level progress through the existing execution model, not a general recursive canonicalization rule.

The invariant becomes:

```text
canonical extension array = extension wrapper over canonical storage
```

### Metadata Validation

Extension metadata deserialization must be total over byte input. Malformed metadata should return `VortexError`, never panic.

This is especially important for built-in extension dtypes because file readers will deserialize untrusted bytes.

## Compatibility

This proposal does not require adding fields to the dtype or array wire format.

It does change the preferred encoding id for extension arrays from `vortex.ext` to the extension dtype id. Backward compatibility is maintained by registering a compatibility plugin for `vortex.ext` and deserializing old arrays into the new in-memory representation.

Forward compatibility depends on reader behavior:

- New readers can read both `vortex.ext` and extension-id encoded arrays.
- Old readers can read newly written files only if writers use the legacy `vortex.ext` mode.

Public Rust APIs will change around extension array construction and extension plugin registration. The migration path is:

- replace generic `vortex.ext` construction with `ExtensionArray::try_new(ext_dtype, storage)`;
- register extension scalar behavior as session `execute_parent` kernels;
- use storage-delegate helper kernels for common storage-transparent operations;
- use `extension_unwrap` and `extension_wrap` for explicit representation access.

Performance should improve for extension-specific dispatch because the array encoding id now carries the concrete extension id. There is some additional session-kernel lookup cost during parent-kernel execution, but this is centralized and should be small compared to actual array execution.

## Drawbacks

This adds a session `execute_parent` kernel registry and a storage-delegate helper API. That is more machinery than direct methods on `ExtVTable`.

The design also changes the meaning of extension array encoding ids. Although this is not a structural wire-format break, it requires compatibility behavior during serde and careful migration of tests and registry setup.

The storage-delegate helper must be expressive enough for common cases without becoming a second scalar-function implementation framework. Complex extension semantics should use custom session `execute_parent` kernels instead of stretching the helper API.

## Alternatives

### Keep `vortex.ext` as the Runtime Array ID

This is the smallest change, but it keeps extension identity hidden from array-kernel dispatch. Every extension-aware kernel would need to inspect dtype metadata manually.

### `ExtensionArray<E: ExtDType>`

A generic extension array gives strong static typing, but Vortex needs a fully type-erased extension array because extension dtypes are registered dynamically and stored in `DType::Extension(ExtDTypeRef)`.

### Let Every Physical Array Support Extension DTypes

For example, allow `PrimitiveArray` to carry `DType::Extension`. This would make extension dtypes an overlay on every physical encoding. It is too invasive and would push extension-specific logic into every canonical and compressed array implementation.

### Add `ExtVTable::delegate_scalar_fn`

This looks simple but creates a bad contract. Every scalar function would need to remember to ask the extension dtype whether storage delegation is allowed. A boolean is also not expressive enough to describe which arguments are unwrapped, whether outputs are wrapped, whether validation is required, or which scalar-function options are allowed.

### Add `ExtVTable::execute_scalar_fn`

This makes the dtype vtable a compute engine and creates arbitration problems for multi-argument functions. For `binary(lhs_ext, rhs_ext)`, it is unclear whether the left extension, right extension, or scalar function owns execution. Session `execute_parent` kernels are a cleaner extension point.

### Add `ExtVTable::register_storage_delegates`

This RFC explicitly rejects registration methods on `ExtVTable`. Registration should happen during plugin initialization or default-session construction. The dtype vtable should describe the type; it should not install compute behavior.

## Prior Art

Apache Arrow extension types store a regular Arrow storage type plus extension metadata on the field. The storage array remains a normal Arrow array. Vortex should preserve this separation between logical extension type and physical storage representation while giving extensions better runtime dispatch. See the Arrow extension type documentation: <https://arrow.apache.org/docs/format/Columnar.html#extension-types>.

Postgres domains are base types with constraints. They are useful prior art for refinement-like types, although this RFC does not model domains as a separate extension kind. Postgres also has the concept of binary-coercible casts through `CREATE CAST ... WITHOUT FUNCTION`, where no conversion is required because the source and target have the same internal representation. That is related to storage delegation, but Vortex should express it through registered kernels rather than a closed set of global flags. See <https://www.postgresql.org/docs/current/sql-createcast.html> and <https://www.postgresql.org/docs/current/sql-createdomain.html>.

DuckDB and Postgres both distinguish type identity from function/operator implementations. Operators and casts are registered behavior, not hard-coded methods on the type descriptor. Vortex should follow that separation by putting extension scalar behavior in session `execute_parent` kernels.

## Unresolved Questions

- What should the exact erased session `execute_parent` kernel trait look like?
- Should session `execute_parent` kernels be ordered by registration order, specificity, or explicit priority?
- Should generated storage-delegate kernels be stored in the same registry as custom session kernels, or in a separate registry checked by the same dispatcher?
- How should session `execute_parent` dispatch handle multi-extension arguments when multiple kernels match?
- What should the exact `extension_wrap` validation-policy API be?
- Should new writers default to extension-id encoding immediately, or should there be a transition period where `vortex.ext` remains the default?
- Which built-in extension dtypes should register storage-delegate kernels initially?

Unions are out of scope for this RFC.

Adding `FixedSizeBinary` is also out of scope. It may be a good storage dtype for UUID-like extensions, but it should be considered separately.

## Future Possibilities

The same session-kernel mechanism can eventually replace more static `execute_parent` implementations beyond scalar functions. Session `reduce_parent` already exists in a limited form; aligning both registries is a natural follow-on.

The extension descriptor could eventually include richer documentation metadata for external systems, such as Arrow extension mappings, SQL type names, and display/formatting preferences.

The storage-delegate helper may grow convenience constructors for common patterns such as equality-only newtypes and value-preserving refinements.
