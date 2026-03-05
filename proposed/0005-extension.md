- Start Date: (2026-02-27)
- RFC PR: [vortex-data/rfcs#5](https://github.com/vortex-data/rfcs/pull/5)
- Tracking Issue: [vortex-data/vortex#6547](https://github.com/vortex-data/vortex/issues/6547)

## Summary

We would like to build a more robust system for extension data types (or `DType`s). This RFC
proposes a direction for extending the `ExtVTable` trait to support richer behavior (beyond
forwarding to the storage type), lays out the completed and in-progress work, and identifies the
open questions that remain.

## Motivation

A limitation of the current type system in Vortex is that we cannot easily add new logical types.
For example, the effort to add `FixedSizeList`
([vortex#4372](https://github.com/vortex-data/vortex/issues/4372)) and also change `List` to
`ListView` ([vortex#4699](https://github.com/vortex-data/vortex/issues/4699)) was very intrusive.
It is much easier to add wrappers around canonical types (treating the canonical dtype as a
"storage type") and implement some additional logic than to add a new variant to the `DType` enum.

### Storage DTypes

Extension types work by wrapping an existing canonical `DType`, called the **storage dtype**. The
storage dtype is itself a logical type (e.g., `Primitive`, `Struct`, `List`), and the extension
type is a logical wrapper over it that layers on additional semantics such as validation, display
formatting, and (eventually) custom compute logic.

For example, a `Timestamp` extension type has a `Primitive` storage dtype. Under the hood, a
timestamp array is just a primitive array of integers, but the extension layer knows that those
integers represent microseconds since the Unix epoch. Similarly, a `Union` extension type might
use `Struct` as its storage dtype, wrapping a struct of fields with union-specific dispatch logic.

This separation means that adding a new logical type does not require changes to the core canonical
type system, the compressor, or the I/O layer. Extension types get compression for free because
data is always read from and written to disk as the underlying storage dtype.

### Current State

Vortex provides an `Extension` variant of `DType` to help with this. Currently, implementors can
add a new extension type by defining an extension ID (for example, `vortex.time` or `vortex.date`)
and specifying a storage dtype. For example, the time extension types use a primitive storage dtype,
meaning they wrap the primitive scalars or primitive arrays with some extra logic on top (mostly
validating that the timestamps are valid).

We would like to add many more extension types. Some notable extension types (and their likely
storage types) include:

- **Matrix / Tensor**: This would be an extension over `FixedSizeList`, where dimensions correspond
  to levels of nesting. There are many open questions on the design of this, but that is out of
  scope of this RFC.
- **Union**: The sum type of an algebraic data type, like a Rust enum. One approach is to implement
  this with a type tag paired with a `Struct` (so `Struct { Primitive, Struct { types } }`).
  Vortex is well suited to represent this because it can compress each of the type field arrays
  independently, so we do not need to distinguish between a "Sparse" or "Dense" Union.
- **UUID**: Since this is a 128-bit number, we likely want to add `FixedSizeBinary`. This is out of
  scope for this RFC.

The issue with the current system is that it only forwards logic to the underlying storage type.
The only other behavior we support is serializing and pretty-printing extension arrays. This means
that we cannot define custom compute logic for extension types.

Take the time extension types as an example of where this limitation does not matter. If we want to
run a `compare` expression over a timestamp array, we just run the `compare` over the underlying
primitive array. For simple types like timestamps, this is sufficient (and this is what we do right
now). For types like Tensors (which are simply type aliases over `FixedSizeList`), this is also
fine.

However, for more complex types like UUID, Union, or JSON, forwarding to the storage type is likely
insufficient as these types need custom compute logic. Given that, we want a more robust
implementation path instead of wrapping `ExtensionArray` and performing significant internal
dispatch work.

## Design

### Background

[vortex#6081](https://github.com/vortex-data/vortex/pull/6081) introduced vtables (virtual tables,
or Rust unit structs with methods) for extension `DType`s. Each extension type (e.g., `Timestamp`)
now implements `ExtDTypeVTable`, which handles validation, serialization, and metadata.
The type-erased `ExtDTypeRef` carries this vtable with it inside `DType::Extension`.

There were a few blockers (detailed in the tracking issue
[vortex#6547](https://github.com/vortex-data/vortex/issues/6547)),
but now that those have been resolved, we can move forward.

### Proposed Design

Now that `vortex-scalar` and `vortex-dtype` have been merged into `vortex-array`, we can place
all extension logic (for types, scalars, and arrays) onto an `ExtVTable` (renamed from
`ExtDTypeVTable`).

It will look something like the following:

```rust
// Note: naming should be considered unstable.

/// The public API for defining new extension types.
///
/// This is the non-object-safe trait that plugin authors implement to define a new extension
/// type. It specifies the type's identity, metadata, serialization, and validation.
pub trait ExtVTable: 'static + Sized + Send + Sync + Clone + Debug + Eq + Hash {
    /// Associated type containing the deserialized metadata for this extension type.
    type Metadata: 'static + Send + Sync + Clone + Debug + Display + Eq + Hash;

    /// A native Rust value that represents a scalar of the extension type.
    ///
    /// The value only represents non-null values. We denote nullable values as `Option<Value>`.
    type NativeValue<'a>: Display;

    /// Returns the ID for this extension type.
    fn id(&self) -> ExtId;

    // Methods related to the extension `DType`.

    /// Serialize the metadata into a byte vector.
    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>>;

    /// Deserialize the metadata from a byte slice.
    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata>;

    /// Validate that the given storage type is compatible with this extension type.
    fn validate_dtype(&self, metadata: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()>;

    // Methods related to the extension scalar values.

    /// Validate the given storage value is compatible with the extension type.
    ///
    /// By default, this calls [`unpack_native()`](ExtVTable::unpack_native) and discards the
    /// result.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage [`ScalarValue`] is not compatible with the extension type.
    fn validate_scalar_value(
        &self,
        metadata: &Self::Metadata,
        storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        self.unpack_native(metadata, storage_dtype, storage_value)
            .map(|_| ())
    }

    /// Validate and unpack a native value from the storage [`ScalarValue`].
    ///
    /// Note that [`ExtVTable::validate_dtype()`] is always called first to validate the storage
    /// [`DType`], and the [`Scalar`](crate::scalar::Scalar) implementation will verify that the
    /// storage value is compatible with the storage dtype on construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage [`ScalarValue`] is not compatible with the extension type.
    fn unpack_native<'a>(
        &self,
        metadata: &'a Self::Metadata,
        storage_dtype: &'a DType,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>>;

    // `ArrayRef`

    fn validate_array(&self, metadata: &Self::Metadata, storage_array: &ArrayRef) -> VortexResult<()>;
    fn cast_array(&self, metadata: &Self::Metadata, array: &ArrayRef, target: &DType) -> VortexResult<ArrayRef> { ... }
    // Additional compute methods TBD.
}
```

Most of the implementation work will be making sure that `ExtDTypeRef` (which we pass around as the
`Extension` variant of `DType`) has the correct methods that access the internal, type-erased
`ExtVTable`.

Take extension scalars as an example. The only behavior we need from extension scalars is validating
that they have correct values, displaying them, and unpacking them into native types. So we added
these methods to `ExtDTypeRef`:

```rust
impl ExtDTypeRef {
    /// Formats an extension scalar value using the current dtype for metadata context.
    pub fn fmt_storage_value<'a>(
        &'a self,
        f: &mut fmt::Formatter<'_>,
        storage_value: &'a ScalarValue,
    ) -> fmt::Result { ... }

    /// Validates that the given storage scalar value is valid for this dtype.
    pub fn validate_storage_value(&self, storage_value: &ScalarValue) -> VortexResult<()> { ... }
}
```

**Open question**: What should the API for extension arrays look like? The answer will determine
what additional methods `ExtDTypeRef` needs beyond the scalar-related ones shown above.

## Compatibility

This should not break anything because extension types are mostly related to in-memory APIs (since
data is read from and written to disk as the storage type).

## Drawbacks

If forwarding to the storage type turns out to be sufficient for all extension types, the
additional vtable surface area adds complexity without clear benefit.

## Alternatives

We could have many `ExtensionArray` wrappers with custom logic. This approach would be clunky and
may not scale.

## Prior Art

Apache Arrow allows defining
[extension types](https://arrow.apache.org/docs/format/Columnar.html#format-metadata-extension-types)
and also provides a
[set of canonical extension types](https://arrow.apache.org/docs/format/CanonicalExtensions.html).

## Unresolved Questions

- Is forwarding to the storage type insufficient, and which extension types genuinely need custom
  compute logic?
- What should the `ExtVTable` API for extension arrays look like? What methods beyond
  `validate_array` are needed?
- How should compute expressions be defined and dispatched for extension types?

## Future Possibilities

If we can get extension types working well, we can add all of the following types:

- `DateTimeParts` (`Primitive`)
- Matrix (`FixedSizeList`)
- Tensor (`FixedSizeList`)
- UUID (Do we need to add `FixedSizeBinary` as a canonical type?)
- JSON (`UTF8`)
- PDX: https://arxiv.org/pdf/2503.04422v1 (`FixedSizeList`)
- Union
  - Sparse (`Struct { Primitive, Struct { types } }`)
  - Dense[^1]
- Map (`List<Struct { K, V }>`)
- Tags: See this
  [discussion](https://github.com/vortex-data/vortex/discussions/5772#discussioncomment-15279892),
  where we think we can represent this with (`ListView<Utf8>`)
- `Struct` but with protobuf-style field numbers (`Struct`)
- **NOT** Variant: see [RFC 0015 (Variant Type)](../accepted/0015-variant-type.md). Variant cannot
  be an extension type because there is no way to define a storage dtype when the schema is not
  known ahead of time for each row. Instead, Variant will have its own `DType` variant.
- And likely more.

[^1]:
    `Struct` doesn't work here because children can have different lengths, but what we could do
    is simply force the inner `Struct { types }` to hold `SparseArray` fields, which would
    effectively be the exact same but with the overhead of tracking indices for each of the child
    fields. In that case, it might just be better to always use a "sparse" union and let the
    compressor decide what to do.
