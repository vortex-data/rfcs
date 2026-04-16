- Start Date: 2026-03-04
- Authors: @connortsui20
- RFC PR: [vortex-data/rfcs#24](https://github.com/vortex-data/rfcs/pull/24)

# Fixed-shape Tensor Extension

## Summary

We would like to add a `FixedShapeTensor` type to Vortex as an extension type backed by
`FixedSizeList`. This RFC proposes the design of a fixed-shape tensor with contiguous backing
memory.

## Motivation

### Tensors in the wild

Tensors are multi-dimensional (n-dimensional) arrays that generalize vectors (1D) and matrices (2D)
to arbitrary dimensions. They are quite common in ML/AI and scientific computing applications. To
name just a few examples:

- Image or video data stored as `height x width x channels`
- Multi-dimensional sensor or time-series data
- Embedding vectors from language models and recommendation systems

### Fixed-shape tensors in Vortex

In the current version of Vortex, there are two ways to represent fixed-shape tensors using the
`FixedSizeList` `DType`, and neither seems satisfactory.

The simplest approach is to flatten the tensor into a single `FixedSizeList<n>` whose size is the
product of all dimensions (this is what Apache Arrow does). However, this discards shape information
entirely: a `2x3` matrix and a `3x2` matrix would both become `FixedSizeList<6>`. Shape metadata
must be stored separately, and any dimension-aware operation (slicing along an axis, transposing,
etc.) reduces to manual index arithmetic with no type-level guarantees.

The alternative is to nest `FixedSizeList` types, e.g., `FixedSizeList<FixedSizeList<n>, m>` for a
matrix. This preserves some structure, but becomes unwieldy for higher-dimensional tensors.
Axis-specific slicing or indexing on individual tensors (tensor scalars, not tensor arrays) would
require custom expressions aware of the specific nesting depth, rather than operating on a single,
uniform tensor type.

Additionally, reshaping requires restructuring the entire nested type, and operations like
transposes would be difficult to implement correctly.

Beyond these structural issues, neither approach stores shape and stride metadata explicitly, making
interoperability with external tensor libraries (NumPy, PyTorch, etc.) that expect contiguous memory
with this metadata awkward.

Thus, we propose a dedicated extension type that encapsulates tensor semantics (shape, strides,
dimension-aware operations) on top of contiguous, row-major (C-style) backing memory.

## Design

Since the design of extension types has not been fully solved yet (see
[RFC #0005](https://github.com/vortex-data/rfcs/pull/5)), the complete design of tensors cannot be
fully described here. However, we do know enough that we can present the general idea here.

### Storage Type

Extension types in Vortex require defining a canonical storage type that represents what the
extension array looks like when it is canonicalized. For fixed-shape tensors, we will want this
storage type to be a `FixedSizeList<p, s>`, where `p` is a primitive type (like `u8`, `f64`, etc.),
and where `s` is the product of all dimensions of the tensor.

For example, if we want to represent a tensor of `i32` with dimensions `[2, 3, 4]`, the storage type
for this tensor would be `FixedSizeList<i32, 24>` since `2 x 3 x 4 = 24`.

This is equivalent to the design of Arrow's canonical Fixed Shape Tensor extension type. For
discussion on why we choose not to represent tensors as nested FSLs (for example
`FixedSizeList<FixedSizeList<FixedSizeList<i32, 4>, 3>, 2>`), see the [alternatives](#alternatives)
section.

### Element Type

We restrict tensor element types to `Primitive`. Tensors are fundamentally about dense numeric
computation, and operations like transpose, reshape, and slicing rely on uniform, fixed-size
elements whose offsets are computable from strides.

Variable-size types (like strings) would break this model entirely. `Bool` is excluded because
Vortex bit-packs boolean arrays, which conflicts with byte-level stride arithmetic. `Decimal` is
excluded because there are no fast implementations of tensor operations (e.g., matmul) for
fixed-point types. This matches PyTorch, which also restricts tensors to floating-point and integer
primitive types.

We could allow more element types in the future if a compelling use case arises, but it should
remain a very low priority.

### Validity

Nullability exists only at the tensor level: within a tensor array, an individual tensor may be
null, but elements within a tensor may not be. This is because tensor operations like matmul cannot
be efficiently implemented over nullable elements, and most tensor libraries (e.g., PyTorch) do not
support per-element nulls either.

Since the storage type is `FixedSizeList`, the validity of the tensor array is inherited from the
`FixedSizeList`'s own validity bitmap (one bit per tensor, not per element).

This is a restriction we can relax in the future if a compelling use case arises.

### Metadata

Theoretically, we only need the dimensions of the tensor to have a useful Tensor type. However, we
likely also want two other pieces of information, the dimension names and the permutation order,
which aligns with Arrow's [Fixed Shape Tensor](https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor)
canonical extension type.

Here is what the metadata of the `FixedShapeTensor` extension type in Vortex might look like (in
Rust):

```rust
/// Metadata for a `FixedShapeTensor` extension type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FixedShapeTensorMetadata {
    /// The logical shape of the tensor.
    ///
    /// `logical_shape[i]` is the size of the `i`-th logical dimension. When a `permutation` is
    /// present, the physical shape (i.e., the row-major memory layout) is derived as
    /// `physical_shape[permutation[i]] = logical_shape[i]`.
    ///
    /// May be empty (0D scalar tensor) or contain dimensions of size 0 (degenerate tensor).
    logical_shape: Vec<usize>,

    /// Optional names for each logical dimension. Each name corresponds to an entry in
    /// `logical_shape`.
    ///
    /// If names exist, there must be an equal number of names to logical dimensions.
    dim_names: Option<Vec<String>>,

    /// The permutation of the tensor's dimensions. `permutation[i]` is the physical dimension
    /// index that logical dimension `i` maps to.
    ///
    /// If this is `None`, then the logical and physical layouts are identical, equivalent to
    /// the identity permutation `[0, 1, ..., N-1]`.
    permutation: Option<Vec<usize>>,
}
```

Note that this metadata would store the _logical_ shape of the tensor, not the physical shape. For
more info on this, see the [physical vs. logical shape](#physical-vs-logical-shape) discussion.

### Stride

The stride of a tensor defines the number of elements to skip in memory to move one step along each
dimension. Rather than storing strides explicitly as metadata, we can efficiently derive them from
the shape and permutation. This is possible because the backing memory is always contiguous.

For a row-major tensor with shape `d = [d_0, d_1, ..., d_{n-1}]` and no permutation, the strides
are:

```
stride[n-1] = 1  (innermost dimension always has stride 1)
stride[i]   = d[i+1] * stride[i+1]
stride[i]   = d[i+1] * d[i+2] * ... * d[n-1]
```

For example, a tensor with shape `[2, 3, 4]` and no permutation has strides `[12, 4, 1]`: moving one
step along dimension 0 skips 12 elements, along dimension 1 skips 4, and along dimension 2 skips 1.
The element at index `[i, j, k]` is located at memory offset `12*i + 4*j + k`.

### Physical vs. logical shape

When a permutation is present, stride derivation depends on whether `logical_shape` stores logical
or physical dimensions. We lean towards storing **logical** dimensions (matching NumPy/PyTorch and
Vortex's logical type system), though this is not yet finalized (see
[unresolved questions](#unresolved-questions)).

With logical shape, we first invert the permutation to recover the physical shape
(`physical_shape[perm[i]] = logical_shape[i]`), compute row-major strides over that, then map them
back to logical order.

For example, with logical shape `[4, 2, 3]` and permutation `[2, 0, 1]`: the physical shape is
`[2, 3, 4]`, physical strides are `[12, 4, 1]`, and logical strides are `[1, 12, 4]`.

Alternatively, if we stored **physical** dimensions instead (matching Arrow's convention), stride
derivation would be simpler: compute row-major strides directly over the stored shape, then permute
them (`logical_stride[i] = physical_stride[perm[i]]`). For the same tensor with physical shape
`[2, 3, 4]` and permutation `[2, 0, 1]`, the result is the same: `[1, 12, 4]`.

In either case, logical strides are always a permutation of the physical strides. The cost of
conversion between conventions is a cheap O(ndim) permutation at the boundary, so the difference is
more about convention than performance.

Physical shape favors Arrow compatibility and simpler stride math. Logical shape favors
NumPy/PyTorch compatibility and is arguably more intuitive for users since Vortex has a logical type
system.

### Conversions

#### Arrow

Our storage type and metadata are designed to closely match Arrow's Fixed Shape Tensor canonical
extension type. The `FixedSizeList` backing buffer, dimension names, and permutation pass through
unchanged, making the data conversion itself zero-copy (for tensors with at least one dimension).

Arrow stores `shape` as **physical** (the dimensions of the row-major layout). Since we lean towards
storing logical shape in Vortex, Arrow conversion will require a cheap O(ndim) scatter:
`arrow_shape[perm[i]] = vortex_shape[i]`. If we instead adopt physical shape, the field would pass
through directly.

#### NumPy and PyTorch

Libraries like NumPy and PyTorch store strides as an independent, first-class field on their tensor
objects. This allows them to represent non-contiguous views of memory.

For example, slicing every other row of a matrix produces a view with a doubled row stride, sharing
memory with the original without copying. However, this means that non-contiguous tensors can appear
anywhere, and kernels must handle arbitrary stride patterns. PyTorch supposedly requires many
operations to call `.contiguous()` before proceeding.

NumPy and PyTorch store `shape` as **logical** (the dimensions the user indexes with). Since we lean
towards storing logical shape in Vortex, the shape field would pass through unchanged. If we instead
adopt physical shape, a cheap O(ndim) permutation would be needed at the boundary.

Since Vortex fixed-shape tensors always have dense backing memory, we can always zero-copy _to_
NumPy and PyTorch by passing the buffer pointer, logical shape, and logical strides. A permuted
Vortex tensor will appear as a non-C-contiguous view in these libraries, which they handle natively.

Going the other direction, we can zero-copy _from_ any NumPy/PyTorch tensor whose memory is dense
(no gaps), even if it is not C-contiguous. A Fortran-order or otherwise permuted tensor can be
represented by deriving the appropriate permutation from its strides. Only tensors with actual
memory gaps (e.g., strided slices like `arr[::2]`) require a copy.

Our proposed design for Vortex `FixedShapeTensor` will handle operations differently than the
Python libraries. Rather than mutating strides to create non-contiguous views, operations like
slicing, indexing, and reordering dimensions would be expressed as lazy `Expression`s over the
tensor.

These expressions describe the operation without materializing it, and when evaluated, they produce
a new tensor with dense backing memory. This fits naturally into Vortex's existing lazy compute
system, where compute is deferred and composed rather than eagerly applied.

The exact mechanism for defining expressions over extension types is still being designed (see
[RFC #0005](https://github.com/vortex-data/rfcs/pull/5)), but the intent is that tensor-specific
operations like axis slicing, indexing, and reshaping would be custom expressions registered for the
tensor extension type.

### Edge Cases: 0D and Size-0 Dimensions

We will support two edge cases that arise naturally from the tensor model. Recall that the number of
elements in a tensor is the product of its shape dimensions, and that the
[empty product](https://en.wikipedia.org/wiki/Empty_product) is 1 (the multiplicative identity).

#### 0-dimensional tensors

0D tensors have an empty shape `[]` and contain exactly one element (since the product of no
dimensions is 1). These represent scalar values wrapped in the tensor type. The storage type is
`FixedSizeList<p, 1>` (semantically equivalent to a flat `PrimitiveArray`).

#### Size-0 dimensions

Shapes may contain dimensions of size 0 (e.g., `[3, 0, 4]`), which produce tensors with zero
elements (since the product includes a 0 factor). The storage type is a degenerate
`FixedSizeList<p, 0>`, which Vortex already handles well.

#### Compatibility

Both NumPy and PyTorch support these cases. NumPy fully supports 0D arrays with shape `()`, and
dimensions of size 0 are valid (e.g., `np.zeros((3, 0, 4))`). PyTorch supports 0D tensors since
v0.4.0 and also allows size-0 dimensions.

Arrow's Fixed Shape Tensor spec, however, requires at least one dimension (`ndim >= 1`), so 0D
tensors would need special handling during Arrow conversion (e.g., returning an error or unwrapping
to a scalar).

### Compression

Since the storage type is `FixedSizeList` over numeric types, Vortex's existing encodings (like ALP,
FastLanes, etc.) will be applied to the flattened primitive buffer transparently.

However, there may be tensor-specific compression opportunities we could take advantage of. We will
leave this as an open question.

### Scalar Representation

Once we add the `ScalarValue::Array` variant (see tracking issue
[vortex#6771](https://github.com/vortex-data/vortex/issues/6771)), we can easily pass around
fixed-shape tensors as `ArrayRef` scalars as well as lazily computed slices.

The `ExtVTable` also requires specifying an associated `NativeValue<'a>` Rust type that an extension
scalar can be unpacked into. We will want a `NativeFixedShapeTensor<'a>` type that references the
backing memory of the tensor, and we can add useful operations to that type.

## Compatibility

Since this is a new type built on an existing canonical type (`FixedSizeList`), there should be no
compatibility concerns.

## Drawbacks

- **Fixed shape only**: This design only supports tensors where every element in the array has the
  same shape. Variable-shape tensors (ragged arrays) are out of scope and would require a different
  type entirely.
- **Yet another crate**: We will likely implement this in a `vortex-tensor` crate, which means
  even more surface area than we already have.

## Alternatives

### Nested `FixedSizeList`

Rather than a flat `FixedSizeList` with metadata, we could represent tensors as nested
`FixedSizeList` types (e.g., `FixedSizeList<FixedSizeList<FixedSizeList<i32, 4>, 3>, 2>` for a
`[2, 3, 4]` tensor). This has several disadvantages:

- Each nesting level introduces its own validity bitmap, even though sub-dimensional nullability is
  not meaningful for tensors. This wastes space and complicates null-handling logic.
- This does not match Arrow's canonical Fixed Shape Tensor type, making zero-copy conversion
  impossible.
- Expressions would need to be aware of the nesting depth, and operations like transpose or reshape
  would require restructuring the type itself rather than updating metadata.

### Do nothing

Users could continue to use `FixedSizeList` directly with out-of-band shape metadata. This works
for simple storage, but as discussed in the [motivation](#motivation), it provides no type-level
support for tensor operations and makes interoperability with tensor libraries awkward.

## Prior Art

_Note: This section was Claude-researched._

### Columnar formats

- **Apache Arrow** defines a
  [Fixed Shape Tensor](https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor)
  canonical extension type. Our design closely follows Arrow's approach: a flat `FixedSizeList`
  storage type with shape, dimension names, and permutation metadata. Arrow also defines a
  [Variable Shape Tensor](https://arrow.apache.org/docs/format/CanonicalExtensions.html#variable-shape-tensor)
  extension type for ragged tensors, which could inform future work.
- **Lance** delegates entirely to Arrow's type system, including extension types. Arrow extension
  metadata (and therefore tensor metadata) is preserved end-to-end through Lance's storage layer,
  which validates the approach of building tensor semantics as an extension on top of `FixedSizeList`
  storage.
- **Parquet** has no native `FixedSizeList` logical type. Arrow's `FixedSizeList` is stored as a
  regular `LIST` in Parquet, which adds conversion overhead via repetition levels. There is active
  discussion about introducing `FixedSizeList` as a Parquet logical type, partly motivated by
  tensor and embedding workloads.

### Database systems

- **DuckDB** has a native `ARRAY` type (fixed-size list) but no dedicated tensor type. Community
  discussions have proposed adding one, noting that nested `ARRAY` types can simulate
  multi-dimensional arrays but lack tensor-specific operations.
- **DataFusion** uses Arrow's type system directly and has no dedicated tensor type. There is open
  discussion about a logical type layer that could support extension types as first-class citizens.

### Tensor libraries

- **NumPy** and **PyTorch** both represent tensors as contiguous (or non-contiguous) memory with
  shape and stride metadata. Our design is a subset of this model — we always require contiguous
  memory and derive strides from shape and permutation, as discussed in the
  [conversions](#conversions) section.
- **[xarray](https://docs.xarray.dev/en/stable/)** extends NumPy with named dimensions and
  coordinate labels. Its
  [data model](https://docs.xarray.dev/en/stable/user-guide/terminology.html) attaches names to each
  dimension and associates "coordinate" arrays along those dimensions (e.g., latitude and longitude
  values for the rows and columns of a temperature matrix). Our `dim_names` metadata is a subset of
  xarray's model; coordinate arrays could be a future extension.
- **[ndindex](https://quansight-labs.github.io/ndindex/index.html)** is a Python library that
  provides a unified interface for representing and manipulating NumPy array indices (slices,
  integers, ellipses, boolean arrays, etc.). It supports operations like canonicalization, shape
  inference, and re-indexing onto array chunks. We will want to implement tensor compute expressions
  in Vortex that are similar to the operations ndindex provides — for example, computing the result
  shape of a slice or translating a logical index into a physical offset.

### Academic work

- **TACO (Tensor Algebra Compiler)** separates the tensor storage format from the tensor program.
  Each dimension can independently be specified as dense or sparse, and dimensions can be reordered.
  The Vortex approach of storing tensors as flat contiguous memory with a permutation is one
  specific point in TACO's format space (all dimensions dense, with a specific dimension ordering).

## Unresolved Questions

- Should `logical_shape` store logical dimensions (matching NumPy/PyTorch) or physical dimensions
  (matching Arrow)? The RFC currently leans towards logical shape, but this is not finalized. See
  the [physical vs. logical shape](#physical-vs-logical-shape) discussion in the stride section.
- Are two tensors with different permutations but the same logical values considered equal? This
  affects deduplication and comparisons. The type metadata might be different but the entire tensor
  value might be equal, so it seems strange to say that they are not actually equal?
- Are there potential tensor-specific compression schemes we can take advantage of?

## Future Possibilities

### Variable-shape tensors

Arrow defines a
[Variable Shape Tensor](https://arrow.apache.org/docs/format/CanonicalExtensions.html#variable-shape-tensor)
extension type for arrays where each tensor can have a different shape. This would enable workloads
like batched sequences of different lengths.

### Sparse tensors

A sparse tensor type could use `List` or `ListView` as its storage type to efficiently represent
tensors with many zero or absent elements.

### A unified `Tensor` type

This RFC proposes `FixedShapeTensor` as a single, concrete extension type. However, tensors
naturally vary along two axes: shape (fixed vs. variable) and density (dense vs. sparse). Both a
variable-shape tensor (fixed dimensionality, variable shape per element) and a sparse tensor would
need a different storage type, since it needs to efficiently skip over zero or null regions (and for
both, this would likely be `List` or `ListView`).

Each combination would be its own extension type (`FixedShapeTensor`, `VariableShapeTensor`,
`SparseFixedShapeTensor`, etc.), but this proliferates types and fragments any shared tensor logic.
With the matching system on extension types, we could instead define a single unified `Tensor` type
that covers all combinations, dispatching to the appropriate storage type and metadata based on the
specific variant. This would be more complex to implement but would give users a single type to work
with and a single place to define tensor operations.

For now, `FixedShapeTensor` is the only variant we need. The others can be added incrementally
as use cases arise.

### Tensor-specific encodings

Beyond general-purpose compression, encodings tailored to tensor data (e.g., exploiting spatial
locality across dimensions) could improve compression ratios for specific workloads.

### ndindex-style compute expressions

As the extension type expression system matures, we can implement a rich set of tensor indexing and
slicing operations inspired by [ndindex](https://quansight-labs.github.io/ndindex/index.html),
including slice canonicalization, shape inference, and chunk-level re-indexing.
