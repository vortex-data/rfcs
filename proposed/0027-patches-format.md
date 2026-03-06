## Summary

Make a backwards compatible change to the serialization format for `Patches` used by the FastLanes-derived encodings:

- BitPacked
- ALP
- ALP-RD

enabling fully data-parallel patch application inside of the CUDA bit-unpacking kernels, while not impacting
CPU performance.

This relies on introducing a new encoding to represent exception patching, which would be a forward-compatibility break
as is always the case when adding a new default encoding.

---

## Data Layout

Patches have a new layout, influenced by the [G-ALP paper](https://ir.cwi.nl/pub/35205/35205.pdf) from CWI.

The key insight of the paper is that instead of holding the patches sorted by their global offset, instead

- Group patches into 1024-element chunks
- Further group the patches within each chunk by their "lanes", where the lane is w/e the lane of the underlying operation you're patching over aligns to

For example, let's say that we have an array of 5,000 elements, with 32 lanes.

- We'd have $\left\lceil\frac{5,000}{1024}\right\rceil = 5$ chunks, each chunk has 32 lanes. Each lane can have up to 32 patch values
- Indices and values are aligned. Indices are indices within a chunk, so they can be stored as u16. Values are whatever the underlying values type is.

```text

                 chunk 0      chunk 0      chunk 0     chunk 0       chunk 0     chunk 0
                 lane  0      lane 1       lane  2     lane 3        lane  4     lane  5
             ┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
lane_offsets │     0      │     0      │     2      │     2      │     3      │     5      │  ...
             └─────┬──────┴─────┬──────┴─────┬──────┴──────┬─────┴──────┬─────┴──────┬─────┘
                   │            │            │             │            │            │
                   │            │            │             │            │            │
             ┌─────┴────────────┘            └──────┬──────┘     ┌──────┘            └─────┐
             │                                      │            │                         │
             │                                      │            │                         │
             │                                      │            │                         │
             ▼────────────┬────────────┬────────────▼────────────▼────────────┬────────────▼
   indices   │            │            │            │            │            │            │
             │            │            │            │            │            │            │
             ├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
   values    │            │            │            │            │            │            │
             │            │            │            │            │            │            │
             └────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
```

This layout has a few benefits

- For GPU operations, each warp handles a single chunk, and each thread handles a single lane. Through the `lane_offsets`, each thread of execution can have quick random access to an iterator of values
- Patches can be trivially sliced to a specific chunk range simply by slicing into the `lane_offsets`
- Bulk operations can be executed efficiently per-chunk by loading all patches for a chunk and applying them in a loop, as before
- Point lookups are still efficient. Convert the target index into the chunk/lane, then do a linear scan for the index. There will be at most `1024 / N_LANES` patches, which in our current implementation is 64. A linear search with loop unrolling should be able to execute this extremely fast on hardware with SIMD registers.

---

## Array Structure

```rust
/// An array that partially "patches" another array with new values.
///
/// Patched arrays implement the set of nodes that do this instead here...I think?
#[derive(Debug, Clone)]
pub struct PatchedArray {
    /// The inner array that is being patched. This is the zeroth child.
    pub(super) inner: ArrayRef,

    /// Number of 1024-element chunks. Pre-computed for convenience.
    pub(super) n_chunks: usize,

    /// Number of lanes the patch indices and values have been split into. Each of the `n_chunks`
    /// of 1024 values is split into `n_lanes` lanes horizontally, each lane having 1024 / n_lanes
    /// values that might be patched.
    pub(super) n_lanes: usize,

    /// Offset into the first chunk
    pub(super) offset: usize,
    /// Total length.
    pub(super) len: usize,

    /// lane offsets. The PType of these MUST be u32
    pub(super) lane_offsets: BufferHandle,
    /// indices within a 1024-element chunk. The PType of these MUST be u16
    pub(super) indices: BufferHandle,
    /// patch values corresponding to the indices. The ptype is specified by `values_ptype`.
    pub(super) values: BufferHandle,
    /// PType of the scalars in `values`. Can be any native type.
    pub(super) values_ptype: PType,

    pub(super) stats_set: ArrayStats,
}
```

The PatchedArray holds buffer handles for the `lane_offsets` which provides chunk/lane-level random indexing
into the patch `indices` and `values`, so these values can live equivalently in device or host memory.

The only operation performed at planning time is slicing, which means that all of its reduce rules would run
without issue in CUDA or on CPU.

---

# Operations

## Slicing

We look at the slice indices, align them to chunk boundaries, then slice both the child and the patches to chunk boundaries, and preserve the offset + len to apply the final intra-chunk slice at execution time.

## Filter / Take Execution

Filter / Take operations can arbitrarily break and reconstruct new chunks, so they cannot be done metadata-only and thus must be a Kernel rather than a Reduce rule.

In practice, we perform the operation by

- Executing the filter on the child, then executing it
- Intersecting the filter with our patches, ideally in a chunk-at-a-time way so we can write a vectorized version.
- Applying the filtered patches over the executed child

## ScalarFns

We do not reduce any ScalarFns through the operation, instead they only run at execution time.

This matches the current behavior of BitPackedArrays.

---

## Compatibility

BitPackedArray and ALPArray both hold a `Patches` internally, which we'd like to replace by wrapping them in a `PatchedArray`.

To do this without breaking backward compatibility, we modify the `VTable::build` function to return `ArrayRef`. This makes it easy to do encoding migrations on read in the future. The alternative is adding a new BitPackedArray and ALPArray that gets migrated to on write.

This requires executing the Patches at read time. From scanning a handful of our tables, this is unlikely to cause any issues as patches are generally not compressed. We only apply constant compression for patch values, and I would expect that to be rare in practice.

## Drawbacks

This will be a forward-compatibility break. Old clients will not be able to read files written with the new encoding.
However, the potential break surface is huge given how ubiquitous bitpacked arrays and patches are in our encoding trees.
This will cause friction as users of Vortex who have separate writer/reader pipelines will need to upgrade their Vortex
clients across both in lockstep.

> Does this add complexity that could be avoided?

IMO this centralizes some complexity that previously was shared across multiple encodings.

## Alternatives

> Transpose the patches within GPU execution

This was found to be not very performant. The time spent D2H copy, transpose patches, H2D copy far exceeded the cost of executing the bitpacking kernel, which puts a serious
limit on our GPU scan performance. Combined with how ubiquitous `BitPackedArray`s with patches are in our encoding trees, would be a permanent bottleneck on throughput.

> What is the cost of **not** doing this?

Our GPU scan performance would be permanently limited by patching overhead, which in TPC-H lineitem scans was shown to be the biggest bottleneck after string decoding.

> Is there a simpler approach that gets us most of the way there?

I don't think so

## Prior Art

The original FastLanes GPU paper did not attempt to implement data-parallel patching within the FastLanes unpacking
kernels.

The G-ALP paper was published later on, and implemented patching for ALP values _after_ unpacking.

We use a data layout that closely matches the one described in _G-ALP_ and apply it to bit-unpacking as well.

## Unresolved Questions

- What parts of the design need to be resolved during the RFC process?
- What is explicitly out of scope for this RFC?
- Are there open questions that can be deferred to implementation?

## Future Possibilities

What natural extensions or follow-on work does this enable? This is a good place to note related ideas that are out of scope for this RFC but worth capturing.
