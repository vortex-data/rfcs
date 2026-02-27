# RFC 0001 - Updated Patches Format

## Goals

1. Create a new `PatchesArray` encoding, which encapsulates the current patch indices/values children currently inlined into `BitPackedArray` and `ALPArray`.
1. New patches encoding will use G-ALP style exception values by default (i.e., values are sorted per-vector, per-lane rather than global offset)
1. All new arrays with patches will write the new encoding. The old encoding can still be ready with fallback

## Background

Both the FastLanes and ALP paper document special handling for
exception values, values that need to be applied back into the array
after the decoding step. In Vortex, we call these _Patches_.

Let's take bit-packing for example. A BitPackedArray with exceptional
values will look like this:

```
     ╔ ═ ═ ═ ═ ═ ═ ═ ═ ═ ═
                          ║
     ║   BitPackedArray
                          ║
     ╚ ═ ═ ═ ═ ═ ═ ═ ╤ ═ ═
      │              │
      │              │
      │              │
      │              │
      │              │
      │              │
      │              │  patch
      │              │  indices    ╔ ═ ═ ═ ═ ═
┌─────▼─────┐        ├─────────────▶ ArrayRef ║
│░░░░░░░░░░░│        │             ╚ ═ ═ ═ ═ ═
│░░Buffer░░░│        │
│░░░░░░░░░░░│        │  patch
└───────────┘        │  values     ╔ ═ ═ ═ ═ ═
   encoded           └─────────────▶ ArrayRef ║
                                   ╚ ═ ═ ═ ═ ═
```

`BitPackedArray` stores the patch indices and patch values as child
arrays inline with itself. NOTE: both ALP and ALP-RD encodings have similar structure.

Patches must support 2 operations: random access and bulk decoding.

Patches are sorted by their indices for fast random access via binary search.

Bulk decoding is handled by executing both the `patch_indices` and `patch_values`, and then applying
them in a straightline loop. There are no loop conditions and several indices/values
can be loaded into vector registers at once, so this is extremely performant on the CPU.

## GPU Considerations

As part of our goal to bring CUDA support to all Vortex encodings, we need to be able to
apply patch values to the output node directly inline.

Let's take `BitPackedArray` as a specific example. The bit unpacking kernels on the GPU map each thread to
a single vector lane.

## Back Compat Plan
