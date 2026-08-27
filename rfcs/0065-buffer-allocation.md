- Start Date: 2026-08-27
- Authors: @gatesn
- RFC PR: [vortex-data/rfcs#65](https://github.com/vortex-data/rfcs/pull/65)

# Allocator-aware buffers

## Summary

Vortex will allocate host buffers through a caller-provided allocator. This lets an engine track,
pool, and limit buffer memory. `vortex-buffer` will own the allocator API and its memory. It will no
longer use `bytes::BytesMut` as its main storage type.

## Motivation

Vortex kernels allocate buffers with the global Rust allocator. An engine cannot see this memory.
It cannot charge the memory to a query or use its own pool.

Vortex also supports alignment chosen at runtime. `BytesMut` cannot request that alignment. Vortex
must allocate extra bytes and move the start pointer forward. This wastes memory and makes memory
tracking less clear.

The first goal is memory tracking. Allocation stays synchronous. Vortex will not add async memory
back-pressure in this change.

## Design

### Allocator

`vortex-buffer` will use the `allocator_api2::alloc::Allocator` API. Vortex will add the bounds that
it needs for shared execution:

```rust
pub trait BufferAllocator:
    allocator_api2::alloc::Allocator + Debug + Send + Sync + 'static
{
}

#[derive(Clone)]
pub struct BufferAllocatorRef(Arc<dyn BufferAllocator>);
```

The first version should use `allocator-api2` 0.2.21. Vortex already has this version through
`hashbrown`.

Each owned allocation will keep the allocator that made it:

```rust
struct Allocation {
    ptr: NonNull<u8>,
    layout: Layout,
    allocator: BufferAllocatorRef,
}
```

`Allocation::drop` will return the memory to the same allocator. The stored layout will use the
actual size returned by the allocator.

### Buffer storage

Mutable buffers will own one `Allocation`. Immutable buffers will share one backing value:

```rust
enum BufferBacking {
    Owned(Allocation),
    External(Box<dyn BufferOwner>),
}

pub struct BufferMut<T> {
    allocation: Allocation,
    offset: usize,
    len: usize,
    capacity: usize,
    alignment: Alignment,
}

pub struct Buffer<T> {
    ptr: NonNull<T>,
    len: usize,
    alignment: Alignment,
    backing: Arc<BufferBacking>,
}
```

`BufferMut::freeze` will move its allocation into an `Arc`. It will not copy the data. Clones and
slices will clone the `Arc` and change the view pointer. `Buffer::try_into_mut` will use
`Arc::try_unwrap`. It will copy through the same allocator if the buffer is shared.

`BufferOwner` will give Vortex a stable byte slice for its full lifetime. It will support memory
owned by Arrow, a memory map, `bytes::Bytes`, or another library. External memory is immutable.
Making it mutable needs an allocator-backed copy.

The backing type will stay private. We can replace `Arc` with a manual vtable later if benchmarks
show a clear gain.

Mutable `split_off` and `unsplit` do not need shared mutable storage. They may copy. This keeps
`BufferMut` uniquely owned.

### Alignment

Vortex will pass the requested runtime alignment in `Layout`. It will not allocate padding or move
the base pointer.

A buffer will keep two facts:

- The allocation layout records the physical alignment.
- The buffer view records the alignment promised to its caller.

Changing alignment is a metadata change when the current pointer is aligned. Otherwise Vortex will
grow with a new layout or allocate and copy. The allocator API allows the old and new layouts to
have different alignments.

### Public allocation

Allocation will be explicit:

```rust
ctx.allocator().with_capacity::<T>(len);
ctx.allocator().zeroed::<T>(len);
ctx.allocator().copy_from(values);

StaticBufferAllocator::with_capacity::<T>(len);
```

`StaticBufferAllocator` will use the Rust global allocator. It is the escape hatch for tests,
benchmarks, and code without an execution context. A workspace lint can reject it in engine paths.

The no-allocator constructors on `BufferMut` will become private. The same rule will cover buffer
cloning, alignment copies, `FromIterator`, buffer macros, and bit-buffer constructors.

`ExecutionCtx` will store a `BufferAllocatorRef`. It will copy the session allocator when the
context is made.

Allocation will stay infallible at the Vortex API. Allocation failure will call
`handle_alloc_error`, like `Vec`. Fallible constructors may be added later.

## Migration

1. Add the allocator and new buffer storage in `vortex-buffer`.
2. Replace the current `HostAllocator` adapter with `BufferAllocatorRef`.
3. Pass the context allocator through builders and kernels.
4. Reject remaining static allocation in engine code.

Tests will cover alignment changes, growth, clone and slice lifetime, `try_into_mut`, external
owners, failed growth, and accounting after the last view is dropped. Miri will test the unsafe
storage code.

## Compatibility

This does not change the Vortex file format.

It will change some Rust construction APIs. Static allocation remains available through
`StaticBufferAllocator`. `bytes` and Arrow conversions can remain as boundary APIs.

The new storage path may affect buffer size and clone or drop cost. Benchmarks must compare it with
the current implementation before the old path is removed.

## Drawbacks and alternatives

Vortex will own more unsafe memory code. The code must keep pointer, layout, allocator, and view
lifetimes in sync.

Keeping `BytesMut` would need the padding trick because a byte vector asks for alignment one. It
would also hide the allocator behind `BytesMut`.

A manual ownership vtable could support each backing type. It would also require custom reference
counting, release, and unique-owner recovery. The private backing type lets us add this later if an
`Arc` is too costly.

Async allocation and memory back-pressure are out of scope. Device memory is also out of scope.
