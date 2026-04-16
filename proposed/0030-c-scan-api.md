- Start Date: 2026-03-13
- Authors: @myrrc

# High level C Scan API
## Summary

Provide a layered C Scan API for non-Rust clients[^1], each layer exposing more
capabilities to the host:

- High level layer is intended for sync multi-threaded or multi-process runtimes
  like DuckDB, ClickHouse, or Postgres. It provides a `DataSource ->
  Partition -> Array` chain with thread safety. It manages the event loop inside
  Vortex but allows offloading IO and memory allocations to the host. This layer
  roughly matches scan API
  [exposed](https://github.com/vortex-data/vortex/tree/develop/vortex-scan)
  in Rust.
- Middle level layer is intended for async runtimes like Velox which want to run
  their own event loop. In addition to offloaded IO, it doesn't do any things
  eagerly[^2] but returns control on IO or CPU work for host to schedule.
  This layer exposes a coroutine-like state machine.
- Low level layer is intended for clients which need greater control over
  scan scheduling. This layer roughly matches a
  [LayoutReader](https://github.com/vortex-data/vortex/tree/develop/vortex-layout/src/reader.rs)
  in Rust.

| Level | Allocations | IO | Event loop | Scan planning |
| ----- | ----------- | -- | ---------- | ------------- |
| High  | Host/Vortex | Host/Vortex | Vortex | Vortex |
| Middle | Host | Host | Host | Vortex |
| Low | Host | Host | Host | Host |

## Motivation

Vortex is a Rust project, and thus integration with non-Rust clients requires a
decent amount of scaffolding[^3] which lowers the probability of such
integration. As reading files is most important feature for a file format,
exposing it to C-compatible users lowers integration costs.

Additional motivation is our DuckDB integration which requires C++-to-Rust
and Rust-to-C++ bridging. Replacing it with direct calls to C API would make
the project much smaller.

[Current](https://github.com/vortex-data/vortex/tree/develop/vortex-ffi/src/file.rs)
C Scan API in the form of `vx_file_scan` is incomplete for the following
reasons:

- Thread unsafety: host needs to serialize calls to `vx_array_iterator`.
- Lack of support for file globs.
- Lack of support for exporting to `ArrowSchema` and `ArrowArrayStream`.
- Lack of introspection over produced `vx_array` objects.

> Vortex also has a C++ API in vortex-cxx which wraps Rust code. Its future is
> outside of scope of this proposal but it's expected to wrap C scan API,
> removing dependency on cxx.rs for vortex-cxx.

## Motivation for layered API

Velox uses a non-coroutine but async runtime based on Folly executors. CoroBase
uses a coroutine-based runtime. DuckDB and ClickHouse runtimes are sync based on
thread pools. Postgres runtime is sync based on processes. Some of engines may
use OS-specific IO like `io_uring`.

Scan API usage may be grouped into two distinct cases:

- sync, single-thread/multi-thread/multi-process runtime, and
- async, multi-thread/coroutine runtime.

Exposing just the high level would not work well with async hosts as there's
no way to suspend to host on IO operations. Exposing only the middle layer
requires sync hosts to manage their own event loop which is tedious.

Another benefit of splitting API into layers is that it can be discussed and
implemented separately. As this is a big change, this PR will from now on focus
just on the high level scan API. Other levels may be implemented later on
demand.

## Overview

```
┌──────────┐
│DataSource│
└─┬────────┘
  ▼ produces a
┌────┐
│Scan│
└┬┬┬─┘
 │││ worker threads pull a
 ▼▼▼
┌───────────┐
│ Partition │ (thread safe)
└┬──────────┘
 │ which produces an
 └─►Array (thread unsafe)
```

## Event loop

Vortex manages the event loop by using a CurrentThreadRuntime runtime handle
which in turn has a `smol::Executor<'static>` without a background thread pool.
This means the event loop will run on host's thread pool scheduler (in other
terms, use the calling thread) implicitly. In case of multiple threads, the
underlying executor is shared between them, and drives from from all of these
threads. The executor is initialized once on loading FFI library.

## DataSource

A DataSource is a reference to multiple possibly remote files. When created, it
opens first file to determine the schema from DType, all other operations are
deferred till a scan is requested and executed. You can request multiple file
scans from a DataSource.

```c
// bindgen
typedef struct vx_data_source vx_data_source;
typedef struct vx_file_handle vx_file_handle;

typedef void* vx_fs_userdata;
typedef void (*vx_list_callback)(vx_fs_userdata userdata, const char* name,
    int is_dir);
typedef void (*vx_glob_callback)(vx_fs_userdata userdata, const char* file);

typedef struct vx_data_source_options {
    const char* files;

    // (1) Filesystem customization
    vx_fs_userdata fs_userdata;
    void (*fs_free_userdata)(vx_fs_userdata fs_userdata);

    // Called after glob expansion, single-file mode
    vx_file_handle (*fs_open)(vx_fs_userdata userdata, const char* path,
        vx_error** err);
    vx_file_handle (*fs_create)(vx_fs_userdata userdata, const char* path,
        vx_error** err);

    // non-recursive, callback is invoked with each path
    void (*fs_list)(vx_fs_userdata userdata, const char* path,
        vx_list_callback cb, vx_error** err);

    void (*fs_close)(vx_file_handle handle);
    uint64_t (*fs_size)(vx_file_handle handle, vx_error** err);

    // positional read, doesn't change file open cursor
    void (*fs_read)(vx_file_handle handle, uint64_t offset, uint64_t len,
        uint8_t *buffer, vx_error** err);

    // not needed for scanning but makes FS API complete
    void (*fs_write)(vx_file_handle handle, uint64_t offset, uint64_t len,
        uint8_t *buffer, vx_error** err);

    void (*fs_sync)(vx_file_handle handle, vx_error** err);

    // (2) Globbing customization
    void (*glob)(const char* glob, vx_glob_callback cb, vx_error** err);
} vx_data_source_options;

// Addition to existing DType API, returns owned ArrowSchema which needs to
// be freed by the caller using release callback. If err is populated, out
// is not set.
void vx_dtype_to_arrow_schema(const vx_dtype* dtype, ArrowSchema* out,
    vx_error** err);

/// Create a new owned datasource which must be freed by the caller.
const vx_data_source *
vx_data_source_new(const vx_session *session,
    const vx_data_source_options *opts, vx_error** err);

// datasource is Arc'd inside so a clone creates another reference rather
// than cloning the state fully.
const vx_data_source *vx_data_source_clone(const vx_data_source *ptr);

// vx_dtype's lifetime is bound to datasource's lifetime, caller doesn't need
// to free it
const vx_dtype *vx_data_source_dtype(const vx_data_source *ds);

typedef enum {
    VX_CARD_UNKNOWN = 0,
    VX_CARD_ESTIMATE = 1,
    VX_CARD_MAXIMUM = 2,
} vx_cardinality;
typedef struct {
    vx_cardinality cardinality;
    uint64_t rows;
} vx_data_source_row_count;

void vx_data_source_get_row_count(const vx_data_source *ds,
    vx_data_source_row_count *rc);
void vx_data_source_free(const vx_data_source *ds);
```

> const vx_data_source* may be misleading as it's actually owned, but that's
> limitation of arc_dyn_wrapper! as for now. This would likely be changed soon

1. Open local or remote file. Allow using Vortex's filesystem or host's
   filesystem e.g. DuckDB fs. Host filesystem customization
   point has the benefit of not duplicating credentials e.g. S3 access key
   between host and Vortex. Local implementation may be more performant.
   User must either implement all `fs_*` callback and set `fs_userdata`, or set
   all callbacks in (1) and `fs_userdata` to NULL.

2. Open single file or multiple files. Hosts may have their own glob expansion
   [^4] which does HTTP IO.

When both customization points are implemented, Vortex offloads all IO
to a query engine.

> Memory allocation customization is out of scope of this proposal, but it's
> possible for Vortex to expose bringing allocator from outside for buffers.

To allow stateful filesystem operations, a DataSource allows providing a custom
filesystem userdata with associated deleter. It may be set per DataSource.
Filesystem customizations do syncrohonous IO without batching. Clients who want
control over IO should use the middle layer or lower.
As filesystem callbacks are synchronous, middle layer would use different option
structs, and there would be no way to re-use DataSource or other structures
between layers.

## Scan

A Scan is a one-time traversal of files in a DataSource. A Scan can't be
restarted once requested. Hosts are encouraged to utilize multiple threads for
scanning. Scan thread-safely produces Partitions[^5] which are chunks of work
for a thread[^6].

Host may instruct a Scan about its parallelism degree by passing `max_threads`.
Vortex may return less partitions than there are `max_threads` but won't return
more.

There will be no option to cancel a scan as this isn't possible on Rust side
either and this is a low priority task.

Scan options:

```c
// roaring bitmap include and exclude won't be supported as for now because of
// exposure complexity
typedef enum {
    VX_S_INCLUDE_ALL = 0,
    VX_S_INCLUDE_RANGE = 1,
    VX_S_EXCLUDE_RANGE = 2,
} vx_scan_selection_include;

// If selection is VX_S_INCLUDE_ALL, idx is not read. idx is copied
// by host on scan invocation and can be freed after first partition is
// requested
typedef struct {
    uint64_t *idx;
    size_t idx_len;
    vx_scan_selection_include include;
} vx_scan_selection;

typedef struct vx_scan_options {
    const vx_expression *projection; // NULL means "return all columns"
    const vx_expression *filter; // NULL means "no filter"
    // special case: row_range_begin == 0 && row_range_end == 0 means "all rows"
    uint64_t row_range_begin; // Inclusive
    uint64_t row_range_end; // Exclusive
    vx_scan_selection selection;
    uint64_t limit; // 0 means no limit
    uint64_t max_threads; // 0 means no limit.
    int ordered; // 0 means unordered scan
} vx_scan_options;
```

Scan interface:

```c
// bindgen
typedef struct vx_scan vx_scan;
// A Partition is a chunk of work for a thread.
typedef struct vx_partition vx_partition;

typedef enum {
    VX_ESTIMATE_UNKNOWN = 0,
    VX_ESTIMATE_EXACT = 1,
    VX_ESTIMATE_INEXACT = 2,
} vx_estimate_boundary;

// If type is VX_ESTIMATE_UNKNOWN, estimate is not populated.
typedef struct {
    uint64_t estimate;
    vx_estimate_boundary type;
} vx_estimate;

// Users are encouraged to create worker threads depending on
// estimate->estimate to distribute work.
// options and estimate fields may be NUL.
// Calling vx_data_source_scan doesn't do IO unless vx_scan_next is called.
//
// As an implementation detail, a scan has a reference to a data source thus
// a scan can outlive a data source.
vx_scan *
vx_data_source_scan(const vx_data_source *ds, const vx_scan_options *options,
    vx_estimate* estimate, vx_error **err);

// Return scan progress from 0.0 to 1.0.
// Reports estimates of read vs. total rows. Thread-safe.
// May be inaccurate, main use case is progress bars in clients.
double vx_scan_progress(const vx_scan *scan);
```

## Partition

A Partition allows a worker thread to produce Arrays thread-unsafely.
Partitions also allow exporting the data to ArrowArrayStream for hosts
that don't want to introspect Arrays. Hosts should call `vx_scan_next`
and `vx_partition_next` till they return NULL which signals there's no more
data.

```c
// Get next owned partition out of a scan request.
// Caller must free this partition using vx_partition_free.
// This method is thread-safe.
// You can call vx_scan_free or vx_data_source_free while you are holding
// partitions.
// Hosts are encouraged to create a worker thread per partition.
// Returns NULL and sets *err to NULL on exhaustion.
// Returns NULL and sets *err on error.
vx_partition *vx_scan_next(vx_scan *scan, vx_error **err);

// Request an array stream in Arrow format from a partition, consuming it
// fully. Calling vx_partition_next on partition after calling this function
// is undefined behaviour.
// User does not need to call vx_partition_free in partition after calling
// this function.
// Not thread-safe, should be called once.
// stream is owned and must be freed using the release callback.
// Does not modify stream and sets err on error.
void vx_partition_scan_arrow(const vx_partition *partition,
    ArrowArrayStream *stream, vx_error** err);

// Thread-unsafe. Get an owned vx_array of an iterator.
// Calling from multiple threads is undefined behaviour.
// Array must be freed by caller.
// You can call vx_scan_free, vx_data_source_free, or vx_partition_free 
// while you are holding arrays from it.
// Returns NULL and sets *err to NULL on exhaustion.
// Returns NULL and sets *err on error.
const vx_array *vx_partition_next(vx_partition *partition, vx_error **err);

// Return a (possibly exact) estimate of rows in a partition.
void vx_partition_row_count(const vx_partition *partition, vx_estimate *count,
    vx_error **err);
```

## Array introspection

The main question is how to transform outputs of iteration, `vx_array`, into
something query engines can operate with. You need to execute the array
iteratively till you recognize data and start exporting it. Thus API provides a
way to scan partitions directly into ArrowArrayStream which should be good
enough for most hosts. However, some hosts may want to work with vx_array
directly as it provides types like ConstantArray which Arrow doesn't have, and
thus convertion requires CPU-intensive work.

## Compatibility

No impact. Existing C Scan API will exist for some time but will be removed
eventually.

## Prior Art

- Dividing scan requests into Partitions for threads is taken from DataFusion's
  [partitioned streams](https://docs.rs/datafusion/latest/datafusion/physical_plan/enum.Partitioning.html).
- Making Partitions produce Arrays without synchronization mimics
  [Arrow Stream](https://arrow.apache.org/docs/format/CStreamInterface.html)
  interface.

## Unresolved Questions

- Should high level API support reading a file from a buffer and not from a
  path? Is there a use case for that?
- Should high level API have a way to use host's persistent caching options?
  In-memory caching may be implemented using host allocator only but for
  persistent caching we need additional filesystem customizations i.e. cache
  location path.
- Should high level API expose batch reads from Partition? There are plans to
  deprecate Partitions on Rust side.
- What introspection should high level API have for hosts which aren't satisfied
  with `Vortex -> ArrowArrayStream` conversion? Should there be iterative
  execution API?
- How should API expose definitions of `ArrowSchema` and `ArrowArrayStream`?
  Rust's implementation exposes `FFI_` structs and you need to typedef
  them manually. Should API bundle `nanoarrow` or let hosts handle typedefs
  themselves since the writer is ABI-compatible? Should we gate it behind a
  macro?

  ```c
  typedef FFI_ArrowSchema ArrowSchema;
  #include "vortex.h"
  ```

  Current implementation copy-pastes ArrowSchema and ArrowArrayStream and gates
  them behind a macro.

## High level API integration example: DuckDB

```
Before:

DuckDB                            Vortex

C++                               C++             Rust
duckdb -> TableFunction vtable -> ffi wrapping -> vtable implementation

After:

C++                               C++             C
duckdb -> TableFunction vtable -> ffi_wrapping -> vx_scan()*

* - vx_array -> DataChunk reuses existing Rust code
```

https://github.com/vortex-data/vortex/tree/myrrc/scan-api-duckdb has a PoC (not
production-ready) implementation of using C Scan API for Duckdb scan
integration. Duckdb has a sync multi-threaded runtime, and table function is
called from multiple threads simultaneously. Users can save a per-thread state.

The integration splits into following parts:
- DType -> LogicalType integration, done except for Temporal extension support.
- Table function binding (creating a DataSource), done.
- Global state initialization (creating a Scan), done except filter pushdown.
- Local state initialization (export batch id), done.
- Utility functions like cardinality estimates, done.
- vx_array -> DataChunk export, delegated to existing Rust code.

On filter pushdown: projection pushdown requires exposing only `select()`
expression. On the other hand, filter pushdown requires `TableFilter -> Vortex
Expression` conversion which is significant porting so left out.

On DataChunk export: it requires exposing features like array optimization,
validity masks, and other features, so left out.

Table function uses Vortex _partition_ concepts as a work splitting term only,
i.e. one worker thread operating on one or multiple partitions. Each thread
pulls out partitions from `vx_scan_next` (thus it's thread-safe) and then
works on its own partition without synchronization.

## Footnotes

[^1]: Clients of this API would mostly be query engines like Velox or
    ClickHouse, but may as well be our own integrations like vortex-duckdb.
[^2]: Like opening the first file in a glob expression to determine schema.
[^3]: Rust ABI is not stable if you don't use `repr(C)`, so clients can't use
    cbindgen directly. C++ clients can use cxx.rs but this still requires
    writing manual bindings and runtime bridging.
[^4]: DuckDB MultiFileReader and MultiFileList.
[^5]: The name may be misleading as it doesn't correspond to Rust side's
    Partitions. Alternative considered was vx_split but it conflicts with
    register_splits in LayoutReader.
[^6]: DuckDB integration currently hides Partitions and Arrays behind a single
    [thread-safe iterator](https://github.com/vortex-data/vortex/blob/e8cd130c8ccac45082a0b458b1f843c4313555bf/vortex-duckdb/src/datasource.rs#L151)
    which implies unnecessary intra-thread synchronization on pulling data. On
    the other hand, the producer, an async crossbeam queue, allows smoothing out
    uneven workloads from the Vortex side, and if that's removed, Vortex's
    partition scheduling must be precise.
