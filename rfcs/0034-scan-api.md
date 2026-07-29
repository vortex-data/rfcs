- Start Date: 2026-04-07
- Authors: @gatesn
- RFC PR: [vortex-data/rfcs#0034](https://github.com/vortex-data/rfcs/pull/0034)

# Scan API

## Summary

This RFC describes the desired end state for the Vortex Scan API. It replaces hidden implicit async
execution with an explicit scheduler model around poll-based partition readers: the host engine
schedules partitions, the source schedules splits within each partition, and the host provides
budgets and services for memory, I/O, and CPU work. The goal is to make the Scan API powerful enough
to embed in different query engines (DataFusion, DuckDB, Velox, Polars) while preserving a clean
separation between physical storage access, scan planning, host runtime control, and operator
execution.

## Motivation

The design builds on concepts already present in the repo today (`DataSource`, `DataSourceScan`,
`Partition`, `SplitBy`, `ScanBuilder`, `LayoutReader`), but is not constrained to preserve the exact
current architecture. In particular, it replaces hidden implicit async execution with an explicit
scheduler model around poll-based partition readers:

- the host engine schedules partitions
- the source schedules splits within each partition
- the host provides budgets and services for memory, I/O, and CPU work

### Goals

- Keep `LayoutReader` focused on bounded, immutable, physically prunable data.
- Keep the Scan API host-engine-neutral.
- Replace hidden runtime behavior with explicit scheduling and budgeting.
- Allow simple embeddings to use a default Vortex runtime.
- Allow advanced embeddings to hand over control of:
  - memory / buffer pools
  - I/O scheduling
  - backpressure
  - task scheduling
  - NUMA placement
  - cancellation
  - metrics
- Support engines with different scheduling models:
  - DataFusion
  - DuckDB
  - Velox
  - Polars
  - Spiral
- Support both `Send` and thread-affine `!Send` execution embeddings.
- Support exact pushdown where semantics match, and safe residual evaluation where they do not.

### Non-goals

- The Scan API is not itself a full relational query engine.
- `LayoutReader` should not grow unknown-cardinality operator semantics.
- Vortex should not require a specific Rust async runtime such as Tokio.
- The source should not hide material scheduling decisions behind an opaque async stream.

## Layering

The end state has three layers.

### 1. Layout Layer

The layout layer is the existing `LayoutReader` world.

Responsibilities:

- row-count-known, immutable, snapshot-based reads
- physical pruning
- split discovery via `SplitBy`
- projection and filter evaluation against layouts
- segment-level I/O registration and prefetch

Non-responsibilities:

- query scheduling policy
- worker placement
- host memory governance
- join / aggregate / explode / regroup operators

This is the layer implemented today in `vortex-layout`, especially in:

- [`vortex-layout/src/scan/layout.rs`](/Users/ngates/git/vortex2/vortex-layout/src/scan/layout.rs)
- [`vortex-layout/src/scan/scan_builder.rs`](/Users/ngates/git/vortex2/vortex-layout/src/scan/scan_builder.rs)
- [`vortex-layout/src/scan/split_by.rs`](/Users/ngates/git/vortex2/vortex-layout/src/scan/split_by.rs)
- [`vortex-layout/src/scan/splits.rs`](/Users/ngates/git/vortex2/vortex-layout/src/scan/splits.rs)
- [`vortex-layout/src/scan/tasks.rs`](/Users/ngates/git/vortex2/vortex-layout/src/scan/tasks.rs)

### 2. Scan Layer

The scan layer is the host-neutral contract between storage and query engines.

Responsibilities:

- take a `ScanRequest`
- produce a `ScanPlan`
- expose `Partition`s to the host engine
- let the host open and poll a `PartitionReader`
- mediate access to `VortexSession` services and per-scan execution budgets

This layer belongs in `vortex-scan`.

### 3. Operator Layer

The operator layer is the host engine proper: DataFusion, DuckDB, Velox, Polars, etc.

Responsibilities:

- global scheduling
- operator graph execution
- join / aggregation / exchange
- backpressure
- runtime filter production
- task placement
- memory policy

The Scan API should be a powerful leaf/source abstraction for this layer, not a replacement for it.

## Core Types

The current names are mostly right. The main adjustment is to make the difference between
engine-visible `Partition`s and Vortex-internal `Split`s explicit.

### `DataSource`

A `DataSource` is any scanable dataset.

Examples:

- a Vortex file
- a Vortex layout tree
- a Vortex table
- a Parquet file
- an Iceberg table
- a remote storage service exposing scan endpoints

It is a reusable object. A single `DataSource` can be scanned many times.

### `ScanPlan`

A `ScanPlan` is a planned scan over a `DataSource`.

It captures:

- projection
- pushed-down predicates
- row selection
- row range
- limit
- ordering requirements
- partitioning preferences

It is not itself execution. It is the host-visible description of how to obtain work.

This corresponds roughly to today's `DataSourceScan`, but the name `ScanPlan` makes the intended
role clearer.

### `Partition`

A `Partition` is the unit of work exposed by the Scan API to a host engine.

Important:

- A `Partition` is not the same thing as a layout chunk.
- A `Partition` is not the same thing as a DuckDB morsel.
- A `Partition` is not the same thing as a Velox split.

It is simply the Scan API's public unit of parallel work.

Each embedding maps host concepts onto `Partition`s differently.

### `SplitBy`

`SplitBy` remains the Vortex internal policy for subdividing a `Partition` into finer work.

Today this is already present:

- `SplitBy::Layout`
- `SplitBy::RowCount`

This should remain an internal physical policy, not an engine contract.

### `SplitPlan`

The current `Splits` type should be thought of as a `SplitPlan`.

It describes how a `Partition` will be chopped into concrete `Split`s for execution:

- natural layout-aligned splits
- exact row ranges for sparse selection

### `Split`

A `Split` is one concrete internal scan unit inside a `Partition`.

Examples:

- a row range
- a layout-aligned chunk
- a sparse exact range

`Split` is where Vortex does prefetch registration, split-local pruning, and split-local projection.

### `Task`

A `Task` is runtime machinery: one in-flight execution of a `Split`.

This is what `vortex-layout/src/scan/tasks.rs` is modeling today.

### `PartitionReader`

This is the main missing execution abstraction in the current Scan API.

A host should open a `Partition` and obtain a reader that it can poll directly.

Unlike an opaque self-scheduling stream implementation, a reader:

- receives host budgets explicitly
- can expose progress and internal state
- can separate prefetch from production
- can be polled by sync or async hosts through adapters
- can be opened either as `Send` or as thread-affine local state, depending on the host contract

## End-state Trait Sketch

This is a sketch, not exact Rust syntax.

```rust
#[async_trait]
pub trait DataSource: Send + Sync + 'static {
    fn dtype(&self) -> &DType;
    fn row_count(&self) -> Option<Precision<u64>> { None }
    fn byte_size(&self) -> Option<Precision<u64>> { None }

    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> { Ok(None) }

    fn deserialize_partition(
        &self,
        data: &[u8],
        session: &VortexSession,
    ) -> VortexResult<PartitionRef>;

    async fn plan(
        &self,
        session: &VortexSession,
        req: ScanRequest,
    ) -> VortexResult<ScanPlanRef>;

    async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<StatsSet>;
}

pub struct ScanRequest {
    pub projection: Expression,
    pub filter: Option<Predicate>,
    pub row_range: Option<Range<u64>>,
    pub selection: Selection,
    pub ordered: bool,
    pub limit: Option<u64>,
    pub options: ScanOptions,
}

pub struct ScanOptions {
    pub partitioning: Partitioning,
    pub split_by: SplitBy,
    pub target_batch_rows: Option<usize>,
    pub target_batch_bytes: Option<u64>,
}

pub enum Partitioning {
    Natural,
    FixedCount(usize),
    SharedWork,
    HostManaged,
}

pub trait ScanPlan: Send + 'static {
    fn dtype(&self) -> &DType;
    fn partition_count(&self) -> Option<Precision<usize>>;
    fn partitions(self: Box<Self>) -> PartitionStream;
}

pub trait Partition: Send + 'static {
    fn as_any(&self) -> &dyn Any;
    fn row_count(&self) -> Option<Precision<u64>>;
    fn byte_size(&self) -> Option<Precision<u64>>;
    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> { Ok(None) }

    fn open_send(
        self: Box<Self>,
        session: &VortexSession,
        budget: &ScanBudget,
    ) -> VortexResult<Box<dyn SendPartitionReader>>;

    fn open_local(
        self: Box<Self>,
        session: &VortexSession,
        budget: &ScanBudget,
    ) -> VortexResult<Box<dyn LocalPartitionReader>>;
}

pub trait PartitionReader {
    fn dtype(&self) -> &DType;

    fn poll_next(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<VortexResult<Option<ArrayRef>>>;

    fn stats(&self) -> PartitionStats;

    fn blocked_on(&self) -> Option<BlockedOn> {
        None
    }

    fn close(&mut self) -> VortexResult<()> {
        Ok(())
    }
}

pub enum BlockedOn {
    Io,
    Cpu,
    Memory,
    Budget,
}

pub trait SendPartitionReader: PartitionReader + Send {}
pub trait LocalPartitionReader: PartitionReader {}
```

This keeps async in the control plane and makes the data plane explicit, poll-based, and
runtime-neutral.

`open_send` and `open_local` are mobility contracts, not two different scan semantics.

- `open_send` is for engines such as DataFusion that require `Send` readers.
- `open_local` is for thread-affine hosts such as DuckDB-style local execution.

A source may implement both natively, implement one in terms of the other, or reject one mode as
unsupported.

## Why `PartitionReader` Instead of `execute() -> SendableArrayStream`

`execute() -> SendableArrayStream` is convenient, but it hides too much:

- who owns scheduling
- who owns buffering
- who is responsible for cancellation
- which runtime is used

The end state should still allow a convenience adapter that turns a `PartitionReader` into a
Rust stream, but the fundamental execution contract should be a host-driven reader.

That works better for:

- async engines
- sync engines
- host-managed runtimes
- FFI embeddings

The important distinction is not that the reader stops looking stream-like. It is that the source
does not get to hide prefetch policy, split lookahead, or buffering behind an opaque
self-scheduling stream implementation. The host still owns execution context and may inspect what
the reader is blocked on.

## `VortexSession` and `ScanBudget`

The Scan API should not be parameterized by an `Engine` trait.

Instead, the engine should configure a `VortexSession` once with the services it wants Vortex to
use, and then pass a separate `ScanBudget` into scan execution.

This keeps the ownership clean:

- `VortexSession` holds capabilities, registries, and services
- `ScanBudget` holds per-scan or per-partition execution ceilings

This avoids introducing a second capability container that overlaps with `VortexSession`.

```rust
pub struct ScanBudget {
    pub max_active_splits: usize,
    pub max_prefetch_splits: usize,
    pub max_prefetch_bytes: u64,
    pub max_inflight_reads: usize,
    pub max_buffered_batches: usize,
}
```

The expectation is that `VortexSession` exposes accessors for services such as:

- memory / buffer manager
- I/O scheduler
- CPU scheduler
- metrics sink
- cancellation defaults
- function registries
- source openers

Simple embeddings can use a default session. Advanced hosts configure the session once globally.

### Execution Backend Coherency

The services configured on `VortexSession` must form one coherent execution backend.

In particular, the following should not be treated as independently swappable in the default path:

- the mechanism used to drive async progress, such as `block_on` or equivalent polling glue
- the default I/O scheduler
- the default CPU scheduler

If a host replaces only one of these, the result may be a configuration where I/O is never driven
or background work is scheduled onto the wrong runtime.

The intended model is:

- a host configures one coherent backend on the session
- that backend may expose both `Send` and local/thread-affine entry points
- `ScanBudget` then constrains how much of that backend a given scan is allowed to consume

## `PartitionReader` Execution Model

`PartitionReader` is intentionally single-owner.

That means:

- one host task, thread, or worker opens one partition reader
- the host drives it by polling
- the host is responsible for parallelism by opening multiple partitions

This lines up with:

- DataFusion task-per-partition execution
- DuckDB local thread state
- Velox driver-local source operators
- Polars source readers

Internally, a `PartitionReader` may still execute multiple split-local tasks, but that concurrency
must remain bounded by the host budget and scan options. A reader must not grow unbounded queues
behind the host's back.

The distinction between `open_send` and `open_local` does not change this rule. It only changes
whether the resulting reader may move across threads.

### Reader Windows

Each reader should manage a few explicit windows:

- `planned_splits`
- `prefetch_window`
- `active_window`
- `ready_batches`

The budget controls the maxima. The source decides how to fill them.

## Macro Scheduling vs Micro Scheduling

This is the core ownership rule.

- The host engine schedules partitions.
- The source schedules splits within each partition.
- The host supplies ceilings through `ScanBudget` and services through `VortexSession`.

Examples:

- the host decides whether there are 8 or 128 partitions
- the source decides which 4 splits inside a partition should be prefetched next
- the host decides the maximum prefetch bytes
- the source decides how to spend that budget probabilistically

This is the right place for sophisticated scan behavior. A query engine typically cannot score
layout-local prefetch candidates as well as the source can.

## Scheduling Hooks

The Scan API should separate I/O services from CPU scheduling services.

### CPU Scheduler Contract

```rust
pub trait CpuScheduler: Send + Sync {
    fn spawn_cpu(
        &self,
        task: BoxFuture<'static, VortexResult<()>>,
    ) -> VortexResult<Box<dyn ScanJoinHandle>>;
}

pub trait ScanJoinHandle: Send {
    fn poll_join(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<VortexResult<()>>;
}
```

The exact shape may vary, but the intended ownership does not:

- the host may provide a CPU scheduler
- Vortex may use it for bounded split-local CPU work
- Vortex must not assume ownership of the whole query runtime

### What the Scheduler Is For

- split-local filter evaluation
- split-local decode and projection work
- bounded lookahead inside a partition
- optional concurrency for expensive compute kernels

### What the Scheduler Is Not For

- global partition placement
- whole-query work stealing
- exchange or join scheduling
- engine-wide backpressure policy

Those remain in the host engine.

### I/O Scheduler Contract

The source should not directly "go async" against some implicit runtime. It should talk to an
explicit I/O scheduler service.

```rust
pub trait IoScheduler: Send + Sync {
    fn submit(&self, req: ReadRequest) -> ReadHandle;
    fn prefetch(&self, req: ReadRequest) -> VortexResult<()>;
}

pub trait ReadHandle: Send {
    fn poll_ready(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<VortexResult<ManagedBuffer>>;
}
```

The important point is that I/O is no longer an implicit effect of polling a Rust stream. It is an
explicit service, visible to both the source and the host.

## Cancellation and Metrics

Cancellation and metrics belong in `VortexSession` because they must cross all layers:

- `DataSource`
- `Partition`
- `PartitionReader`
- `LayoutReader`
- `IoScheduler`

Design rules:

- cancellation must be cooperative and cheap to check
- a cancelled reader should stop registering new I/O immediately
- split-local tasks should observe cancellation before issuing expensive decode work
- metrics should distinguish:
  - planning
  - pruning
  - bytes read
  - bytes prefetched
  - decode CPU
  - rows produced
  - rows pruned
  - residual filter ratio

## Buffer Management

Buffer management is a first-class part of the end state.

The Scan API should not assume that Vortex always allocates from its own global allocator.

### Requirements

- Host engines may provide a memory pool or buffer manager.
- Vortex should be able to allocate scan output from that pool.
- Vortex should still support borrowing existing memory such as mmap'd or cached segment buffers.
- Returned arrays must be self-contained and safe to retain after the reader advances.
- The scan layer must not create hidden unbounded caches outside the host's view.

### Buffer Manager Contract

```rust
pub struct BufferRequest {
    pub size: usize,
    pub alignment: usize,
    pub zeroed: bool,
    pub lifetime: BufferLifetime,
}

pub enum BufferLifetime {
    Scan,
    Partition,
    Batch,
}

pub trait BufferManager: Send + Sync {
    fn allocate(&self, req: BufferRequest) -> VortexResult<ManagedBuffer>;
    fn register_external(&self, buf: ExternalBuffer) -> VortexResult<ManagedBuffer>;
}
```

The important part is not the exact API, but the policy:

- output buffers should come from the host pool when one is provided
- borrowed buffers should be registerable, not copied by default
- Vortex may use a local allocator only when the host has not supplied one

### What This Enables

- Velox memory pools
- host-side memory accounting in high-control runtimes
- Arrow-compatible allocation strategies
- future device-aware or NUMA-aware allocation

### Memory Ownership Rules

- `PartitionReader` may keep scratch buffers internal to the partition.
- `Batch` buffers must remain valid after `poll_next` returns them.
- If a batch borrows underlying segment memory, that borrow must be refcounted and visible to the
  buffer manager.
- Vortex must not require batch consumers to copy in order to outlive the scan.

## I/O

I/O is just as important as memory.

Today Vortex has layout readers, segment sources, and runtime handles. The end state should make
I/O an explicit host-pluggable service.

### Requirements

- no runtime-specific dependency
- support for local files, object stores, caches, and remote scan services
- prefetch registration
- read priority and locality hints
- support for aligned reads and zero-copy where possible
- compatibility with host-controlled I/O executors

### I/O Scheduler Contract

```rust
pub struct ReadRequest {
    pub handle: ObjectHandle,
    pub offset: u64,
    pub len: u64,
    pub alignment: usize,
    pub priority: IoPriority,
    pub intent: IoIntent,
}

pub enum IoIntent {
    Demand,
    Prefetch,
}

pub trait IoScheduler: Send + Sync {
    fn submit(&self, req: ReadRequest) -> ReadHandle;
    fn prefetch(&self, req: ReadRequest) -> VortexResult<()>;
}
```

### Design Rules

- `LayoutReader` should register reads as early as possible.
- `Split` planning should eagerly hint future ranges to the I/O manager.
- `PartitionReader` should not spin up hidden I/O threads unless the session is configured to let
  Vortex use its own default runtime.
- Remote sources are just another `IoScheduler` or `DataSource` implementation.

### Why This Matters

The host may want control over:

- object-store concurrency
- request coalescing
- cache admission
- disk queues
- remote worker RPC
- NUMA-local page placement

If `LayoutReader` goes directly to "whatever async runtime happens to exist", that control is lost.

## Prefetch and Split Lookahead

Prefetch and split lookahead belong to the source, but only within host-supplied budgets.

The host should control:

- maximum active splits
- maximum prefetched splits
- maximum prefetch bytes
- maximum in-flight reads
- maximum buffered batches

The source should control:

- which specific splits to run next
- which reads to prefetch
- how much lookahead is useful for a given layout and filter

### Why the Source Should Own This

Only the source understands:

- layout boundaries
- zone-map and pruning statistics
- sparse selection density
- expected decode cost
- remote vs local latency
- cacheability and locality
- ordering constraints
- `limit` interaction

### Probabilistic Prefetch

The source should be free to rank candidate splits with a probabilistic score, for example:

`priority ~= P(split still needed) * stall_saved / resource_cost`

Where:

- `P(split still needed)` depends on selectivity and limits, and may later incorporate
  late-arriving filter information if the Scan API grows that extension
- `stall_saved` estimates the latency hidden by early I/O
- `resource_cost` estimates bytes, memory pressure, and decode work

This is not a host-engine concern. It is a scan-source concern, constrained by host budgets.

## Scheduling

The end state separates three things that are often conflated:

- host-visible `Partition`s
- Vortex-internal `Split`s
- runtime `Task`s

### Core Rule

`Partition` is the public scheduling boundary.

`SplitBy` and `SplitPlan` are internal physical execution policy.

This gives one common model that adapts to different query engines.

### How a `Partition` Executes

1. The host opens a `Partition` with a `VortexSession` and a `ScanBudget`.
2. The resulting `PartitionReader` constructs a `SplitPlan`.
3. The reader registers I/O interest for upcoming `Split`s.
4. The reader executes `Task`s over those `Split`s.
5. The reader returns `Batch`es back to the host through `poll_next`.

### Scheduling Ownership

The host owns:

- how many `Partition`s it wants
- how many workers drive them
- how much memory is available
- how much I/O is allowed
- how much split lookahead is allowed
- cancellation and task lifetime

Vortex owns:

- layout-aware split planning
- split-local pruning
- projection/filter execution
- prefetch ranking
- split lookahead policy
- efficient materialization of output batches

### What `Partitioning` Means

`ScanOptions.partitioning` is how the host expresses what it wants the `ScanPlan` to expose.

- `Natural`: expose source-native public partitions
- `FixedCount(n)`: expose a fixed or near-fixed number of public partitions
- `SharedWork`: expose public work through shared state from which workers pull dynamically
- `HostManaged`: expose work in a form intended to be placed into a host-owned split/task queue

This enum is intentionally about host execution model, not row or byte sizing. Size and batch
targets remain separate hints on `ScanOptions`.

## Engine Integration

The same Scan API should support all of these, but not all with the same level of host control.

### DataFusion

DataFusion wants a plan with a known number of partitions, and each partition is executed as a
stream.

Recommended mapping:

- host asks for `FixedCount(n)`
- `Partition` maps to a DataFusion partition
- `open_send` is required
- the `SendPartitionReader` is wrapped as a `SendableRecordBatchStream`
- DataFusion drives polling and owns higher-level operator scheduling

Control level:

- moderate
- batch stream friendly
- less direct control over fine-grained scan scheduling

What Vortex should do:

- expose exact or approximate partition counts
- make partition size planning cheap
- respect host memory and I/O contexts if supplied
- avoid requiring any thread-affine reader state at the public boundary

If a source only supports `open_local`, a DataFusion adapter would need to bridge it through a
dedicated local thread and channels. That is an adapter escape hatch, not the preferred contract.

### DuckDB

DuckDB table functions expose global state, local thread state, and per-thread pulling of work.
Newer DuckDB paths can also report `BLOCKED` and optionally integrate with async wakeup.

Recommended mapping:

- host asks for `SharedWork`
- global table-function state owns shared scan state and the pool of remaining public work
- each local thread state uses `open_local`
- workers pull work dynamically from shared state rather than binding one static partition per
  thread
- when a reader blocks on I/O, the adapter should prefer returning `BLOCKED` to DuckDB over
  running a competing hidden Vortex scheduler
- I/O completion may happen outside a DuckDB worker, but it should only make work ready and wake
  DuckDB; follow-on scan progress should continue on DuckDB-scheduled work

DuckDB's own vector size should be reflected in:

- `target_batch_rows`
- `target_batch_bytes`

Control level:

- high control over worker-local execution
- naturally compatible with shared work and local cursors

The preferred DuckDB integration is cooperative:

- DuckDB owns worker scheduling and blocked/runnable transitions
- Vortex owns split planning, prefetch ranking, and readiness tracking inside the shared state
- Vortex should not schedule substantial scan progression onto a separate CPU runtime when the
  DuckDB scheduler can own it

### Velox

Velox already has a strong notion of split queues, task state, drivers, and memory pools.

Recommended mapping:

- host asks for `HostManaged`
- `Partition` maps closely to a Velox connector split
- `open_send` is the expected public boundary
- the reader is opened with a `VortexSession` configured with Velox memory and I/O services
- Vortex should do very little hidden scheduling
- `SplitBy` should still be used internally for physical alignment inside a Velox split

Control level:

- very high
- host should own memory, I/O, and task placement

This is the best fit for the advanced `VortexSession + ScanBudget` model.

### Polars

Polars wants a lazy scan source and may run in streaming mode, but the public embedding surface is
not as runtime-explicit as Velox.

Recommended mapping:

- host asks for `Natural` or `FixedCount(n)`
- `open_send` is typically the useful boundary
- the reader is wrapped as Arrow/Vortex batch source
- residual semantics stay in Polars where necessary

Control level:

- moderate
- simpler than Velox

## Why `Partition` and `Split` Both Exist

This is the most important conceptual distinction.

### `Partition`

Public unit of parallel work exposed by the Scan API.

Used by:

- DataFusion partitions
- DuckDB work queue items
- Velox connector splits
- Polars source partitions

### `Split`

Internal Vortex subdivision used for:

- physical alignment
- prefetch
- exact sparse range planning
- split-local pruning
- split-local filter/projection tasks

This keeps the public API stable while allowing Vortex to remain storage-aware internally.

## Pushdown and Host Functions

Pushdown is not just about syntax. It is about semantics.

The key problem is that host engines often have functions whose semantics do not exactly match
Vortex functions:

- different null behavior
- different error behavior
- different collation or timezone rules
- volatile vs stable behavior
- order sensitivity
- implicit casts

The end state must make this explicit.

### Predicate Model

The scan request should not treat all filters as equally pushdown-safe.

Conceptually:

```rust
pub struct Predicate {
    pub exact: Option<Expression>,
    pub residual: Option<Expression>,
}
```

Meaning:

- `exact`: safe for Vortex to apply fully
- `residual`: must still be evaluated by the host after scan

In practice an engine adapter may derive these from its own expression IR.

### Host Function Wrappers

Host-specific functions should be wrapped as registered Vortex functions with extra semantic
metadata.

```rust
pub trait HostFunction: Send + Sync {
    fn name(&self) -> &str;
    fn volatility(&self) -> Volatility;
    fn null_semantics(&self) -> NullSemantics;
    fn error_semantics(&self) -> ErrorSemantics;
    fn ordering_semantics(&self) -> OrderingSemantics;
    fn evaluate(&self, args: &[ArrayRef]) -> VortexResult<ArrayRef>;
    fn pruning_rule(&self) -> Option<&dyn PruningRule> { None }
}
```

The point is not that Vortex itself must invent host semantics, but that the adapter must declare
them explicitly.

### Allowed Pushdown Outcomes

For every host predicate or function call, the adapter should choose one of:

- `ExactPushdown`
  - safe to evaluate entirely inside Vortex
- `PruningOnly`
  - safe only to eliminate impossible splits, but host must still re-check rows
- `ResidualOnly`
  - not safe to push into Vortex at all
- `Unsupported`
  - reject or keep fully outside scan

### Design Rule

If semantics are even slightly unclear, prefer:

- Vortex pruning only
- or no pushdown

and keep residual evaluation in the host engine.

Correctness is more important than maximal pushdown.

## Ordering, Limits, and Future Dynamic Filters

The end state should also make the following explicit.

### Ordering

- `ordered = true` means the host needs stable row order across the entire scan.
- `ordered = false` allows the source to expose partitions in any order and emit batches in any
  order within each partition, as long as each partition remains internally correct.

This is vital for engines that want parallel scan speedups.

### Limits

Limits are host-visible semantics.

The source may use limits to reduce work, but:

- exact global limit semantics stay at the scan level
- the host may still enforce a residual limit above scan

### Future Dynamic Filters

The first-class Scan API does not need a late mutation hook for dynamic filters.

In many engines, if a filter is known early enough to affect planning, it is simply folded into
the original scan request. That should remain the default model.

However, the design should leave room for a future extension that allows late-arriving filters to
affect unread work.

If added later, the extension should be best-effort:

- it may prune unopened partitions
- it may prune or filter future splits within an open partition
- it must not require retracting already-produced batches
- it should be optional for sources to implement

## `LayoutReaderDataSource`

The current `LayoutReaderDataSource` in
[`vortex-layout/src/scan/layout.rs`](/Users/ngates/git/vortex2/vortex-layout/src/scan/layout.rs)
is the right reference implementation for the scan/source boundary.

In the end state it should:

- implement the richer `ScanRequest`
- produce `Partition`s according to `Partitioning`
- open `SendPartitionReader`s and `LocalPartitionReader`s instead of returning only streams
- use `VortexSession` services and `ScanBudget` limits when supplied
- retain `SplitBy` and split-local task execution internally

That keeps the current implementation direction intact.

## `VortexTableScan`

`VortexTableScan` should not be a monolithic runtime. It should be a `DataSource` built over:

- table metadata
- snapshots
- manifests
- one or more `LayoutReader`s

It is responsible for:

- planning partitions
- planning layout-backed reads
- exposing pushdown-safe filters and projections

It is not responsible for:

- global operator scheduling
- joins
- aggregations
- query-wide memory policy

For a future Vortex table abstraction, this is the correct layering:

- table metadata chooses which fragments/layouts participate
- layout readers execute the physical scan
- scan API exposes partitions
- host engine drives execution

## Default Vortex Runtime

The end state still needs a good out-of-the-box experience.

If the host does not configure custom services on `VortexSession`, Vortex should provide defaults
for:

- default allocator
- default coherent execution backend for I/O, progress driving, and CPU work
- default metrics sink
- default cancellation behavior

This keeps the API easy to use directly while preserving the ability for advanced hosts to take
full control.

## Design Summary

The end state should look like this:

- `LayoutReader` stays immutable, bounded, and physical.
- `DataSource` / `ScanPlan` / `Partition` remain the public Scan API.
- `SendPartitionReader` and `LocalPartitionReader` become the main execution contracts.
- `VortexSession` carries advanced host services and `ScanBudget` carries execution ceilings.
- `SplitBy` and `SplitPlan` remain internal physical execution policy.
- `Partition` is the public scheduling unit.
- host functions are pushed down only through explicit semantic wrappers.
- `VortexTableScan` is a powerful source operator, not a full query engine.

This is enough to support:

- DataFusion's partitioned execution
- DuckDB's shared-work, blocked/resume table-function model
- Velox's host-managed split and memory model
- Polars' lazy and streaming scan sources
- high-control runtimes that want custom memory, I/O, and scheduling

without forcing all of them into the same scheduler or the same runtime model.

## Compatibility

- This RFC changes the public Scan API surface. Existing `DataSource` implementations and
  `execute() -> SendableArrayStream` call sites will need migration to the new
  `ScanPlan` / `Partition` / `PartitionReader` model.
- The file format is not affected. This is purely an API and runtime change.
- Engine adapters (DataFusion, DuckDB, etc.) will need to be updated to use the new partition-based
  interface, but the migration can be done incrementally per engine.

## Drawbacks

- The poll-based `PartitionReader` model is more complex than a simple stream interface. Simple
  embeddings that do not need fine-grained control pay for that complexity.
- The dual `SendPartitionReader` / `LocalPartitionReader` contract adds surface area. Sources must
  decide which modes to support and adapters must bridge when modes do not match.
- The `ScanBudget` / `VortexSession` service model requires engines to understand and configure more
  knobs than the current API.
- Migration cost: all existing scan integrations must be updated.

The default Vortex runtime mitigates the complexity for simple use cases.

## Alternatives

- **Keep `execute() -> SendableArrayStream`**: simpler, but hides scheduling, buffering, and
  cancellation ownership. This is the status quo and the primary thing this RFC replaces.
- **Engine trait parameterization**: instead of `VortexSession` services, parameterize the scan by
  an `Engine` trait. Rejected because it introduces a second capability container that overlaps with
  `VortexSession`.
- **Fully async scan interface**: require Tokio or another async runtime. Rejected because it
  prevents embedding in sync engines and FFI contexts.
- **Single `PartitionReader` trait with `Send` bound**: simpler, but forces thread-affine hosts
  (DuckDB local state) to use unnecessary synchronization or channel bridges.

## Unresolved Questions

- Exact `BufferManager` API shape and how it interacts with Arrow memory ownership.
- Whether `ScanBudget` should be mutable or use a feedback channel for dynamic budget adjustment.
- How `SharedWork` partitioning surfaces shared state to the host (trait methods on `ScanPlan`,
  a separate `WorkPool` handle, or something else).
- How the `CpuScheduler` trait interacts with engines that want zero Vortex-internal concurrency.
- Whether `PartitionReader` should support a `size_hint` for output batches.
- Exact fallback behavior when a source implements only one of `open_send` / `open_local`.

## Future Possibilities

- **Dynamic filters**: late-arriving filters that prune unopened partitions or future splits within
  an open partition. The design leaves room for this as a best-effort, optional extension.
- **Remote scan services**: a `DataSource` backed by a remote worker that implements the same
  partition/reader contract over RPC.
- **NUMA-aware placement**: `ScanBudget` or `VortexSession` extensions for NUMA locality hints.
- **Adaptive split planning**: adjusting `SplitPlan` mid-scan based on observed selectivity or I/O
  latency.
- **Device-aware buffers**: extending `BufferManager` to support GPU or accelerator memory.
