- Start Date: 2026-04-16
- Authors: @gatesn
- RFC PR: [vortex-data/rfcs#54](https://github.com/vortex-data/rfcs/pull/54)

# Push-based Vortex File Writer

## Summary

Make the primary Vortex file writer API push-based. Users should be able to open a writer, push
array chunks or Arrow batches into it, inspect progress and estimated file size after each push, and
then finish the file when they decide to rotate. The existing stream-based API should remain as a
convenience adapter implemented on top of the push writer.

## Motivation

Many Vortex users are familiar with Parquet writer APIs, where the writer is an object with methods
that accept row groups or record batches. That model makes it straightforward to implement file
rotation:

1. Open a writer for the current output file.
2. Push chunks into it.
3. Check the estimated file size.
4. Finish the file once the estimate reaches a threshold.
5. Open a new writer and continue.

Today Vortex exposes a stream-oriented writer as the primary API. That works well when the caller
already owns a stream, but it is awkward when the caller owns an ingestion loop and wants to decide
when to rotate. The public Rust writer has a push-shaped wrapper, but internally it builds a channel,
wraps the receiving end as an `ArrayStream`, and then manually polls the stream writer while waiting
for sends to make progress. The JNI writer has a similar shape: Java pushes batches into a channel
that feeds a background stream write task.

This has several practical problems:

- The push API is harder to reason about than it looks. It is not the primary execution model; it is
  a stream writer hidden behind a channel.
- Progress reporting is incomplete. `bytes_written()` only counts data that has reached the
  underlying sink, and `buffered_bytes()` is currently reported from shared layout strategy objects
  rather than from a per-file writer state.
- Some layout strategies either cannot report buffering accurately yet or do not include their own
  buffered data. `CollectStrategy::buffered_bytes()` is currently unimplemented, and
  `RepartitionStrategy` intentionally omits its buffered chunks.
- Language bindings that are naturally push-based need to recreate the channel and background task
  pattern instead of calling a simple writer object.

The goal of this RFC is to make the user-facing model and the internal model match.

## Design

### User-facing Rust API

The main API should be an owned writer object:

```rust
let mut writer = session
    .write_options()
    .writer(&mut output, dtype.clone())
    .await?;

for chunk in chunks {
    writer.push(chunk).await?;

    if writer.progress().estimated_size() >= target_file_size {
        writer.flush().await?;
        let summary = writer.finish().await?;
        // Open the next output file and create a new writer.
    }
}

let summary = writer.finish().await?;
```

The exact constructor can remain named `writer`, but it should initialize the file and layout writer
state directly instead of creating an `ArrayStream` and storing a hidden stream-write future. The
current `VortexWriteOptions::write(write, stream)` method should remain available, but become an
adapter:

```rust
pub async fn write<W, S>(self, write: W, mut stream: S) -> VortexResult<WriteSummary>
where
    W: VortexWrite + Unpin,
    S: ArrayStream + Send + 'static,
{
    let dtype = stream.dtype().clone();
    let mut writer = self.writer(write, dtype).await?;

    while let Some(chunk) = stream.next().await {
        writer.push(chunk?).await?;
    }

    writer.finish().await
}
```

The blocking writer should wrap the same object:

```rust
let mut writer = session
    .write_options()
    .blocking(runtime)
    .writer(file, dtype)?;

writer.push(chunk)?;
writer.flush()?;
let progress = writer.progress();
let summary = writer.finish()?;
```

`push` and `flush` have separate contracts:

- `push(chunk).await` waits until the writer has accepted ownership of the chunk and there is enough
  capacity in the bounded write pipeline to continue. It is an admission-control and backpressure
  point, not a durability point.
- `flush().await` waits for all chunks accepted before the call to be driven as far as possible
  without finishing the file, writes all ready segment buffers to the underlying `VortexWrite`, and
  calls `VortexWrite::flush`.
- `finish().await` implies `flush`, emits EOF-only data such as the final layout footer, writes the
  footer and EOF marker, flushes the sink, and returns `WriteSummary`.

This lets callers keep pushing while earlier chunks are being compressed or written, while still
providing an explicit point to wait for pending buffers before making a final rotation decision.

### Progress and file size estimates

The push writer should expose progress from the per-file writer instance, not from the immutable
layout strategy configuration:

```rust
pub struct WriteProgress {
    /// Rows accepted by the writer, excluding empty chunks.
    pub rows: u64,

    /// Bytes successfully written to the underlying sink.
    pub bytes_written: u64,

    /// Bytes reserved in the file for known segments, including alignment padding.
    ///
    /// This may be ahead of `bytes_written` when segment buffers have been assigned offsets but
    /// have not yet reached the sink.
    pub reserved_file_position: u64,

    /// Bytes held by layout writers that have accepted input but have not emitted segments yet.
    pub buffered_input_bytes: u64,

    /// Best-effort estimate of footer and EOF bytes if the file were finished now.
    pub estimated_footer_bytes: Option<u64>,
}

impl WriteProgress {
    pub fn lower_bound_size(&self) -> u64 {
        self.bytes_written.max(self.reserved_file_position)
    }

    pub fn estimated_size(&self) -> u64 {
        self.lower_bound_size()
            + self.buffered_input_bytes
            + self.estimated_footer_bytes.unwrap_or(0)
    }
}
```

For file rotation, `estimated_size()` is intentionally an estimate. It should be good enough for
threshold-based rotation between pushed chunks, but it does not need to guarantee that the finished
file will be below the threshold. A single large input chunk can still make the file exceed the
target, just like a Parquet row group can.

The writer should also expose convenience methods:

```rust
impl Writer<'_> {
    pub fn rows_written(&self) -> u64;
    pub fn bytes_written(&self) -> u64;
    pub fn estimated_size(&self) -> u64;
    pub fn progress(&self) -> WriteProgress;
}
```

The current `bytes_written()` method can continue to mean bytes that reached the sink. New callers
that want rotation should use `estimated_size()` or `progress()`.

### Layout writer traits

The core change is to split layout configuration from per-file layout writer state. Today
`LayoutStrategy::write_stream` consumes a stream and returns a layout. This RFC proposes adding a
stateful writer trait:

```rust
pub trait LayoutStrategy: 'static + Send + Sync {
    fn new_writer(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn LayoutWriter>>;
}

#[async_trait]
pub trait LayoutWriter: Send {
    /// Accept a chunk into this layout writer.
    ///
    /// The returned future waits for bounded pipeline capacity, child-writer admission, and any
    /// synchronous validation or accounting needed before the caller may reuse or drop the input
    /// chunk. It must not wait for all segment I/O derived from this chunk to reach the underlying
    /// file sink.
    async fn push(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()>;

    /// Drain all data accepted before this call as far as possible without closing the layout.
    ///
    /// The returned future waits for internal work that must complete to make buffered data
    /// visible downstream: pending compression tasks, repartition buffers, buffered strategy
    /// queues, active dictionary runs, and child writer flushes. It may create a layout boundary
    /// and can therefore affect compression ratio or read granularity if called frequently.
    ///
    /// A layout writer may retain state that is intrinsically finalized at EOF, such as a stats
    /// table layout that is written only after the data layout is complete.
    async fn flush(&mut self, session: &VortexSession) -> VortexResult<()>;

    /// Finish the layout and return its footer layout.
    ///
    /// The returned future first performs `flush`-equivalent work, then emits EOF-dependent data
    /// and waits for all child layout futures needed to construct the final `LayoutRef`.
    async fn finish(
        self: Box<Self>,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef>;

    fn buffered_bytes(&self) -> u64 {
        0
    }
}
```

`LayoutStrategy` remains the reusable configuration object. `LayoutWriter` holds per-file state:
buffers, active child writers, dictionary encoders, stats accumulators, and progress counters.
`LayoutStrategy::new_writer` is intentionally synchronous: constructing layout writer state should
allocate local state and spawn any background workers it needs, but it should not perform I/O or
wait on asynchronous work. File-level asynchronous initialization belongs in `VortexWriteOptions`
because it owns the output sink.

The futures returned by `LayoutWriter` methods have specific meanings:

- `push` waits for the chunk to be admitted into bounded layout state. It may wait for child queue
  capacity, per-column fan-out queues, or enough downstream work to complete to preserve memory
  bounds. It should return before all compression and file I/O caused by the chunk is complete.
- `flush` waits for a barrier over all chunks accepted before the flush call. It should wait for
  layout-owned background work and child writer flushes needed to push ready segments downstream,
  but it must leave the writer open for future chunks.
- `finish` waits for a final flush plus EOF-only layout work, including child layout completion and
  construction of the final `LayoutRef`.

During migration, `write_stream` can be kept as a compatibility method on `LayoutStrategy`, with a
default implementation that creates a `LayoutWriter`, pushes the stream into it, and finishes it.
After all in-tree strategies implement `new_writer`, the old stream-first method can be deprecated.

### File writer internals

The push writer should own the file write state directly:

- array context
- root sequence pointer
- footer statistics accumulator
- segment sink
- root layout writer
- output writer
- progress counters

On construction, it writes the magic bytes and initializes the segment sink:

```rust
pub async fn writer<W>(
    self,
    mut write: W,
    dtype: DType,
) -> VortexResult<Writer<W>>
where
    W: VortexWrite + Unpin,
{
    write.write_all(ByteBuffer::copy_from(MAGIC_BYTES)).await?;

    let ctx = file_write_context(&self.session);
    let segment_sink = BufferedSegmentSink::new(/* ... */);
    let layout = self.strategy
        .new_writer(ctx.clone(), segment_sink.clone(), dtype.clone(), &self.session)?;

    Ok(Writer { /* ... */ })
}
```

On `push`, it filters empty chunks, updates row counts and file statistics, assigns a sequence ID,
and pushes into the root layout writer:

```rust
pub async fn push(&mut self, chunk: ArrayRef) -> VortexResult<()> {
    if chunk.is_empty() {
        return Ok(());
    }

    ensure_dtype_matches(&chunk, &self.dtype)?;
    self.rows += chunk.len() as u64;
    self.file_stats.push(&chunk)?;

    let sequence_id = self.sequence.advance();
    drive_layout_future(
        self.layout.push(sequence_id, chunk),
        &mut self.segment_recv,
        &mut self.write,
        &mut self.progress,
    )
    .await?;

    Ok(())
}
```

`drive_layout_future` is a free helper over disjoint writer fields: the layout future, segment
receiver, output sink, and progress counters. It polls the layout admission future and drains ready
segment buffers while that future is pending. This prevents the segment channel from becoming a
hidden deadlock point, but it does not require all work for the chunk to be complete before `push`
returns.

On `flush`, the writer flushes the root layout, drains ready segment buffers, and flushes the
underlying sink:

```rust
pub async fn flush(&mut self) -> VortexResult<()> {
    drive_layout_future(
        self.layout.flush(&self.session),
        &mut self.segment_recv,
        &mut self.write,
        &mut self.progress,
    )
    .await?;
    self.drain_ready_segments().await?;
    self.write.flush().await?;
    Ok(())
}
```

On `finish`, it finishes the root layout, drains segments, serializes and writes the footer, flushes
the sink, and returns `WriteSummary`.

### Segment sink

The first implementation should keep the current segment-buffer channel pattern so the file writer
can avoid requiring `W: Send`. Layout tasks can be `Send` and can emit `ByteBuffer`s into a
`SegmentSink`, while the public `Writer<W>` keeps ownership of `W` on the caller's task:

- `SegmentSink::write` still assigns segment offsets and sends buffers into an internal queue.
- `Writer::push`, `Writer::flush`, and `Writer::finish` drain ready buffers into `W` while waiting
  for layout admission, layout flush, or layout finish futures.
- Progress reports both bytes written to `W` and bytes reserved by the sink.

This does mean the writer still needs to select over layout futures and segment-buffer receives, but
that select loop is now a file-output implementation detail. It no longer exists to translate public
push calls into a fake input stream.

A future implementation can consider a direct ordered segment sink that writes to `W` from inside
`SegmentSink::write`, but that should be a deliberate optimization with explicit `W: Send` tradeoffs
rather than the initial design.

### Strategy-specific migration

Most existing layout strategies already contain state machines that can move into writer structs.

`TableStrategy` should preserve the current fan-out shape. The table writer has to drive validity
and field writers concurrently: a child layout can wait for input, output channel capacity, segment
ordering, or EOF-dependent work, and driving one field to completion before the others risks both
deadlock and a loss of per-column parallelism. A push-based `TableWriter` should therefore keep
bounded per-column queues and child tasks, or an equivalent concurrent driver. `push` should extract
the validity and field arrays from a struct chunk and wait only until those child pipelines accept
the per-column chunks.

`RepartitionStrategy` should keep `ChunksBuffer` as writer state. `push` slices incoming chunks,
accumulates enough rows and bytes for full blocks, and pushes full blocks into its child writer.
`finish` flushes the tail. Its `buffered_bytes()` can report the current `ChunksBuffer` bytes.

`BufferedStrategy` should keep its queue as writer state instead of storing an `AtomicU64` on the
shared strategy. `buffered_bytes()` then reports exact per-writer buffering.

`ChunkedLayoutStrategy` should create a child writer for each pushed chunk and finish it as a child
layout. It can preserve the current parallelism by spawning child work on the session handle, but
the parent writer remains the object receiving chunks.

`CompressingStrategy` should spawn compression work from `push`, then push compressed chunks into
its child writer as they complete in sequence order. If this is implemented with a bounded internal
queue, that queue's size should be reflected in `buffered_bytes()`.

`ZonedStrategy` should keep its stats accumulator as writer state. Each pushed chunk computes or
schedules per-chunk statistics, feeds the child writer, and records zone stats. `finish` writes the
stats table after finishing the data child, preserving the existing EOF ordering.

`DictStrategy` should replace `DictionaryTransformer` with a stateful dictionary writer that owns
the active encoder, active codes writer, and pending values writer. Dictionary rollover becomes a
normal state transition in `push`.

`CollectStrategy` should either report its collected chunks in `buffered_bytes()` or be avoided in
the default writer path where possible. A push-based API makes all-whole-stream collection more
visible, so any retained collect strategy should be explicit.

### Stream fallback

The stream API remains valuable for callers that naturally produce `ArrayStream`s. It should be a
thin fallback:

```rust
pub async fn push_stream<S>(&mut self, mut stream: S) -> VortexResult<()>
where
    S: ArrayStream + Send + 'static,
{
    while let Some(chunk) = stream.next().await {
        self.push(chunk?).await?;
    }
    Ok(())
}
```

If callers want producer and writer work to run concurrently, provide an explicit spawned adapter:

```rust
pub fn spawn_stream_writer<W, S>(
    options: VortexWriteOptions,
    write: W,
    stream: S,
) -> Task<VortexResult<WriteSummary>>
where
    W: VortexWrite + Unpin + Send + 'static,
    S: ArrayStream + Send + 'static,
{
    session.handle().spawn(async move {
        options.write(write, stream).await
    })
}
```

This makes spawning a deliberate boundary instead of a hidden part of the push API.

### Language bindings

The Java and JNI writer should call the Rust push writer directly. `NativeWriter` should hold a
`Writer<ObjectStoreWrite>` or an equivalent erased writer object instead of holding an `mpsc::Sender`
and a background stream task. `writeBatch` and `writeBatchFfi` should convert Arrow data to
`ArrayRef`, validate the schema, and call `push`.

The Java `VortexWriter` interface should add progress methods:

```java
long rowsWritten();
long bytesWritten();
long estimatedSize();
```

Spark can then rotate files using actual Vortex writer progress instead of approximating
`bytesWritten` from Arrow vector buffer sizes.

Python should gain a writer object in addition to `vx.io.write`:

```python
with vx.io.VortexWriter(path, dtype) as writer:
    for batch in batches:
        writer.write(batch)
        if writer.estimated_size() >= target_size:
            summary = writer.close()
            break
```

The existing `vx.io.write(iter, path)` API remains a convenience wrapper over this object.
This RFC deliberately does not add Java or Python file rotation helpers. Rotation policy belongs in
the application or engine integration, using the progress methods exposed by the writer.

## Compatibility

This RFC does not change the Vortex file format or wire format. Files written through the new push
writer should be byte-for-byte equivalent when the same chunks, strategy, and execution order are
used, subject to existing concurrency-driven determinism guarantees.

The initial implementation should preserve the existing stream API. Rust callers using
`VortexWriteOptions::write` should not need to change code. The current `Writer::push` and
`BlockingWriter::push` names can remain, but their internals should be replaced.

This is a breaking change for custom `LayoutStrategy` implementations if `new_writer` becomes a
required trait method. To smooth migration, add a default compatibility method first and migrate
in-tree strategies before requiring downstream implementations to update.

Performance should improve for push-based callers by removing channel hops and cooperative polling
from the public writer. Stream-based callers should see roughly the same performance once the stream
adapter is implemented on top of `push`. The main implementation risk is preserving the concurrency
that layout strategies currently get from streams and spawned nested tasks.

## Drawbacks

The main cost is implementation complexity. The current stream model lets many strategies express
their behavior as stream transforms. A push model requires explicit state structs for each strategy.
That is more code, and some of it will be less compact than the current `async_stream` blocks.

There is also a risk of reducing write parallelism if the first implementation is too synchronous.
The design should preserve parallel compression, per-column writing, and child layout work where
those matter for throughput.

Finally, progress estimates can never be exact before finishing the file because footer size,
compression results, buffering, and one large incoming chunk can all shift the final size. The API
must document that this is for operational rotation, not a hard file-size guarantee.

## Alternatives

### Keep the stream-first internals and polish the wrapper

We could keep `LayoutStrategy::write_stream` as the only real implementation path and improve the
current `Writer` wrapper. For example, we could fix `push_stream` documentation, improve
`buffered_bytes()`, and add a more careful progress object around the existing channel.

This is lower risk, but it leaves the core mismatch intact. Push callers would still interact with
a stream writer indirectly, and language bindings would still need background tasks or cooperative
polling to make progress.

### Add only a file rotation helper

Another option is to add a helper that consumes a stream and writes multiple files according to a
target size:

```rust
write_partitioned_by_size(options, stream, target_size, make_output).await?
```

This would solve one user-facing workflow, but it would not help callers that already have their
own batching, partitioning, transaction, or object naming logic. It also would not make Java,
Python, or Spark writer APIs feel like conventional file writers.

### Expose a `Sink<ArrayRef>`

We could model the writer as a `Sink<ArrayRef>` from the `futures` ecosystem. This has a good
conceptual fit, but it is less familiar to many users than a writer object with `push` and
`finish`, and it does not directly solve progress reporting. A `Sink` implementation can be added
later on top of the push writer if useful.

## Prior Art

Parquet writers in Rust, Java, and Python are generally push-oriented. Callers open a file writer,
write row groups or record batches, and close the writer to emit the footer. This makes file
rotation a responsibility of the ingestion loop and avoids requiring every caller to represent its
data as a stream.

Arrow IPC writers follow a similar object model: construct a writer around an output sink, call
`write` for each batch, then finish or close.

Vortex should keep accepting streams because Rust async pipelines can produce them naturally, but
the file writer object itself should look like these established file writer APIs.

## Unresolved Questions

- Should `Writer::writer(...)` be async, or should it keep the current synchronous constructor and
  delay writing magic bytes until the first `push`? An async constructor is cleaner because file
  initialization errors happen up front.
- How exact should `estimated_footer_bytes` try to be? We can start with `None` or a conservative
  heuristic and improve later.
- How much of the stream-first `LayoutStrategy::write_stream` API should remain public during the
  migration?

## Future Possibilities

Once the writer has per-file state and explicit progress, Vortex can add higher-level helpers for
common production workflows:

- rotate files by estimated size, row count, or partition key
- expose write progress metrics for observability
- support adaptive chunk sizing based on compression ratio
- tune compression and layout concurrency from writer options
- add `Sink<ArrayRef>` or `Sink<RecordBatch>` adapters for Rust async applications
