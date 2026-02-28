- Start Date: 2026-02-25
- RFC PR: [vortex-data/rfcs#0000](https://github.com/vortex-data/rfcs/pull/0000)
- Tracking Issue: [vortex-data/vortex#0000](https://github.com/vortex-data/vortex/issues/0000)

## Summary

Replace the current execution VTable with a scheduler-driven model. `reduce` and
`reduce_parent` remain metadata-only. `execute` returns an `ExecutionStep` telling the scheduler
which child to execute next, or that execution is done. `execute_parent` is retained and returns
an `ArrayRef` in any encoding. The scheduler drives all iteration and runs reduce rules between
steps.

## Motivation

**No decompress-into-buffer.** Encodings allocate their own output during execution.
ChunkedArray must decompress each chunk separately and concatenate — an extra copy of the
entire column. Exporters (DuckDB, Arrow) can't write directly into their output format.

**Stack overflow from recursion.** The current executor recurses into children. Deep encoding
trees overflow the stack.

**Unclear execute/reduce boundary.** Some `execute_parent` implementations are metadata-only
and belong in `reduce_parent`. The boundary isn't enforced.

## Design

### reduce / reduce_parent (unchanged)

These methods keep their current signatures. They are **strictly metadata-only** — they never
read data buffers.

```rust
fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>>;
fn reduce_parent(
    array: &Self::Array,
    parent: &ArrayRef,
    child_idx: usize,
) -> VortexResult<Option<ArrayRef>>;
```

`reduce` rewrites an array using only metadata. Filter(AllTrue(x)) → x. Constant folding.

`reduce_parent` rewrites a parent from a child's perspective. Dict child pushes ScalarFn into
values. RunEnd child pulls ScalarFn through to its values. FSST child rewrites Compare by
compressing the RHS literal.

The framework runs these to a fixpoint before execution begins and between execution steps.

Implementations currently misplaced in `execute_parent` that are metadata-only (Dict + Compare,
ALP + Compare, FoR + Compare, FSST + Compare) move to `reduce_parent`. Implementations that
are really canonical-type compute kernels (Primitive + Compare, Bool + FillNull, Decimal +
Between) move into `ScalarFn::execute`.

### FilterArray

FilterArray continues to exist as a lazy wrapper. It is not subsumed by the execution method
signature.

### execute / execute_parent

```rust
fn execute(array: &Self::Array, ctx: &mut ExecutionCtx)
    -> VortexResult<ExecutionStep>;

fn execute_parent(
    array: &Self::Array,
    parent: &ArrayRef,
    child_idx: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ExecutionStep>>;
```

```rust
pub enum ExecutionStep {
    /// Ask the scheduler to execute the child at this index one step,
    /// replace it, then call execute (or execute_parent) on this array again.
    ExecuteChild(usize),

    /// Execution is complete.
    Done(Columnar),
}
```

Both `execute` and `execute_parent` return `ExecutionStep`. The encoding never recurses into
children — it yields control back to the scheduler, telling it what work is needed. The
scheduler maintains the work stack.

`execute_parent` returns `Option<ExecutionStep>`:
- `None` — the child cannot handle this parent, fall through to the parent's own `execute`.
- `Some(ExecuteChild(i))` — the child needs its own child at index `i` executed one step before
  it can handle the parent. The scheduler does this, then retries `execute_parent`.
- `Some(Done(result))` — the child handled the parent, here is the result in **any encoding**
  (not just canonical). This is critical for encoding-preserving execution.

Making `execute_parent` iterative is necessary because some parent-handling logic requires
data access to the child's own children. For example, `Slice(RunEnd(ends=PCodec(...)))`:
RunEnd's `execute_parent` sees the Slice parent and wants to binary-search its `ends` to find
the physical indices for the slice boundaries. But `ends` is PCodec-compressed — each
`scalar_at` probe during binary search would decompress the full array. Instead, RunEnd returns
`ExecuteChild(0)` to request its ends be executed first. The scheduler decompresses PCodec to
PrimitiveArray, then retries `execute_parent`. RunEnd now binary-searches canonical ends and
returns an efficiently sliced RunEnd array.

```rust
// RunEnd's execute_parent for Slice
fn execute_parent(
    re: &RunEndArray, parent: &ArrayRef, child_idx: usize, ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ExecutionStep>> {
    let Some(slice) = parent.as_opt::<SliceVTable>() else { return Ok(None) };

    if !re.ends().is_columnar() {
        // Need canonical ends for binary search
        return Ok(Some(ExecutionStep::ExecuteChild(0)));
    }

    // Ends are canonical — binary search is cheap
    let physical_start = re.find_physical_index(slice.start())?;
    let physical_end = re.find_physical_index(slice.end())?;
    let sliced = RunEndArray::new(
        re.ends().slice(physical_start..physical_end)?,
        re.values().slice(physical_start..physical_end)?,
        slice.start() + re.offset(),
        slice.len(),
    )?;
    Ok(Some(ExecutionStep::Done(Columnar::from(sliced))))
}
```

### Scheduler

```rust
fn execute_one_step(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let mut array = optimize_recursive(array)?;

    if array.as_columnar().is_some() { return Ok(array); }

    // Try execute_parent (child-driven specialized execution)
    for child_idx in 0..array.nchildren() {
        let child = array.child(child_idx);
        match child.vtable().execute_parent(&child, &array, child_idx, ctx)? {
            None => continue,
            Some(ExecutionStep::ExecuteChild(i)) => {
                // Child needs its own child executed first — do one step, retry
                let grandchild = child.child(i);
                let stepped = execute_one_step(grandchild, ctx)?;
                let new_child = child.with_child(i, stepped);
                array = array.with_child(child_idx, new_child);
                return Ok(array);
            }
            Some(ExecutionStep::Done(result)) => {
                return Ok(result.into_array());
            }
        }
    }

    // Fall through to the array's own execute
    match array.vtable().execute(&array, ctx)? {
        ExecutionStep::ExecuteChild(i) => {
            let child = array.child(i);
            let stepped = execute_one_step(child, ctx)?;
            array = array.with_child(i, stepped);
            Ok(array)
        }
        ExecutionStep::Done(result) => Ok(result.into_array()),
    }
}
```

Each call to `execute_one_step` makes one unit of progress. The top-level
`execute_to_columnar` simply loops until the result is columnar. After each step,
`optimize_recursive` runs again — reduce rules can fire on the new tree shape. For truly
bounded stack depth, an explicit work stack replaces the recursive calls.

### Per-encoding examples

**DictArray** — requests codes first, then gathers:

```rust
fn execute(dict: &DictArray, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
    if !dict.codes().is_columnar() {
        return Ok(ExecutionStep::ExecuteChild(0));
    }
    let codes = dict.codes().as_primitive();
    let gathered = gather(dict.values(), &codes, ctx)?;
    Ok(ExecutionStep::Done(gathered))
}
```

**ScalarFnArray** — requests children left-to-right, then evaluates:

```rust
fn execute(sfn: &ScalarFnArray, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
    for (i, child) in sfn.children().iter().enumerate() {
        if !child.is_columnar() {
            return Ok(ExecutionStep::ExecuteChild(i));
        }
    }
    let result = sfn.scalar_fn().execute(sfn.columnar_children(), ctx)?;
    Ok(ExecutionStep::Done(result))
}
```

**FilterArray** — requests child, then applies mask:

```rust
fn execute(filter: &FilterArray, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
    if !filter.child().is_columnar() {
        return Ok(ExecutionStep::ExecuteChild(0));
    }
    let filtered = filter.mask().apply_to(filter.child().as_canonical())?;
    Ok(ExecutionStep::Done(Columnar::Canonical(filtered)))
}
```

**BitPacked** — leaf encoding, decompresses directly:

```rust
fn execute(bp: &BitPackedArray, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
    let primitive = unpack(bp)?;
    Ok(ExecutionStep::Done(Columnar::Canonical(Canonical::Primitive(primitive))))
}
```

### Cross-step optimization

Because `optimize_recursive` runs after each child execution, patterns that only become visible
after partial execution can still be optimized.

Example: `ScalarFn(upper, [Dict(BitPacked(codes), values)])`. After the scheduler executes
BitPacked codes to PrimitiveArray, the tree becomes
`ScalarFn(upper, [Dict(Primitive(codes), values)])`. A reduce_parent rule on Dict pushes
`upper` into values — this optimization fires between steps.

### Encoding-preserving execute_parent

`execute_parent` returns `Option<ArrayRef>` in any encoding. This enables exporters to
intercept intermediate forms without decompressing.

Concrete example: `Filter(FSST(data), mask)`. FSST's `execute_parent` sees the FilterArray
parent and returns a filtered FSST array (still FSST-encoded). An exporter driving the
scheduler sees FSST on the next iteration and exports it as a DuckDB FSST vector — no
decompression needed.

Same applies to DictArray for DuckDB dictionary vector export, or any encoding where the
downstream consumer natively supports the compressed form.

### Exporter integration

Exporters drive the scheduler loop directly and inspect the array after each step:

```rust
fn export_chunk(mut array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<DuckDBVector> {
    let mut array = optimize_recursive(array)?;

    loop {
        if let Some(dict) = array.as_opt::<DictVTable>() {
            return export_dict(dict, ctx);
        }
        if let Some(fsst) = array.as_opt::<FSSTVTable>() {
            return export_fsst(fsst, ctx);
        }
        if let Some(c) = array.as_columnar() {
            return export_columnar(c, ctx);
        }

        if let Some(rewritten) = try_execute_parent(&array, ctx)? {
            array = optimize_recursive(rewritten)?;
            continue;
        }

        match array.vtable().execute(&array, ctx)? {
            ExecutionStep::ExecuteChild(i) => {
                let child = array.child(i);
                let executed = execute_to_columnar(child, ctx)?;
                array = array.with_child(i, executed.into_array());
                array = optimize_recursive(array)?;
            }
            ExecutionStep::Done(c) => return export_columnar(c, ctx),
        }
    }
}
```

If an opaque encoding decodes to DictArray during execution, the exporter discovers it on the
next iteration and exports it natively. This is impossible in a model that always produces
canonical output.

### Decompress-into-buffer

The scheduler-driven model does not natively support caller-owned output buffers (each
`Done(Columnar)` allocates its own output). The two concrete cases where this matters —
ChunkedArray concat and FSST zero-copy push — can be addressed at the framework level:

- **ChunkedArray**: The scheduler special-cases ChunkedArray by pre-allocating a single output
  buffer and copying each chunk's `Columnar` result into it. One allocation + N memcpys rather
  than N allocations + one concat. Not zero-copy, but eliminates the extra full-column copy.

- **FSST**: The extra allocation is the views array (16 bytes per string). The actual string
  data buffers are shared via `Arc` — they are not copied. The overhead is bounded and small
  relative to decompression cost.

These are framework-level optimizations, not VTable concerns. They don't require encoding
authors to think about builders.

### Execution cache in ExecutionCtx

The scheduler executes each child one step at a time, replacing it in the parent tree. This
means the result of executing an array is only visible to its immediate parent — if the same
`Arc<dyn Array>` appears as a child of multiple parents (e.g., `a < 10 & a > 5` where `a` is
the same ArrayRef), it will be executed independently each time.

`ExecutionCtx` holds a cache keyed by the raw pointer of the source array (`Arc::as_ptr()`).
The cache entry stores a clone of the source `Arc` (to pin the pointer address — preventing
deallocation and reuse) alongside the one-step execution result.

```rust
pub struct ExecutionCtx {
    // ...
    cache: HashMap<*const dyn Array, CacheEntry>,
}

struct CacheEntry {
    /// Holds a strong reference to the source array, preventing the Arc from
    /// being deallocated and its pointer reused for a different array.
    _source: ArrayRef,
    /// The result of executing the source array one step.
    result: ArrayRef,
}
```

**When to cache.** Not every array should be cached — most are executed exactly once, and
caching adds overhead (HashMap insert + Arc clone). The `Arc::strong_count` is used as a
heuristic: if `strong_count > 1`, the array is referenced from multiple places and its
execution result is likely to be reused. This is an imperfect approximation — strong_count is a
global reference count, not local to the array tree being executed. An array may have
`strong_count > 1` because the layout reader or scan builder holds a reference, not because it
appears twice in the expression tree. But it never misses a genuinely shared sub-expression (if
shared, the count is > 1), and false positives only cost memory, not correctness.

**Scope.** The cache lives in `ExecutionCtx`, not in individual arrays. Users can share an
`ExecutionCtx` across multiple independent array executions (e.g., multiple columns in a scan),
so the cache naturally deduplicates work across columns that share sub-arrays (shared
dictionaries, common filter masks). The cache is dropped when the `ExecutionCtx` is dropped.

## Compatibility

No file format or wire format changes. All changes are internal to the execution engine.

Public API breakage: the `Executable` trait and `execute::<T>()` method are replaced.
Third-party encodings must migrate. A default implementation bridging to the old path eases
migration.

## Alternatives

### Canonical builder model

Instead of returning `ExecutionStep`, `execute` could push results into a caller-owned
`CanonicalBuilder` (a closed enum mirroring `Canonical` in mutable builder form). Each encoding
decompresses directly into the builder in a single recursive descent. `execute_parent` would
also push into the builder.

This model natively supports decompress-into-buffer: ChunkedArray writes all chunks into one
builder (no concat), FSST pushes views directly into VarBinViewBuilder (zero-copy). No
iteration loop, no MAX_ITERATIONS.

However, requiring canonical output from `execute` is a structural limitation that cannot be
worked around:

- **No encoding-preserving execute_parent.** `execute_parent` must push canonical into the
  builder. FSST can't return a filtered FSST array — the encoding is always lost. Exporters
  must use encoding-specific logic outside the framework for every format they want to preserve.

- **No cross-step optimization.** Reduce runs once before the single descent. Patterns visible
  only after partial execution are missed.

- **No exporter intermediate inspection.** Exporters see the tree after reduce, before
  execution. Encodings hidden behind opaque wrappers are never visible.

- **Stack overflow.** `execute` recurses into children. Stack depth equals encoding tree depth.

The scheduler-driven model's allocation wins (encoding-preserving intermediates, cross-step
optimization, exporter interception, bounded stack) are structurally unrecoverable in the
builder model. The builder model's allocation wins (shared builder, zero-copy push) are
recoverable in the scheduler model via framework-level optimizations. This asymmetry makes the
scheduler-driven model strictly preferable.

## Unresolved Questions

- **Explicit work stack:** The scheduler can be recursive (simple, but same stack overflow risk
  as today) or use an explicit work stack (bounded depth, but more complex). The RFC assumes an
  explicit stack is feasible but doesn't specify the implementation.

- **Constants:** `ConstantArray` continues to exist. ScalarFnArray special-cases it to avoid
  expanding constants. The `Columnar` enum (`Canonical | Constant`) is preserved.
