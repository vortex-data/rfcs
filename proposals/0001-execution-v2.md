- Start Date: 2026-02-25
- RFC PR: [vortex-data/rfcs#0000](https://github.com/vortex-data/rfcs/pull/0000)
- Tracking Issue: [vortex-data/vortex#0000](https://github.com/vortex-data/vortex/issues/0000)

## Summary

Replace the current execution model with a scheduler-driven design. `reduce` and
`reduce_parent` remain metadata-only. `execute` returns an `ExecutionStep` — requesting a child
be executed or declaring completion. The scheduler drives all iteration using an explicit work
stack, runs reduce rules between steps, and caches results for shared sub-expressions.

## Motivation

**Stack overflow from recursion.** The current executor recurses into children. Deep encoding
trees overflow the stack. The 128-iteration limit applies to the outer loop, not inner
recursive calls.

**Repeated decompression.** Some operations access the same compressed child multiple times.
Each access decompresses independently. For example, binary search on RunEnd's PCodec-compressed
ends calls `scalar_at` per probe, each decompressing the full array.

**Shared sub-expressions.** `a < 10 & a > 5` references the same array `a` twice. The current
model executes it independently for each reference.

**Unclear execute/reduce boundary.** Some `execute_parent` implementations are metadata-only
and belong in `reduce_parent`. The boundary isn't enforced.

## Design

### VTable methods

```rust
// Metadata-only rewrites. Never read data buffers.
fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>>;
fn reduce_parent(
    array: &Self::Array, parent: &ArrayRef, child_idx: usize,
) -> VortexResult<Option<ArrayRef>>;

// Execution. May read data buffers.
fn execute(
    array: &Self::Array, ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionStep>;
fn execute_parent(
    array: &Self::Array, parent: &ArrayRef, child_idx: usize, ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>>;
```

```rust
pub enum ExecutionStep {
    /// Execute the child at this index to columnar, replace it,
    /// then call execute on this array again.
    ExecuteChild(usize),

    /// Execution is complete.
    Done(Columnar),
}
```

**reduce / reduce_parent** are unchanged. Strictly metadata-only. The framework runs them to a
fixpoint before execution and between execution steps. Implementations currently misplaced in
`execute_parent` that are metadata-only (Dict + Compare, ALP + Compare, FoR + Compare, FSST +
Compare) move to `reduce_parent`.

**execute** returns `ExecutionStep`. The encoding never recurses into children — it yields
control to the scheduler.

- `ExecuteChild(i)` asks the scheduler to execute child `i` to columnar, replace it, and call
  `execute` again.
- `Done(columnar)` returns the final columnar result.

**execute_parent** returns `Option<ArrayRef>`. `None` means the child can't handle this parent.
`Some(result)` means it handled the parent — the result can be in **any encoding**, not just
canonical, enabling encoding-preserving execution (e.g., FSST returning a filtered FSST array
to a DuckDB exporter).

### Constant handling

When an encoding matches on a specific child type (e.g., `as_opt::<PrimitiveVTable>()`), a
`ConstantArray` child won't match — it's columnar but not Primitive. Encodings should either
check for constants explicitly, or — preferably — register a reduce rule that handles the
constant case before execution runs. For example:
`Dict(Constant(code), values)` → `Constant(values.scalar_at(code))`.

Note: an encoding that blindly returns `ExecuteChild(i)` when its child is already columnar
(but doesn't match the expected concrete type) will loop forever. This is trivial for the
scheduler to detect — if `ExecuteChild(i)` is returned and child `i` is already columnar, the
scheduler can abort with an error.

### Scheduler

The scheduler uses an explicit work stack, bounding stack depth regardless of encoding tree
depth.

```rust
fn execute_to_columnar(root: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Columnar> {
    let mut current = optimize_recursive(root, ctx)?;
    let mut stack: Vec<(ArrayRef, usize)> = vec![];  // (parent, child_idx)

    loop {
        // Check if current is columnar — return to parent or finish
        if let Some(c) = current.as_columnar() {
            match stack.pop() {
                None => return Ok(c),
                Some((parent, child_idx)) => {
                    current = parent.with_child(child_idx, c.into_array());
                    current = optimize_recursive(current, ctx)?;
                    continue;
                }
            }
        }

        // Try execute_parent
        if let Some(rewritten) = try_execute_parent(&current, ctx)? {
            current = optimize_recursive(rewritten, ctx)?;
            continue;
        }

        // Execute
        match current.vtable().execute(&current, ctx)? {
            ExecutionStep::ExecuteChild(i) => {
                let child = current.child(i);
                stack.push((current, i));
                current = optimize_recursive(child, ctx)?;
            }
            ExecutionStep::Done(result) => {
                current = result.into_array();
            }
        }
    }
}
```

### Execution cache

If the same `Arc<dyn Array>` appears as a child of multiple parents (e.g., `a < 10 & a > 5`),
the scheduler executes it independently each time. The execution cache deduplicates this work.

The cache lives in `ExecutionCtx`, keyed by `ByPtr<ArrayRef>` — a newtype that implements
`Hash` and `Eq` via `Arc::as_ptr()`. Because the key holds a clone of the `Arc`, the source
array cannot be deallocated while cached, so the pointer cannot be reused for a different array.

```rust
struct ByPtr(ArrayRef);

impl Hash for ByPtr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

impl PartialEq for ByPtr {
    fn eq(&self, other: &Self) -> bool {
        Arc::as_ptr(&self.0) == Arc::as_ptr(&other.0)
    }
}

pub struct ExecutionCtx {
    cache: HashMap<ByPtr, ArrayRef>,  // source → one-step result
    // ...
}
```

**When to cache.** Three options:

1. **Always cache.** Memory explodes — most arrays are executed once and never revisited.

2. **Pre-scan the tree.** Walk the tree before execution, count pointer occurrences, cache only
   shared nodes. Accurate for a single tree, but doesn't work across multiple trees executed
   with the same `ExecutionCtx` (e.g., scan columns sharing a dictionary or filter mask).

3. **`Arc::strong_count > 1` heuristic.** O(1) to check. Over-caches when external references
   exist (layout reader, scan builder) but never under-caches for a genuinely shared
   sub-expression. Works across independent tree executions within the same `ExecutionCtx`.
   False positives cost memory, not correctness.

We use option 3. The cache is dropped when the `ExecutionCtx` is dropped.

### Examples

**DictArray** — execute codes into Primitive, then gather:

```rust
fn execute(dict: &DictArray, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
    let Some(codes) = dict.codes().as_opt::<PrimitiveVTable>() else {
        return Ok(ExecutionStep::ExecuteChild(0));
    };
    let gathered = gather(dict.values(), codes, ctx)?;
    Ok(ExecutionStep::Done(gathered))
}
```

Note: if codes is a `ConstantArray`, the scheduler returns it as columnar. A reduce rule
`Dict(Constant(code), values) → Constant(values.scalar_at(code))` handles this before execute
runs.

**ScalarFnArray** — columnarize children left-to-right, then evaluate:

```rust
fn execute(sfn: &ScalarFnArray, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
    for (i, child) in sfn.children().iter().enumerate() {
        if child.as_opt::<AnyColumnar>().is_none() {
            return Ok(ExecutionStep::ExecuteChild(i));
        }
    }
    let result = sfn.scalar_fn().execute(sfn.columnar_children(), ctx)?;
    Ok(ExecutionStep::Done(result))
}
```

Left-to-right ordering is deterministic and simple. ScalarFn designers should consider this
when choosing input order — the first input is executed first.

**FilterArray** — columnarize child, then apply mask:

```rust
fn execute(filter: &FilterArray, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
    let Some(child) = filter.child().as_opt::<AnyCanonical>() else {
        return Ok(ExecutionStep::ExecuteChild(0));
    };
    let filtered = filter.mask().apply_to(child.into())?;
    Ok(ExecutionStep::Done(Columnar::Canonical(filtered)))
}
```

**BitPacked** — leaf, decompresses directly:

```rust
fn execute(bp: &BitPackedArray, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
    Ok(ExecutionStep::Done(Columnar::Canonical(Canonical::Primitive(unpack(bp)?))))
}
```

### Cross-step optimization

`optimize_recursive` runs when the scheduler pops from the stack (after a child reaches
columnar and is replaced into its parent). Reduce rules fire on the new tree shape.

Example: `ScalarFn(upper, [Dict(BitPacked(codes), values)])`. Pre-execution optimization fires
a reduce_parent rule on Dict, pushing `upper` into values. The tree becomes
`Dict(BitPacked(codes), upper(values))`. Dict then executes its codes into Primitive and
gathers.

### Encoding-preserving execution

`execute_parent` returns results in any encoding. Exporters drive the scheduler loop and
inspect the tree after each step. If an encoding the exporter cares about becomes visible
(DictArray for DuckDB dictionary vectors, FSST for DuckDB FSST vectors), the exporter
intercepts it without decompressing.

### Removing ExecutionCtx from VTable methods

The `execute` and `execute_parent` signatures shown above accept `&mut ExecutionCtx`. This gives
encodings the ability to recursively execute children, bypassing the scheduler's caching and
cross-step optimization. Nothing in the type system prevents it.

A stronger design: remove `ExecutionCtx` from the VTable method signatures entirely. The
scheduler owns the execution state (cache, tracing). `execute` receives no context. The method signature itself
communicates "return a step, don't execute anything."

This also eliminates the current ergonomic friction of
`let ctx = session.create_execution_ctx(); array.execute(&mut ctx)` — callers just call the
scheduler directly.

If `execute_parent` also yields `ExecutionStep` (see unresolved questions), the same argument
applies: it gets resource access but not execution power. The scheduler is the only code that
drives execution.

### Decompress-into-buffer

This model does not support caller-owned output buffers. Each `Done(Columnar)` allocates its
own output. ChunkedArray cannot share a builder across chunks, and FSST cannot push views
directly into a caller's VarBinViewBuilder. This is a trade-off we accept in exchange for
encoding-preserving execution, cross-step optimization, and bounded stack depth.

## Alternatives

### Canonical builder model

`execute` pushes results into a caller-owned `CanonicalBuilder` (closed enum mirroring
`Canonical` in mutable builder form). Single recursive descent, no iteration loop. Natively
supports decompress-into-buffer: ChunkedArray writes all chunks into one builder, FSST pushes
views directly into VarBinViewBuilder.

However, always producing canonical output has structural limitations:

- **No encoding-preserving execution.** `execute_parent` pushes into a builder, so results are
  always canonical. FSST can handle a FilterArray parent (fused filter + decompress into the
  builder), but it can't return a filtered FSST array for a DuckDB exporter. Exporters that
  want to preserve an encoding must use encoding-specific logic outside the framework.
- **No cross-step optimization.** Reduce runs once before the single descent.
- **Stack overflow.** Recurses into children; stack depth equals encoding tree depth.

The scheduler model's wins (encoding preservation, cross-step optimization, bounded stack) are
structurally unrecoverable in the builder model. The builder model's win (zero-copy
decompress-into-buffer) is a real cost of the scheduler model, but one we accept given the
asymmetry.

## Unresolved Questions

- **Explicit work stack details.** The scheduler sketch shows the concept. The exact data
  structure (e.g., handling multiple `ExecuteChild` calls from the same parent for different
  children) needs design work.

- **Iterative execute_parent.** The current design has `execute_parent` return
  `Option<ArrayRef>`. An alternative is to return `Option<ExecutionStep>`, allowing it to
  request child execution before handling the parent. The execution cache may be sufficient for
  cases where `execute_parent` needs data access to children (e.g., RunEnd binary-searching
  compressed ends — the cache ensures decompression happens once). We have no compelling
  example for or against iterative `execute_parent` yet.

- **Targeted child execution.** `ExecuteChild(i)` currently executes the child to columnar.
  An alternative is to allow the encoding to specify a matcher for when it should be re-entered
  — e.g., `ExecuteChildInto(i, ArrayId)` would execute child `i` until it matches a specific
  encoding or reaches columnar, whichever comes first. This enables early re-entry before full
  columnarization — for example, an encoding could request its child be executed until it becomes
  a DictArray, then operate on the dictionary directly. The trade-off is additional complexity in
  the scheduler and ExecutionStep enum.

- **FilterArray.** FilterArray continues to exist as a lazy wrapper. It is not subsumed by the
  execution method. Whether to unify Filter/Slice/Take wrappers is orthogonal.
