- Start Date: 2026-02-25
- RFC PR: [vortex-data/rfcs#0000](https://github.com/vortex-data/rfcs/pull/0000)
- Tracking Issue: [vortex-data/vortex#0000](https://github.com/vortex-data/vortex/issues/0000)

## Summary

Replace the current execution VTable with a cleaner model. `reduce` and `reduce_parent` remain
metadata-only in both proposals and are identical. The proposals differ in how `execute` and
`execute_parent` work:

- **Proposal A (Scheduler-driven)**: `execute` returns an `ExecutionStep` — either requesting
  the scheduler to execute a specific child, or declaring that it is done. The scheduler drives
  all iteration. `execute_parent` is retained and returns an `ArrayRef` (any encoding).

- **Proposal B (Canonical builder)**: `execute` pushes its result into a caller-owned
  `CanonicalBuilder`. The result is always canonical. `execute_parent` is retained and also
  pushes into the builder.

## Motivation

**No decompress-into-buffer.** Encodings allocate their own output during execution.
ChunkedArray must decompress each chunk separately and concatenate — an extra copy of the
entire column. Exporters (DuckDB, Arrow) can't write directly into their output format.

**Stack overflow from recursion.** The current executor recurses into children. Deep encoding
trees overflow the stack.

**Unclear execute/reduce boundary.** Some `execute_parent` implementations are metadata-only
and belong in `reduce_parent`. The boundary isn't enforced.

## Shared design

### reduce / reduce_parent (identical in both proposals)

Both proposals keep these methods with unchanged signatures. They are **strictly metadata-only**
— they never read data buffers.

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

The framework runs these to a fixpoint before execution begins (and, in Proposal A, between
execution steps).

Implementations currently misplaced in `execute_parent` that are metadata-only (Dict + Compare,
ALP + Compare, FoR + Compare, FSST + Compare) move to `reduce_parent`. Implementations that
are really canonical-type compute kernels (Primitive + Compare, Bool + FillNull, Decimal +
Between) move into `ScalarFn::execute`.

### FilterArray

FilterArray continues to exist as a lazy wrapper in both models. It is not subsumed by the
execution method. Both models handle it — the difference is what `execute_parent` can return
when a child encoding wants to handle a FilterArray parent (see comparison section).

---

## Proposal A: Scheduler-driven iterative execution

### VTable

```rust
fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>>;
fn reduce_parent(array: &Self::Array, parent: &ArrayRef, child_idx: usize)
    -> VortexResult<Option<ArrayRef>>;

fn execute(array: &Self::Array, ctx: &mut ExecutionCtx)
    -> VortexResult<ExecutionStep>;

fn execute_parent(array: &Self::Array, parent: &ArrayRef, child_idx: usize, ctx: &mut ExecutionCtx)
    -> VortexResult<Option<ArrayRef>>;
```

```rust
pub enum ExecutionStep {
    /// Ask the scheduler to execute the child at this index to columnar,
    /// replace it, then call execute on this array again.
    ExecuteChild(usize),

    /// Execution is complete.
    Done(Columnar),
}
```

The encoding never recurses into children. It yields control back to the scheduler, telling it
what work is needed. The scheduler maintains the work stack.

### Scheduler

```rust
fn execute_to_columnar(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Columnar> {
    let mut array = optimize_recursive(array)?;

    loop {
        if let Some(c) = array.as_columnar() { return Ok(c); }

        // Try execute_parent (child-driven specialized execution)
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
            ExecutionStep::Done(result) => return Ok(result),
        }
    }
}
```

After each child execution, `optimize_recursive` runs again — reduce rules can fire on the
new tree shape. For truly bounded stack depth, an explicit work stack replaces the recursive
`execute_to_columnar(child)` call.

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

---

## Proposal B: Canonical builder execution

### VTable

```rust
fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>>;
fn reduce_parent(array: &Self::Array, parent: &ArrayRef, child_idx: usize)
    -> VortexResult<Option<ArrayRef>>;

fn execute(
    array: &Self::Array,
    builder: &mut CanonicalBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>;

fn execute_parent(
    array: &Self::Array,
    parent: &ArrayRef,
    child_idx: usize,
    builder: &mut CanonicalBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool>;  // true if handled
```

### CanonicalBuilder

A closed enum mirroring `Canonical` in mutable builder form:

```rust
pub enum CanonicalBuilder {
    Null(NullBuilder),
    Bool(BoolBuilder),
    Primitive(PrimitiveBuilder),
    Decimal(DecimalBuilder),
    VarBinView(VarBinViewBuilder),
    List(ListViewBuilder),
    FixedSizeList(FixedSizeListBuilder),
    Struct(StructBuilder),
    Extension(ExtensionBuilder),
}
```

Because it is an enum (not `dyn`), encodings can match on the concrete variant. This enables
zero-copy decompression paths impossible with the current `dyn ArrayBuilder`.

### Framework

```rust
fn canonicalize(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
    let array = optimize_recursive(array)?;
    let mut builder = CanonicalBuilder::new(array.dtype());

    // Try execute_parent
    for child_idx in 0..array.nchildren() {
        let child = array.child(child_idx);
        if child.vtable().execute_parent(&child, &array, child_idx, &mut builder, ctx)? {
            return Ok(builder.finish());
        }
    }

    array.vtable().execute(&array, &mut builder, ctx)?;
    Ok(builder.finish())
}
```

Single call. No iteration loop. No MAX_ITERATIONS. Reduce runs once, then one recursive
descent fills the builder.

### Per-encoding examples

**FSST** — zero-copy push into VarBinViewBuilder:

```rust
fn execute(fsst: &FSSTArray, builder: &mut CanonicalBuilder, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    let CanonicalBuilder::VarBinView(vbv) = builder else { unreachable!() };
    let (buffers, views) = fsst_decompress(fsst, vbv.completed_block_count())?;
    vbv.push_buffers_and_views(&buffers, &views);
    Ok(())
}
```

**ChunkedArray** — shared builder, no concat:

```rust
fn execute(chunked: &ChunkedArray, builder: &mut CanonicalBuilder, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    for chunk in chunked.chunks() {
        canonicalize_into(chunk, builder, ctx)?;
    }
    // All chunks wrote into the same builder. No concatenation needed.
    Ok(())
}
```

**DictArray** — executes codes into sub-builder, then gathers:

```rust
fn execute(dict: &DictArray, builder: &mut CanonicalBuilder, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    let mut code_builder = CanonicalBuilder::new(dict.codes().dtype());
    canonicalize_into(dict.codes(), &mut code_builder, ctx)?;
    let codes = code_builder.finish().into_primitive();
    gather_into_builder(dict.values(), &codes, builder, ctx)
}
```

**FilterArray** — execute child, apply mask:

```rust
fn execute(filter: &FilterArray, builder: &mut CanonicalBuilder, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    let mut sub = CanonicalBuilder::new(filter.child().dtype());
    canonicalize_into(filter.child(), &mut sub, ctx)?;
    let canonical = sub.finish();
    let filtered = filter.mask().apply_to(canonical)?;
    builder.extend_from_canonical(&filtered);
    Ok(())
}
```

**BitPacked** — unpack directly into PrimitiveBuilder:

```rust
fn execute(bp: &BitPackedArray, builder: &mut CanonicalBuilder, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    let CanonicalBuilder::Primitive(pb) = builder else { unreachable!() };
    unpack_into(bp, pb)?;
    Ok(())
}
```

---

## Comparison

### What Proposal A can do that Proposal B cannot

**1. Encoding-preserving execute_parent.**

In A, `execute_parent` returns `Option<ArrayRef>` — the result can be in *any* encoding. In B,
`execute_parent` pushes into a `CanonicalBuilder` — the result is always canonical.

Concrete example: `Filter(FSST(data), mask)`. FSST's `execute_parent` sees the FilterArray
parent.

- In A, it can return a filtered FSST array (still FSST-encoded). An exporter driving the
  scheduler sees FSST on the next iteration and exports it as a DuckDB FSST vector — no
  decompression.
- In B, it must push decompressed VarBinView into the builder. The FSST encoding is gone.

Same applies to DictArray for DuckDB dictionary vector export, or any encoding where the
downstream consumer natively supports the compressed form.

**2. Cross-step optimization.**

In A, the scheduler runs `optimize_recursive` after each child execution. Reduce rules fire on
the new tree shape. Patterns only visible after partial execution can still be optimized.

Example: `ScalarFn(upper, [Dict(BitPacked(codes), values)])`. After the scheduler executes
BitPacked codes to PrimitiveArray, the tree becomes
`ScalarFn(upper, [Dict(Primitive(codes), values)])`. A reduce_parent rule on Dict pushes
`upper` into values — this optimization fires between steps.

In B, reduce runs once before the single recursive descent. If a pattern only becomes visible
after partial execution, it's missed.

**3. Bounded stack depth.**

In A, the scheduler can use an explicit work stack instead of recursion. Stack depth is O(1)
regardless of encoding depth. In B, `execute` recurses into children — stack depth equals
encoding tree depth.

### What Proposal B can do that Proposal A cannot

**1. Caller-owned output buffers (decompress-into).**

The builder is created by the caller and passed down. This enables three concrete optimizations:

- **ChunkedArray without concat.** One builder, each chunk writes into it. Single contiguous
  allocation. In A, each chunk returns its own Columnar; the framework must concat them — an
  extra copy of the entire column.

- **FSST zero-copy push.** FSST decompresses views and data buffers directly into
  VarBinViewBuilder's internal storage, adjusting view offsets to account for already-completed
  blocks. In A, FSST must allocate its own VarBinViewArray.

- **Pre-allocated output.** When output length is known, the builder pre-allocates once. In A,
  each Done allocates independently.

**2. No intermediate allocations.**

Every encoding writes directly into the final output buffer. No temporary array trees between
steps. For hot paths on large batches, this is measurable.

**3. No iteration bound.**

Single recursive descent. No MAX_ITERATIONS. No risk of non-convergence.

### Summary table

| Capability                          | Proposal A | Proposal B |
|-------------------------------------|------------|------------|
| Caller-owned output buffers         | No         | **Yes**    |
| ChunkedArray without concat         | No         | **Yes**    |
| FSST zero-copy into builder         | No         | **Yes**    |
| Pre-allocated output                | No         | **Yes**    |
| No intermediate allocations         | No         | **Yes**    |
| Encoding-preserving execute_parent  | **Yes**    | No         |
| Exporter intercepts intermediates   | **Yes**    | No         |
| Cross-step optimization             | **Yes**    | No         |
| Bounded stack depth (explicit stack)| **Yes**    | No         |
| No iteration bound / convergence    | No         | **Yes**    |

### The core trade-off

Proposal A optimizes for **flexibility**: intermediate encodings are visible, exporters can
intercept, reduce rules fire between steps. The cost is that every encoding allocates its own
output and ChunkedArray must concat.

Proposal B optimizes for **allocation efficiency**: caller-owned buffers, zero-copy
decompression, no intermediates. The cost is that execution always produces canonical — encodings
can't preserve themselves through execution, and exporters must use encoding-specific logic
outside the framework.

## Compatibility

No file format or wire format changes. All changes are internal to the execution engine.

Public API breakage: the `Executable` trait and `execute::<T>()` method are replaced.
Third-party encodings must migrate. A default implementation bridging to the old path eases
migration.

## Unresolved Questions

- **CanonicalBuilder design (Proposal B):** The exact interface for one-level-deep building,
  especially for List and Extension types, needs design work. Struct fields are accumulated as
  `Vec<ArrayRef>` (potentially compressed), not recursively built.

- **Explicit work stack (Proposal A):** The scheduler can be recursive (simple, but same stack
  overflow risk as today) or use an explicit work stack (bounded depth, but more complex).
  The RFC assumes an explicit stack is feasible but doesn't specify the implementation.

- **Constants:** `ConstantArray` continues to exist. ScalarFnArray special-cases it to avoid
  expanding constants. The `Columnar` enum (`Canonical | Constant`) is preserved. This is
  orthogonal to the proposal choice.

- **Decompression cache / CSE:** `ExecutionCtx` can hold a pointer-identity cache so that shared
  sub-arrays (e.g., `x + x`) are only executed once. Orthogonal to the proposal choice.
