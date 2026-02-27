- Start Date: 2026-02-25
- RFC PR: [vortex-data/rfcs#0000](https://github.com/vortex-data/rfcs/pull/0000)
- Tracking Issue: [vortex-data/vortex#0000](https://github.com/vortex-data/vortex/issues/0000)

## Summary

Replace the current execution model with a cleaner architecture that:
- Enables decompress-into-buffer (caller-owned output) for zero-copy decompression
- Passes a `Mask` to the execution method, forcing encodings to handle row selection
- Cleans up the reduce/execute boundary and removes migration-era adaptor boilerplate

Two candidate designs are presented: **Option A** (three-phase recursive) and **Option B**
(cleaned-up iterative). Both share the `Mask`-based row selection and the reduce boundary cleanup. They
differ in how materialization is structured -- Option A passes a caller-owned builder through
recursive descent, Option B keeps an iterative loop with typed execution steps.

## Motivation

**No decompress-into-buffer.** The current execution model has no way to decompress an array
into a caller-provided output buffer. Each encoding allocates its own output during execution.
This means ChunkedArray must decompress each chunk separately and concatenate the results --
an extra copy of the entire column. Exporters (DuckDB, Arrow) cannot write directly into their
output format; they must receive Vortex's output and copy it. This is the primary pain point.

**Stack overflows from recursive execution.** The current executor is recursive: each
`execute` call may recurse into children, which recurse into their children. On deep array
trees (deeply nested expressions, many-chunk arrays) this overflows the stack. The recursion
depth is bounded by a 128-iteration limit, but this limit applies to the outer loop -- inner
recursive calls have no bound.

**Easy to forget selection implementations.** Filter (boolean mask), Slice (contiguous range),
and Take (integer indices) are separate adaptor traits. Each encoding can implement one and
forget the others. Fusing selection with decompression is really just
`execute_parent(FilterArray)` -- which is fine -- but the current design makes it easy to miss.
Passing a `Mask` parameter to the execution method forces every encoding to handle
selection, eliminating the gap.

**Adaptor explosion is migration noise.** The adaptor types (FilterReduceAdaptor,
SliceExecuteAdaptor, TakeExecuteAdaptor, etc.) are a bridge from the old compute kernel
dispatch to the new VTable world. The reduce/execute distinction is already enforced -- we just
haven't deprecated the APIs that don't require an `ExecutionCtx`. The ~40+ adaptor impls are
migration artifacts, not a fundamental design problem, but they add noise and make the codebase
harder to navigate.

**Strict metadata-only reduce is useful for GPU.** When buffers live on a GPU device,
host-side code cannot easily access them. A clean metadata-only reduce boundary is useful
(though not strictly required) for GPU execution paths.

## Shared design

Both options share these design elements.

### Mask as row selection

The existing `Mask` type (in `vortex-mask`) already represents row selection with optimized
internal variants: `AllTrue(usize)`, `AllFalse(usize)`, and `Values(Arc<MaskValues>)` for a
full bitmap. It already supports `slice()` for restricting to a range. No new type is needed --
`Mask` is passed directly to the execution method as the row selection parameter.

Composition is mask-over-mask: slicing a mask produces a narrower mask. The main addition is a
`compose` method (or equivalent) for restricting one mask by another, and `apply` for
compacting buffer elements.

```rust
// Existing type, extended with:
impl Mask {
  /// Restrict self by other: keep only positions where both are true.
  pub fn compose(&self, other: &Mask) -> Mask;

  /// Apply mask to a buffer, returning a new buffer with only selected elements.
  pub fn apply<T>(&self, data: &Buffer<T>) -> BufferMut<T>;
}
```

**Take is not selection.** Take (integer indices) can reorder rows, repeat rows, and introduce
nulls via out-of-bounds indices. These semantics are fundamentally different from subsetting.
Gather remains an encoding-internal concern -- DictArray handles codes → values lookup inside
its own materialization.

**`scalar_at` remains a separate VTable method.** Single-element access via a builder
round-trip would regress hot paths like RunEnd binary search probing individual positions.

### CanonicalBuilder

`CanonicalBuilder` is specifically for the decompress-into case: a caller creates a builder,
and the encoding writes directly into it. This is structurally incompatible with iterative
execution (where each step returns an owned result), so `CanonicalBuilder` only applies to
Option A's recursive model.

It replaces the current `dyn ArrayBuilder` with a closed enum. The current `ArrayBuilder` is a
trait object -- callers get a `Box<dyn ArrayBuilder>` and can only interact with it through
generic methods like `append_scalar()` and `extend_from_array()`. Encodings cannot know what
concrete builder they are writing into, so they must produce intermediate arrays or scalars and
let the builder copy the data in.

`CanonicalBuilder` is an enum that mirrors the `Canonical` variants in mutable builder form:

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

impl CanonicalBuilder {
  pub fn new(dtype: &DType) -> Self { ... }
  pub fn finish(self) -> Canonical { ... }
}
```

Because it is an enum (not `dyn`), encodings can match on it to access the concrete builder
type directly. This enables zero-copy decompression paths that are impossible with `dyn
ArrayBuilder`. For example, FSST can match for the `VarBinView` variant and push decompressed
buffers and views directly into the builder's internal storage, avoiding an intermediate string
allocation:

```rust
fn canonicalize_into(fsst: &FSSTArray, builder: &mut CanonicalBuilder, mask, ctx) {
  let CanonicalBuilder::VarBinView(vbv_builder) = builder else {
    unreachable!("FSST must canonicalize into VarBinView");
  };
  let (buffers, views) = fsst_decode_views(fsst, vbv_builder.completed_block_count(), ctx)?;
  vbv_builder.push_buffer_and_adjusted_views(&buffers, &views, ...);
}
```

`CanonicalBuilder` is only canonical at the root level. For Struct, field arrays are
accumulated as `Vec<ArrayRef>` (potentially still compressed/chunked) rather than being
recursively decompressed into nested builders.

### Reduce boundary cleanup

Both options retain `reduce` and `reduce_parent` unchanged:

```rust
fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>>;
fn reduce_parent(
  array: &Self::Array,
  parent: &ArrayRef,
  child_idx: usize,
) -> VortexResult<Option<ArrayRef>>;
```

Both are strictly metadata-only, never reading data buffers. This boundary is already enforced
today -- the remaining work is deprecating the APIs that don't require an `ExecutionCtx` and
migrating the misplaced implementations. Specifically: the metadata-only rewrites currently
registered as `execute_parent` (Dict + Compare, ALP + Compare, FoR + Compare, FSST + Compare)
move to `reduce_parent`. The canonical-type kernels currently in `execute_parent` (Primitive +
Compare, Bool + FillNull, Decimal + Between) move into the scalar function's `execute` method,
where they belong.

### Constants

`ConstantArray` continues to exist as an encoding. It is special-cased only in ScalarFnArray
materialization -- one check to avoid expanding constants before passing to the scalar function.
This preserves the `Columnar` enum:

```rust
pub enum Columnar {
  Canonical(Canonical),
  Constant(ConstantArray),
}
```

**Open question:** The current scalar representation uses heap-allocated value trees for nested
types (structs, lists). For deeply nested types, this is expensive. An alternative is to hold
constants as length-1 arrays instead of scalars, avoiding the scalar value tree entirely. This
is orthogonal to the execution model choice but may affect `ConstantArray`'s internal
representation.

### Decompression cache and CSE

`ExecutionCtx` holds a pointer-identity cache (`HashMap<*const dyn Array, ArrayRef>`) scoped
to a single materialization call. When the same array node appears as a child of multiple
ScalarFnArray nodes (e.g., `x + x`), it is materialized once. When a ZStd-compressed array is
sliced into many chunks, the first slice decompresses the block and caches the result. The
cache is dropped after materialization returns.

---

## Option A: Three-phase recursive model

### VTable surface (3 methods)

```rust
pub trait VTable {
  fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>>;
  fn reduce_parent(
    array: &Self::Array,
    parent: &ArrayRef,
    child_idx: usize,
  ) -> VortexResult<Option<ArrayRef>>;

  /// Materialize this array into a canonical builder, fusing the given row mask
  /// with decompression. This is where all buffer-reading work happens.
  fn canonicalize_into(
    array: &Self::Array,
    builder: &mut CanonicalBuilder,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
  ) -> VortexResult<()>;
}
```

| Method             | Purpose                                     | Buffer access |
|--------------------|---------------------------------------------|---------------|
| `reduce`           | Self-rewrite (metadata only)                | Never         |
| `reduce_parent`    | Rewrite parent (expression push-down, etc.) | Never         |
| `canonicalize_into`| Decompress with fused mask                  | Yes           |

### Three-phase pipeline

```rust
pub fn canonicalize(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
  // Phase 1: optimize tree (reduce rules, expression push-down)
  let array = array.optimize_recursive()?;

  // Phase 2: peel selection wrappers
  let (array, mask) = array.peel_selection();

  // Phase 3: single recursive descent, no loop
  let mut builder = CanonicalBuilder::new(array.dtype());
  array.vtable().canonicalize_into(&array, &mut builder, &mask, ctx)?;
  Ok(builder.finish())
}
```

**Phase 1** runs `reduce`/`reduce_parent` in a loop until the tree is stable. All
metadata-only rewrites happen here: expression push-down into Dict values, constant folding,
comparison target transformation for ALP/FoR/FSST.

**Phase 2** strips Filter and Slice wrappers from the root, composing them into a single
`Mask`. *(Open question: does lifting selection to the root provide meaningful benefit
beyond what passing `Mask` through recursion already gives? It may be orthogonal to the
core proposal.)*

**Phase 3** is a single recursive descent. Each encoding decompresses with the fused selection.
No iteration. Selection wrappers deeper in the tree compose themselves into the selection
during descent:

```rust
// SliceArray's canonicalize_into: slicing composes as masking out the excluded rows
fn canonicalize_into(slice: &SliceArray, builder, mask, ctx) {
  let composed = mask.slice(slice.range());
  slice.child().canonicalize_into(builder, &composed, ctx)
}
```

### Per-encoding canonicalize_into

- **BitPacked**: unpack only selected positions into PrimitiveBuilder
- **FSST**: decompress only selected strings into VarBinViewBuilder
- **RunEnd**: binary-search ends to find runs intersecting the selection, expand only those
- **Dict**: materialize codes with selection, gather from values
- **Chunked**: split selection across chunks, recurse per-chunk
- **Struct**: pass selection to each field's materialization
- **Constant**: create scalar repeated to selection count

For fused decompression patterns (e.g., Dict-RLE), the parent checks inline:

```rust
fn canonicalize_into(dict: &DictArray, builder, mask, ctx) {
  if let Some(runend) = dict.codes().as_opt::<RunEndVTable>() {
    return fused_dict_rle(dict.values(), runend, builder, mask, ctx);
  }

  let mut code_builder = primitive_builder(dict.codes().dtype());
  dict.codes().canonicalize_into(&mut code_builder, mask, ctx)?;
  let codes = code_builder.finish_into_primitive();
  take_into_builder(builder, dict.values(), &codes, ctx)
}
```

### ScalarFnArray materialization

```rust
fn canonicalize_into(
  array: &ScalarFnArray,
  builder: &mut CanonicalBuilder,
  mask: &Mask,
  ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
  let children: Vec<Columnar> = array.children().iter()
          .map(|child| {
            if let Some(c) = child.as_opt::<ConstantVTable>() {
              return Ok(Columnar::Constant(c.clone()));
            }
            let mut child_builder = CanonicalBuilder::new(child.dtype());
            child.canonicalize_into(&mut child_builder, mask, ctx)?;
            Ok(Columnar::Canonical(child_builder.finish()))
          })
          .try_collect()?;

  let result = array.scalar_fn().execute(children, ctx)?;
  builder.extend_from_array(&result);
  Ok(())
}
```

### Trade-offs

**Strengths:**

- **Minimal VTable surface.** 3 methods, down from 5. Each has a clear, non-overlapping role.
- **No iteration loop.** Materialization is a single recursive descent. Predictable, easy to
  debug, easy to reason about performance. (Note: still recursive -- see weakness below.)
- **Mask fused with decompression.** Each encoding handles row selection once, in one place.
- **Strict reduce/canonicalize boundary.** reduce = metadata-only. canonicalize_into =
  buffer-reading. No ambiguity, works on GPU.

**Weaknesses:**

- **No encoding-preserving mask application.** `canonicalize_into` always decompresses into a
  `CanonicalBuilder`. There is no VTable mechanism for an encoding to absorb a mask without
  decompressing. Exporters that want to preserve an encoding (DuckDB FSST vectors, Arrow
  dictionary arrays) must call encoding-specific filter methods outside the framework. See
  trade-off #2 for the concrete FSST example.

- **Fused patterns require parent-knows-child.** Dict must check `as_opt::<RunEndVTable>()` to
  fuse Dict-RLE. An external encoding can't volunteer a fused path for a built-in parent
  without modifying the parent. The known fused patterns (Dict-RLE) are all between encodings
  in `vortex-array`, so this isn't a practical problem today, but it limits extensibility.

- **Exporters can't inspect intermediates.** After optimize + peel, the exporter sees the tree
  as-is. If the root is an opaque encoding that would decode to Dict, the exporter can't
  discover that -- Phase 3 goes straight to canonical. The DuckDB dictionary export use case
  only works when DictArray is already visible after Phase 1, which is the common case (Vortex
  files store Dict as DictArray) but not guaranteed.

- **Still recursive.** `canonicalize_into` recurses into children. On deeply nested trees
  (many layers of encoding, deeply nested structs/lists), this can still overflow the stack --
  the same class of problem the current executor has. An explicit work stack or trampoline
  could mitigate this but adds implementation complexity.

---

## Option B: Cleaned-up iterative model

Keep the framework-driven iteration loop but clean up the VTable surface, enforce the
reduce/execute boundary, and thread `Mask` through execution.

### VTable surface (3 methods)

```rust
pub trait VTable {
  fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>>;
  fn reduce_parent(
    array: &Self::Array,
    parent: &ArrayRef,
    child_idx: usize,
  ) -> VortexResult<Option<ArrayRef>>;

  /// Take one execution step toward materialization.
  /// Returns an ExecutionStep telling the framework what happened.
  fn execute(
    array: &Self::Array,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
  ) -> VortexResult<ExecutionStep>;
}
```

```rust
pub enum ExecutionStep {
  /// Fully materialized.
  Done(Columnar),
  /// Rewrote to a new array tree. Framework should optimize and execute again.
  Rewrite(ArrayRef),
  /// Composed the mask and delegated to child. Framework should continue
  /// executing the returned array with the new mask.
  Peel(ArrayRef, Mask),
}
```

| Method    | Purpose                                     | Buffer access |
|-----------|---------------------------------------------|---------------|
| `reduce`  | Self-rewrite (metadata only)                | Never         |
| `reduce_parent` | Rewrite parent (expression push-down)  | Never         |
| `execute` | One step toward materialization              | Yes           |

### Framework loop

```rust
pub fn execute_to_columnar(
  mut array: ArrayRef,
  mut mask: Mask,
  ctx: &mut ExecutionCtx,
) -> VortexResult<Columnar> {
  for _ in 0..MAX_ITERATIONS {
    // Check termination
    if let Some(c) = array.as_opt::<ConstantVTable>() {
      return Ok(Columnar::Constant(c.apply_mask(&mask)));
    }
    if let Some(c) = array.as_opt::<AnyCanonical>() {
      return Ok(Columnar::Canonical(mask.apply_to(c.into())?));
    }

    // Optimize (reduce/reduce_parent to fixpoint)
    array = array.optimize_recursive()?;

    // Execute one step
    match array.vtable().execute(&array, &mask, ctx)? {
      ExecutionStep::Done(c) => return Ok(c),
      ExecutionStep::Rewrite(new_array) => {
        array = new_array;
      }
      ExecutionStep::Peel(child, new_mask) => {
        array = child;
        mask = new_mask;
      }
    }
  }
  vortex_bail!("exceeded max iterations")
}
```

Note: `optimize_recursive()` runs inside the loop, after every step. This means reduce_parent
rules can fire on intermediate forms exposed by execution -- the key capability that Option A
lacks.

### Per-encoding execute

Each encoding returns a typed step, making execution transparent:

```rust
// FilterArray: compose masks, delegate
fn execute(filter: &FilterArray, mask, ctx) -> ExecutionStep {
  let composed = mask.compose(&filter.mask());
  ExecutionStep::Peel(filter.child().clone(), composed)
}

// SliceArray: slice the mask, delegate
fn execute(slice: &SliceArray, mask, ctx) -> ExecutionStep {
  let composed = mask.slice(slice.range());
  ExecutionStep::Peel(slice.child().clone(), composed)
}

// BitPacked: fully decompress with mask
fn execute(bp: &BitPackedArray, mask, ctx) -> ExecutionStep {
  let mut builder = primitive_builder(bp.dtype());
  unpack_selected(bp, &mut builder, mask)?;
  ExecutionStep::Done(Columnar::Canonical(builder.finish().into()))
}

// Dict: decompress one step -- execute codes, then take
fn execute(dict: &DictArray, mask, ctx) -> ExecutionStep {
  if !dict.codes().is_canonical() {
    // Execute codes one step, rebuild dict
    let stepped_codes = dict.codes().execute_one_step(mask, ctx)?;
    let new_dict = DictArray::new(stepped_codes, dict.values().clone());
    return ExecutionStep::Rewrite(new_dict.into_array());
  }
  // Codes are canonical -- do the take with mask
  let result = take_canonical(dict.values(), dict.codes().as_primitive(), mask, ctx)?;
  ExecutionStep::Done(Columnar::Canonical(result))
}

// ScalarFnArray: execute children, then apply function
fn execute(sfn: &ScalarFnArray, mask, ctx) -> ExecutionStep {
  // Find first non-columnar child, execute it one step
  for (i, child) in sfn.children().iter().enumerate() {
    if !child.is_columnar() {
      let stepped = child.execute_one_step(mask, ctx)?;
      let new_sfn = sfn.replace_child(i, stepped);
      return ExecutionStep::Rewrite(new_sfn.into_array());
    }
  }
  // All children are columnar -- apply function
  let children = sfn.children_as_columnar(mask);
  let result = sfn.scalar_fn().execute(children, ctx)?;
  ExecutionStep::Done(Columnar::Canonical(result.into()))
}
```

### Fused patterns via reduce_parent

Because `optimize_recursive()` runs after every step, fused patterns work naturally. Dict
executes its codes one step (RunEnd → Primitive), then on the next iteration,
`optimize_recursive()` runs and reduce_parent rules can fire on the new tree. The Dict-RLE
fused path can be implemented as a reduce_parent rule on RunEnd that recognizes the Dict parent
and rewrites the tree, or Dict can check inline in its execute -- both work because the
framework loops.

### Exporter tree inspection

Exporters drive the loop directly, inspecting after each step:

```rust
fn export_chunk(mut array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<DuckDBVector> {
  let mut mask = Mask::new_true(array.len());

  for _ in 0..MAX_ITERATIONS {
    array = array.optimize_recursive()?;
    let (peeled, m) = array.peel_selection();
    array = peeled;
    mask = mask.compose(&m);

    if let Some(dict) = array.as_opt::<DictVTable>() {
      return export_dict(dict, &mask, ctx);
    }
    if array.is_canonical() {
      return export_canonical(mask.apply_to(array)?, ctx);
    }

    match array.vtable().execute(&array, &mask, ctx)? {
      ExecutionStep::Done(c) => return export_columnar(c, ctx),
      ExecutionStep::Rewrite(new) => { array = new; }
      ExecutionStep::Peel(child, m) => { array = child; mask = m; }
    }
  }
  vortex_bail!("exceeded max iterations")
}
```

The exporter inspects the tree after every step. If an opaque encoding decodes to DictArray
during execution, the exporter sees it on the next iteration and can export it as a dictionary.

### Trade-offs

**Strengths:**

- **Encoding-preserving mask application.** An encoding can implement `execute(mask)` to
  absorb the mask and return `Rewrite(still_encoded)`. FSST can filter without decompressing,
  and exporters see the result through the standard loop -- no encoding-specific logic needed
  for the mask-application step.
- **Exporters inspect intermediates.** The DuckDB exporter can find DictArray even when it's
  hidden behind an opaque encoding, because it inspects after each step.
- **Child-volunteers extensibility preserved.** A fused decompression pattern can be expressed
  as a reduce_parent rule or an inline check, without closing the door on external encodings
  discovering patterns through the iteration.
- **Same 3 VTable methods.** reduce, reduce_parent, execute. Clean.
- **No stack overflow risk.** The outer loop is iterative -- each step returns control to the
  framework. Stack depth is bounded to one execution step, regardless of tree depth.

**Weaknesses:**

- **Iteration loop remains.** Execution is still iterative with a MAX_ITERATIONS bound. Each
  step produces a new array tree, and the framework loops. This is more predictable than before
  (no execute_parent, Mask threaded through, typed ExecutionStep) but still requires
  tracing through iterations to understand execution. The recursive model's "one call, done"
  is easier to reason about.
- **Intermediate allocations.** Each `Rewrite` step allocates a new array tree. The recursive
  model writes directly into the builder with no intermediates. For hot paths (large batches,
  tight loops), the allocation overhead may be measurable.
- **Mask and the loop interact awkwardly.** When an encoding returns `Peel(child,
  new_mask)`, the mask changes mid-loop. The framework must track the current mask across
  iterations. In the recursive model, the mask flows naturally through function arguments.
- **No decompress-into-buffer.** The iterative model returns `Columnar` from `Done` -- the
  encoding owns its output. There is no way to pass a caller-owned `CanonicalBuilder` through
  the loop, because the framework doesn't know whether this step is final or intermediate.
  This is the primary pain point that motivated this RFC, and Option B does not address it.
  ChunkedArray still can't share a builder across chunks. Exporters still can't write directly
  into their output format.

---

## Fundamental trade-offs

The two options differ on three architectural axes. These are genuine trade-offs -- each option
wins on some axes and loses on others. They are not fixable with clever engineering; they follow
from the structural decision of recursion vs. iteration.

### 1. Caller-owned output buffers (Option A wins)

In Option A, the caller creates a `CanonicalBuilder` and passes it down through the recursive
descent. Every encoding writes directly into this caller-owned buffer. The buffer exists before
materialization begins and is filled during a single pass.

In Option B, each encoding returns `ExecutionStep::Done(Columnar)` -- it allocates and owns its
output. The caller receives the finished result. There is no way to pass a builder through the
loop because the encoding doesn't know at call time whether this is the final step or an
intermediate rewrite that will be fed back into the loop.

**Why this matters:**

- **ChunkedArray concat avoidance.** With a caller-owned builder, ChunkedArray iterates its
  chunks and each chunk writes into the same builder. The output is a single contiguous
  allocation. Without it, each chunk produces its own `Columnar`, and the framework must
  concatenate them -- an extra copy of the entire column.

- **FSST zero-copy push.** FSST can decompress string views and data buffers directly into a
  `VarBinViewBuilder`'s internal storage, adjusting view offsets in-place. This avoids
  materializing an intermediate string array. With `Done(Columnar)`, FSST must allocate the
  `VarBinView` canonical array itself, losing the ability to share buffer space with other
  chunks or avoid a final copy.

- **Pre-allocated output.** When the output length is known (common case: the selection count),
  the builder can pre-allocate once. The iterative model allocates per step.

This trade-off is structural: passing a mutable builder down requires a call stack (recursion).
A loop that yields intermediate values cannot thread a mutable builder through iterations
without fundamentally changing the `ExecutionStep` API (e.g., adding a `Done`-with-builder
variant), which would complicate the common case where an encoding returns `Rewrite`.

### 2. Encoding-preserving mask application (Option B wins)

Some encodings can apply a row mask **without decompressing**. FSST can filter its compressed
data and produce a smaller FSST array. This matters when the downstream consumer natively
supports the encoding -- DuckDB supports FSST vectors, so exporting a filtered FSST directly
avoids decompression entirely.

**Concrete example:** `filter(scalar_fn(upper, [fsst(data)]), mask)`

Both options share the reduce phase. FSST's reduce_parent pushes `upper` into the symbol
table (metadata-only, analogous to Dict pushing scalar functions into values). After reduce +
peel, both options see: `fsst_upper(data)` + mask.

In **Option B**, FSST's standard `execute(mask)` method handles this:

```rust
fn execute(fsst: &FSSTArray, mask, ctx) -> ExecutionStep {
  let filtered = fsst.filter_preserving(&mask)?;  // still FSST
  ExecutionStep::Rewrite(filtered.into_array())
}
```

The DuckDB exporter drives the loop, sees FSST on the next iteration, and exports directly.
The standard path would execute again next iteration and decompress -- no harm. Any encoding
can implement this pattern through the general VTable.

In **Option A**, the exporter must use encoding-specific logic outside the framework:

```rust
fn export_chunk(array: ArrayRef, ctx: &mut ExecutionCtx) -> DuckDBVector {
  let array = array.optimize_recursive()?;
  let (array, mask) = array.peel_selection();

  // Encoding-specific: exporter must know about FSST
  if let Some(fsst) = array.as_opt::<FSSTVTable>() {
    let filtered = fsst.filter_preserving(&mask)?;
    return export_fsst(filtered, ctx);
  }

  let mut builder = CanonicalBuilder::new(array.dtype());
  array.canonicalize_into(&mut builder, &mask, ctx)?;
  export_canonical(builder.finish())
}
```

The general `canonicalize_into` path always decompresses -- there is no VTable mechanism for
an encoding to absorb a mask without decompressing. The exporter must know about each encoding
it wants to preserve and call encoding-specific methods.

**The difference:** Option B handles encoding-preserving mask application through the standard
`execute` VTable method -- any encoding can participate. Option A requires the exporter to have
encoding-specific knowledge for each format it wants to preserve. In practice, exporters
already need encoding-specific logic to export to native formats (DuckDB FSST vectors, Arrow
dictionary arrays), so the additional burden is the mask-application call rather than just an
`as_opt` check.

This extends to cross-crate encodings. An external encoding that can filter itself without
decompressing just implements `execute(mask) → Rewrite(filtered_self)` in Option B. In
Option A, every exporter that wants to preserve that encoding must be updated to call its
specific filter method.

### 3. Exporter intermediate inspection (Option B wins)

In Option B, exporters drive the loop directly and can inspect the array tree after each step.
After an encoding applies a mask via `Rewrite`, the exporter sees the result on the next
iteration and can export it in its native format.

In Option A, exporters see the tree after Phase 1 (optimize) and Phase 2 (peel selection).
When the encoding the exporter cares about is already visible after peel (the common case --
Dict is stored as DictArray, FSST is stored as FSSTArray), the exporter can intercept and
handle it with encoding-specific logic. This covers the typical Vortex file scan path. The
gap is when an encoding is hidden behind a layer that must be executed to reveal it, or when
the exporter wants the encoding to absorb the mask through the general VTable (see trade-off
#2 above).

### Summary

| Trade-off | Winner | Practical impact today |
|-----------|--------|----------------------|
| Caller-owned output buffers | **Option A** | **Concrete.** ChunkedArray concat avoidance and FSST zero-copy decompression are measurable. |
| Encoding-preserving mask application | **Option B** | **Concrete.** FSST filter-without-decompress, Dict filter-without-decompress. Exporters need encoding-specific logic in Option A. |
| Exporter intermediate inspection | **Option B** | **Moderate.** Exporters can discover encodings revealed by execution. Fixed by peel_selection when the filter is at the root. |

Secondary differences:

| Aspect | Option A | Option B |
|--------|----------|----------|
| Stack overflow risk | Still recursive (can overflow on deep trees) | Iterative loop, bounded stack depth |
| Intermediate allocations | None (direct to builder) | One array tree per `Rewrite` step |
| Debuggability | One recursive call, stack trace shows position | Loop iterations, requires tracing |
| Mask threading | Natural function argument | Mutable state across iterations |
| Fused patterns (Dict-RLE) | Parent checks child inline | reduce_parent on intermediates, or inline |
| GPU kernel fusion potential | Clean tree maps to single kernel | Step-by-step harder to fuse |
| Determinism | Always same call sequence | Iteration count varies with input |

## What changes (shared)

**Removed (both options):**

- `execute_parent` VTable method -- metadata-only rewrites move to `reduce_parent`;
  canonical-type kernels move into `ScalarFn::execute`.
- `append_to_builder` VTable method.
- Selection-related adaptors -- `FilterReduceAdaptor`, `FilterExecuteAdaptor`,
  `SliceReduceAdaptor`, `SliceExecuteAdaptor`, and all their per-encoding implementations.
  `TakeExecuteAdaptor` is also removed (Take/gather is not row masking; it remains an
  encoding-internal concern).

**Retained (both options):**

- `reduce` / `reduce_parent` -- strictly metadata-only.
- `ScalarFnArray` -- per-node lazy computation with `reduce_parent` push-down.
- Filter/Slice array wrappers -- for lazy representation. Could be unified into a single
  `MaskedSelectionArray` in the future. Take remains separate (gather semantics, not masking).
- `scalar_at` -- separate VTable method for single-element access.

**New (both options):**

- `Mask::compose()` and `Mask::apply()` methods on the existing `Mask` type.
- `Array::peel_selection()` -- strip selection wrappers from root.

**New (Option A only):**

- `CanonicalBuilder` -- one-level-deep builder enum for decompress-into.
- `canonicalize_into` VTable method -- single recursive descent with fused mask.

**New (Option B only):**

- `ExecutionStep` enum -- typed return from execute, replacing bare `ArrayRef`.
- `execute` takes `&Mask` parameter -- row mask threaded through the loop.

## Worked example: DuckDB dictionary export

A Vortex file scan produces a chunk with a dictionary column. The query has a filter predicate
(`name = 'alice'`) evaluated to a mask, and a projection (`upper(name)`). The array tree:

```
filter(
    scalar_fn(upper, [
        dict(
            codes: bitpacked([0,1,0,2,1,0,...]),
            values: fsst(["alice","bob","charlie"])
        )
    ]),
    mask
)
```

**Optimize.** `reduce_parent` pushes `upper` into dict values:

```
filter(
    dict(
        codes: bitpacked([0,1,0,2,1,0,...]),
        values: scalar_fn(upper, [fsst(["alice","bob","charlie"])])
    ),
    mask
)
```

**Peel selections.** Filter is at the root, so it peels:

```
array = dict(codes: bitpacked(...), values: scalar_fn(upper, [fsst(...)]))
mask = filter_mask
```

**Export (Option A).** The exporter inspects the tree after optimize + peel:

```rust
fn export_chunk(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<DuckDBVector> {
  let array = array.optimize_recursive()?;
  let (array, mask) = array.peel_selection();

  if let Some(dict) = array.as_opt::<DictVTable>() {
    return export_dict(dict, &mask, ctx);
  }

  let mut builder = CanonicalBuilder::new(array.dtype());
  array.canonicalize_into(&mut builder, &mask, ctx)?;
  export_canonical(builder.finish())
}
```

**Export (Option B).** The exporter drives the loop, inspecting after each step:

```rust
fn export_chunk(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<DuckDBVector> {
  let (mut array, mut mask) = (array, Mask::new_true(array.len()));
  for _ in 0..MAX_ITERATIONS {
    array = array.optimize_recursive()?;
    let (peeled, m) = array.peel_selection();
    array = peeled;
    mask = mask.compose(&m);

    if let Some(dict) = array.as_opt::<DictVTable>() {
      return export_dict(dict, &mask, ctx);
    }
    if array.is_canonical() {
      return export_canonical(mask.apply_to(array)?);
    }
    match array.vtable().execute(&array, &mask, ctx)? {
      ExecutionStep::Done(c) => return export_columnar(c),
      ExecutionStep::Rewrite(a) => { array = a; }
      ExecutionStep::Peel(a, m) => { array = a; mask = m; }
    }
  }
  vortex_bail!("export did not converge")
}
```

Both options produce the same result: DuckDB gets a dictionary vector with 3 uppercase values
and only the selected codes. The difference is that Option B can discover DictArray even if it
was hidden behind an opaque encoding, while Option A only sees what's visible after Phase 1.

## Migration path

The migration is the same for both options. The key difference is Phase 3/5 -- whether
encodings implement `canonicalize_into` (Option A) or return `ExecutionStep` (Option B).

**Phase 1: Foundation types.** Add `Mask::compose()`/`Mask::apply()`, introduce `CanonicalBuilder`,
`Array::peel_selection()`. All additive.

**Phase 2: Add new execution method with fallback.** Add `canonicalize_into` (Option A) or
typed `execute` (Option B) with a default that bridges to the old path. Both old and new paths
work.

**Phase 3: Migrate canonical encodings.** One per PR. Independent, parallelizable.

**Phase 4: Migrate selection wrappers.** Filter, Slice, Chunked, Masked compose into
`Mask`.

**Phase 5: Migrate compressed encodings.** Mask-fused decompression. Independent per
encoding.

**Phase 6: Move execute_parent → reduce_parent.** Metadata-only rewrites become reduce rules.
Canonical-type kernels move into ScalarFn::execute. Remove execute_parent.

**Phase 7: Delete old path.** Remove old `execute` method, the `Executable` trait, and all
selection adaptors.

**Phase 8: Wire up exporters.** DuckDB, Arrow exporters migrate to new API.

## Compatibility

This RFC does not change the file format or wire format. All changes are internal to the
execution engine.

**Public API breakage:**

- `Executable` trait and `execute::<T>()` are removed. Callers migrate to `canonicalize()` /
  `execute_to_columnar()`.
- Third-party encodings must migrate to the new execution method. The fallback default
  (Phase 2) provides a bridge during migration.

## Drawbacks

- **Migration effort.** ~33 encodings must migrate. The phased approach with a fallback default
  mitigates this.

- **Mask composition complexity.** `Mask::compose` must be correct for all combinations of
  internal representations (`AllTrue`, `AllFalse`, `Values`). This is concentrated, well-tested
  complexity, but getting it wrong causes data corruption.

## Alternatives

### Keep the iterative execution model as-is

Address individual pain points without restructuring. Avoids migration cost but does not
solve decompress-into-buffer and perpetuates the adaptor boilerplate.

### Use a single ExpressionArray instead of per-node ScalarFnArray

Prevents `reduce_parent` from interacting with individual expression nodes -- the key
optimization for dictionary-encoded predicates.

### Use `dyn ArrayBuilder` instead of `CanonicalBuilder` enum

More extensible but prevents zero-copy builder paths. FSST pushing directly into
`VarBinViewBuilder` depends on knowing the builder variant at compile time.

### Subsume `scalar_at` into Mask

Deferred. Single-element access via a builder round-trip would regress hot paths. Revisit
after profiling.

## Prior Art

- **DataFusion** has `PhysicalExpr::evaluate_selection`, threading an explicit selection
  through expression evaluation.

## Unresolved Questions

- **Mask composition correctness**: `Mask::compose` and `Mask::slice` interactions need
  careful implementation and thorough testing.

- **CanonicalBuilder design**: The exact interface for one-level-deep building, especially for
  List and Extension types, needs design work.

- **Aggregate and window functions**: Out of scope. AggregateFnArray and WindowFnArray will
  need their own materialization paths.

- **Option A vs Option B**: This RFC presents both. The decision reduces to: are caller-owned
  output buffers (ChunkedArray concat avoidance, FSST zero-copy decompression) worth giving up
  encoding-preserving mask application through the general VTable (FSST filter-without-decompress,
  Dict filter-without-decompress for exporters)? Both trade-offs are concrete. In Option A,
  exporters need encoding-specific filter methods; in Option B, ChunkedArray can't share a
  builder across chunks.

## Future Possibilities

### Mask pull-up

When a filter mask is nested inside an expression (`scalar_fn(upper, [filter(dict(...), mask)])`),
`peel_selection()` can't extract it. A future optimization could pull masks out of expressions
to the root, enabling further optimizations.

### Aggregate and window functions

`AggregateFnArray` and `WindowFnArray` will follow a similar deferred pattern with
`reduce`/`reduce_parent` optimization and `Mask`-based materialization.

### GPU kernel fusion

The strict metadata-only reduce constraint means the optimized tree can be shipped to a GPU
context. A GPU-aware materialization path could fuse the entire optimized tree into a single
kernel launch.
