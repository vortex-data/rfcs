- Start Date: 2026-02-25
- RFC PR: [vortex-data/rfcs#0000](https://github.com/vortex-data/rfcs/pull/0000)
- Tracking Issue: [vortex-data/vortex#0000](https://github.com/vortex-data/vortex/issues/0000)

## Summary

Replace the current iterative `execute`/`execute_parent` execution loop with a three-phase
pipeline -- **optimize**, **peel selections**, **canonicalize** -- that reduces the VTable
surface from five methods to three, unifies Filter/Slice/Take into a composable `Selection`
type fused with decompression, and replaces the executor loop with a single recursive descent
through a new `canonicalize_into` VTable method.

## Motivation

The current execution model has five VTable extension points: `reduce`, `reduce_parent`,
`execute`, `execute_parent`, and `append_to_builder`. Each encoding can implement all five, and
adaptor types (Filter, Slice, Take, Compare, Cast, FillNull, Between, Mask, Zip, Like, Not,
ListContains) each come in Reduce and Execute variants. This leads to a combinatorial explosion
of implementations -- Dict alone has 7 parent reduce rules and 3 parent execute kernels.

**The distinction between extension points is unclear.** The stated rule is "reduce =
metadata-only, execute = may read buffers," but many operations blur this boundary. Some
`execute_parent` implementations (Dict + Compare, ALP + Compare, FSST + Compare) are
metadata-only tree rewrites that belong in `reduce_parent`. Others (Primitive + Compare,
Bool + FillNull) are just the scalar function's default implementation on canonical inputs --
redundant with the function's own `execute` method.

**Selection operations are the same thing expressed differently.** Filter (boolean mask), Slice
(contiguous range), and Take (integer indices) are all row selection. The per-encoding
selection logic itself is inherent -- each encoding needs to know how to select rows -- but
today it is spread across three separate adaptor traits. Unifying them into a single
`Selection` parameter on `canonicalize_into` eliminates the trait boilerplate and makes it harder
for an encoding to implement Slice but forget Take.

**The executor loop is hard to reason about.** The iterative reduce → reduce_parent →
execute_parent → execute loop runs up to 128 iterations. Each iteration may rewrite the tree
in surprising ways. Debugging requires tracing through multiple levels of indirection to
understand which vtable method fired and why.

**Strict metadata-only reduce is required for GPU.** When buffers live on a GPU device,
host-side code *cannot* access them. The boundary between reduce (metadata-only) and execute
(buffer-accessing) is well-defined, but several existing implementations violate it -- some
`execute_parent` implementations that are actually metadata-only tree rewrites are registered
as execute rather than reduce. The new model eliminates execute entirely, making violations
impossible.

Concrete use cases that are painful today:

- **DuckDB dictionary export**: The DuckDB exporter wants to export dictionary-encoded columns
  as DuckDB dictionary vectors. Today it must drive the executor loop, hoping it converges to a
  DictArray, with no guarantee of success. With the new model, it runs optimize → peel → inspect
  for DictArray in a predictable sequence.

- **Selection-fused decompression**: BitPacked, FSST, and RunEnd can decompress only selected
  rows, but today each must implement separate Filter, Slice, and Take adaptor traits. The
  selection logic itself is the same, but with a unified `Selection` parameter it is harder
  for an encoding to forget to implement one of the three selection forms, and the boilerplate
  of three separate trait impls is eliminated.

- **GPU execution**: The reduce/reduce_parent phase must be strictly metadata-only so it can run
  on arrays whose buffers live on a GPU. The current model does not enforce this.

## Design

### VTable surface

The new model has three VTable methods with clear, non-overlapping roles:

```rust
pub trait VTable {
    /// Rewrite this array to a simpler form without touching data buffers.
    /// Returns Ok(None) when no rewrite is possible.
    fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>>;

    /// Rewrite the parent array using knowledge of this child's encoding.
    /// Used for expression push-down, cast push-down, constant folding, etc.
    /// Must not access data buffers.
    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;

    /// Materialize this array into a canonical builder, fusing the given row selection
    /// with decompression. This is where all O(n) buffer-reading work happens.
    fn canonicalize_into(
        array: &Self::Array,
        builder: &mut CanonicalBuilder,
        selection: &Selection,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;
}
```

| Method             | Called by         | Purpose                                     | Buffer access |
|--------------------|-------------------|---------------------------------------------|---------------|
| `reduce`           | `optimize()` loop | Self-rewrite                                | Never         |
| `reduce_parent`    | `optimize()` loop | Rewrite parent (expression push-down, etc.) | Never         |
| `canonicalize_into`| `canonicalize()`  | Decompress with fused selection             | Yes           |

`reduce` and `reduce_parent` are unchanged from the current model. `canonicalize_into` replaces
`execute`, `execute_parent`, and `append_to_builder`. The top-level `canonicalize()` free
function runs the full three-phase pipeline (optimize → peel → `canonicalize_into`).

### Selection

```rust
pub enum Selection {
    /// All rows are selected.
    All,
    /// A single row by index (replaces scalar_at).
    Index(usize),
    /// A contiguous range of rows.
    Range(Range<usize>),
    /// A boolean mask (true = selected).
    Mask(Mask),
    /// Arbitrary indices, possibly repeated, possibly nullable.
    /// A null index produces a null row in the output.
    Indices(PrimitiveArray),
}

impl Selection {
    /// Compose two selections. self is applied first, then further restricted by other.
    fn compose(&self, other: &Selection) -> Selection;
    /// Number of selected rows.
    fn count(&self) -> usize;

    /// Apply selection in-place, compacting the selected elements to the front.
    fn apply_in_place<T>(&self, data: &mut [T]);
    /// Apply selection to a buffer, returning a new buffer with only selected elements.
    /// Takes &Buffer<T> (not &[T]) so the output alignment is known.
    fn apply<T>(&self, data: &Buffer<T>) -> BufferMut<T>;
}
```

`Index` and `Range` avoid allocating a `Mask` for trivial cases (single element lookup,
contiguous slice).

The first four variants (`All`, `Index`, `Range`, `Mask`) are pure row subsetting -- the
output is always a subset of the input, in input order, with no repetition. `Indices` is
semantically different: it is a **gather** that can repeat positions, reorder rows, and insert
nulls via null indices. This is not an ideal unification -- gather has different semantics from
selection (output can be larger than input, row order is caller-defined, nullability is
introduced). However, the alternative is a separate gather API that every encoding must
implement alongside `canonicalize_into`, which recreates the combinatorial problem this RFC
aims to eliminate. DictArray is used both as genuine dictionary encoding (small values, many
codes) and as a general-purpose take over large arrays (e.g., defining a take mask over a full
Vortex file, with null indices to insert null rows). The latter case requires `Indices` --
materializing all values with `Selection::All` would be catastrophic when values is an entire
file column.

Encodings can handle `Indices` as a fallback: canonicalize with `Selection::All`, then gather
from the canonical result. Performance-critical encodings (BitPacked, FSST) can optimize the
gather path directly, just as they implement `TakeExecute` today.

The `Index` variant subsumes `scalar_at` -- getting a single element is just materializing with
a one-row selection:

```rust
pub fn scalar_at(array: &dyn Array, idx: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let mut builder = CanonicalBuilder::new(array.dtype());
    array.canonicalize_into(&mut builder, &Selection::Index(idx), ctx)?;
    builder.finish().into_array().scalar_at(0)  // trivial on canonical
}
```

### CanonicalBuilder

`CanonicalBuilder` replaces the current `dyn ArrayBuilder` with a closed enum. The current
`ArrayBuilder` is a trait object -- callers get a `Box<dyn ArrayBuilder>` and can only interact
with it through generic methods like `append_scalar()` and `extend_from_array()`. Encodings
cannot know what concrete builder they are writing into, so they must produce intermediate
arrays or scalars and let the builder copy the data in.

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
fn canonicalize_into(fsst: &FSSTArray, builder: &mut CanonicalBuilder, selection, ctx) {
    let CanonicalBuilder::VarBinView(vbv_builder) = builder else {
        unreachable!("FSST must canonicalize into VarBinView");
    };
    let (buffers, views) = fsst_decode_views(fsst, vbv_builder.completed_block_count(), ctx)?;
    vbv_builder.push_buffer_and_adjusted_views(&buffers, &views, ...);
}
```

`CanonicalBuilder` is only canonical at the root level. For Struct, field arrays are
accumulated as `Vec<ArrayRef>` (potentially still compressed/chunked) rather than being
recursively decompressed into nested builders. This generalizes the `pack_struct_chunks`
pattern that already exists in the chunked canonicalization path.

### The three-phase pipeline

Canonicalization is a three-phase pipeline with no executor loop:

```rust
pub fn canonicalize(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
    // Phase 1: optimize tree (reduce rules, expression push-down)
    let array = array.optimize_recursive()?;

    // Phase 2: peel selection wrappers
    let (array, selection) = array.peel_selection();

    // Phase 3: canonicalize_into (single recursive descent, no loop)
    let mut builder = CanonicalBuilder::new(array.dtype());
    array.canonicalize_into(&mut builder, &selection, ctx)?;
    Ok(builder.finish())
}
```

**Phase 1** runs `reduce`/`reduce_parent` in a loop until the tree is stable. This handles all
metadata-only rewrites: expression push-down into Dict values, RunEnd values; constant folding;
comparison target transformation for ALP, FoR, FSST; structural simplification. No data buffers
are read.

**Phase 2** calls `array.peel_selection()` to strip Filter, Slice, and Take wrappers from the
root, composing them into a single `Selection`. The remaining array tree has no selection
wrappers at its root.

**Phase 3** is a single recursive descent through `canonicalize_into`. Each encoding decompresses
with the fused selection. There is no iteration -- each encoding knows how to decompress in one
call. Selection wrappers encountered deeper in the tree compose themselves into the selection
during the recursive descent.

### Selection composition during materialization

Selection wrappers (Slice, Filter) still exist as array types for lazy representation (e.g., a
file reader produces sliced arrays). During materialization, their `canonicalize_into` composes the
wrapper into the selection and delegates to the child:

```rust
// SliceArray's canonicalize_into
fn canonicalize_into(slice: &SliceArray, builder, selection, ctx) {
    let composed = selection.compose(Selection::Range(slice.range()));
    slice.child().canonicalize_into(builder, &composed, ctx)
}

// FilterArray's canonicalize_into
fn canonicalize_into(filter: &FilterArray, builder, selection, ctx) {
    let composed = selection.compose(Selection::Mask(filter.mask()));
    filter.child().canonicalize_into(builder, &composed, ctx)
}
```

Phase 2 (`peel_selection`) is only needed when the caller wants to inspect the array before
materializing (e.g., the DuckDB exporter checking for DictArray). During normal
materialization, selection composition happens naturally during the recursive descent.

### ScalarFnArray materialization

ScalarFnArray's `canonicalize_into` materializes its children, then evaluates the function on
the results:

```rust
fn canonicalize_into(
    array: &ScalarFnArray,
    builder: &mut CanonicalBuilder,
    selection: &Selection,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let children: Vec<Columnar> = array.children().iter()
        .map(|child| {
            // Constants pass through without expansion
            if let Some(c) = child.as_opt::<ConstantVTable>() {
                return Ok(Columnar::Constant(c.clone()));
            }
            let mut child_builder = CanonicalBuilder::new(child.dtype());
            child.canonicalize_into(&mut child_builder, selection, ctx)?;
            Ok(Columnar::Canonical(child_builder.finish()))
        })
        .try_collect()?;

    let result = array.scalar_fn().execute(children, ctx)?;
    builder.extend_from_array(&result);
    Ok(())
}
```

By the time `canonicalize_into` runs, `reduce_parent` has already pushed operations into compressed
domains. Dict's `reduce_parent` rewrites `compare(dict(codes, values), x)` into
`dict(codes, compare(values, x))`, so the comparison only runs on unique values, not every row.

### Constants

Constants are inherently special. "This value repeated N times without N copies" requires
*someone* to know about broadcasting, and no representation trick eliminates that.

`ConstantArray` continues to exist as an encoding. Its `canonicalize_into` expands the value
into the builder, just like any other encoding decompresses. The only place constants get
special treatment is ScalarFnArray's `canonicalize_into` -- one check to avoid expanding
constants before
passing to the scalar function. This preserves the `Columnar` enum:

```rust
pub enum Columnar {
    Canonical(Canonical),
    Constant(ConstantArray),
}
```

`ScalarFn::execute` receives `Vec<Columnar>`. Most scalar function implementations handle
constant inputs efficiently -- e.g., `add(primitive, constant(1))` broadcasts the scalar
without allocating a repeated buffer.

Reduce rules interact with constants via `as_opt::<ConstantVTable>()` to read the scalar value
(metadata access, not buffer access). Constant folding, constant propagation (`slice(constant)
→ constant`, `filter(constant, mask) → constant`), and related rules are all metadata-only.

### Per-encoding canonicalize_into

Each encoding implements a single decompression method that handles selection:

- **BitPacked**: unpack only selected positions into PrimitiveBuilder
- **FSST**: decompress only selected strings into VarBinViewBuilder
- **RunEnd**: binary-search ends to find runs intersecting the selection, expand only those
- **Dict**: materialize codes with selection, take from values
- **Chunked**: split selection across chunks, recurse per-chunk
- **Struct**: pass selection to each field's materialization
- **Constant**: create scalar repeated to selection count

For fused decompression patterns (e.g., Dict-RLE), the parent encoding checks inline:

```rust
// Dict's canonicalize_into
fn canonicalize_into(dict: &DictArray, builder, selection, ctx) {
    // Check for fused Dict-RLE pattern
    if let Some(runend) = dict.codes().as_opt::<RunEndVTable>() {
        return fused_dict_rle(dict.values(), runend, builder, selection, ctx);
    }

    // Generic path: materialize codes with selection, take from values
    let mut code_builder = primitive_builder(dict.codes().dtype());
    dict.codes().canonicalize_into(&mut code_builder, selection, ctx)?;
    let codes = code_builder.finish_into_primitive();
    take_into_builder(builder, dict.values(), &codes, ctx)
}
```

### Decompression cache and CSE

`ExecutionCtx` holds a pointer-identity cache (`HashMap<*const dyn Array, ArrayRef>`) scoped
to a single `canonicalize()` call. This serves two purposes:

- **Common sub-expression elimination**: when the same array node appears as a child of
  multiple ScalarFnArray nodes (e.g., `x + x`), it is materialized once.
- **Shared block decompression**: when a ZStd-compressed array is sliced into many chunks, all
  slices share the same underlying compressed block. The first slice decompresses the block and
  caches the result; subsequent slices index into it.

The cache key is the raw pointer to the array, which is stable because arrays are `Arc`-based.
The cache is dropped after `canonicalize()` returns, so it does not leak memory across calls.

### Worked example: DuckDB dictionary export

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

**Phase 1: optimize.** `reduce_parent` pushes `upper` into dict values:

```
filter(
    dict(
        codes: bitpacked([0,1,0,2,1,0,...]),
        values: scalar_fn(upper, [fsst(["alice","bob","charlie"])])
    ),
    mask
)
```

**Phase 2: peel selections.** Filter is at the root, so it peels:

```
array = dict(codes: bitpacked(...), values: scalar_fn(upper, [fsst(...)]))
selection = Selection::Mask(mask)
```

**Phase 3: export.** The DuckDB exporter inspects the tree:

```rust
fn export_chunk(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<DuckDBVector> {
    let array = array.optimize_recursive()?;
    let (array, selection) = array.peel_selection();

    if let Some(dict) = array.as_opt::<DictVTable>() {
        return export_dict(dict, &selection, ctx);
    }

    // Fallback: full materialization
    let mut builder = CanonicalBuilder::new(array.dtype());
    array.canonicalize_into(&mut builder, &selection, ctx)?;
    export_canonical(builder.finish())
}

fn export_dict(
    dict: &DictArray,
    selection: &Selection,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DuckDBVector> {
    // Materialize ALL unique values (no selection -- values are shared)
    let mut val_builder = CanonicalBuilder::new(dict.values().dtype());
    dict.values().canonicalize_into(&mut val_builder, &Selection::All, ctx)?;
    let values = val_builder.finish();

    // Materialize codes WITH selection (only rows that passed the filter)
    let mut code_builder = primitive_builder(dict.codes().dtype());
    dict.codes().canonicalize_into(&mut code_builder, selection, ctx)?;
    let codes = code_builder.finish_into_primitive();

    Ok(DuckDBVector::dictionary(values, codes))
}
```

**Values path**: `scalar_fn(upper, [fsst(["alice","bob","charlie"])])` materializes via
ScalarFnArray's `canonicalize_into`. FSST bulk-decompresses 3 strings, `upper` runs on canonical
input, producing `["ALICE","BOB","CHARLIE"]`.

**Codes path**: `bitpacked([0,1,0,2,1,0,...])` materializes with `Selection::Mask(mask)`.
BitPacked unpacks only the positions where the mask is true.

**Result**: DuckDB gets a dictionary vector with 3 uppercase values and only the selected
codes. `upper()` was computed on 3 unique values instead of N rows. The filter was fused with
bit-unpacking. No intermediate arrays were materialized.

### What changes

**Removed:**

- `execute` VTable method -- replaced by `canonicalize_into(builder, selection)`.
- `execute_parent` VTable method -- selection-related uses replaced by `Selection`;
  compressed-domain computation uses replaced by `reduce_parent` rules.
- The executor loop -- no more iterative execute-until-canonical. Optimization is a reduce-only
  loop; materialization is a single recursive descent.
- Selection-related adaptors -- `FilterReduceAdaptor`, `FilterExecuteAdaptor`,
  `SliceReduceAdaptor`, `SliceExecuteAdaptor`, `TakeExecuteAdaptor`, and all their per-encoding
  implementations (~40+ impls total).
- Canonical-type `execute_parent` kernels -- Primitive + Compare, Bool + FillNull, Decimal +
  Between, etc. These move into the scalar function's `execute` method where they belong.

**Retained:**

- `reduce` / `reduce_parent` -- unchanged. Still drive expression push-down, cast push-down,
  constant folding, structural simplification. Strictly metadata-only, no buffer access.
- `ScalarFnArray` -- kept as per-node lazy computation. Composes with `reduce_parent` for
  expression push-down into compressed encodings.
- Filter/Slice/Take array wrappers -- still exist for lazy representation. Their
  `canonicalize_into` composes them into the Selection and delegates to the child. These could be
  unified into a single `SelectionArray` wrapping a `Selection` in the future.

**New:**

- `Selection` type -- unifies Filter, Slice, and Take into a single row-selection
  representation that composes and threads through materialization.
- `CanonicalBuilder` -- one-level-deep builder enum. Struct children accumulate as
  `Vec<ArrayRef>` rather than being decompressed into child builders.
- `Array::peel_selection()` -- method to strip selection wrappers from the root, used by
  exporters that want to inspect the tree before materializing.

**Summary comparison:**

| Aspect                         | Current                                                                         | Proposed                                          |
|--------------------------------|---------------------------------------------------------------------------------|---------------------------------------------------|
| VTable methods                 | 5 (`reduce`, `reduce_parent`, `execute`, `execute_parent`, `append_to_builder`) | 3 (`reduce`, `reduce_parent`, `canonicalize_into`)     |
| Adaptor types per operation    | 2 (Reduce + Execute)                                                            | 1 (Reduce only, for expression push-down ops)     |
| Selection handling             | Per-encoding Filter, Slice, Take adaptors                                       | Single `Selection` parameter                      |
| Materialization                | Iterative loop calling execute until convergence                                | Single recursive descent through `canonicalize_into`   |
| Executor loop                  | reduce → reduce_parent → execute_parent → execute, repeated                     | optimize (reduce loop) → peel selections → build  |
| Builder model                  | Fully recursive (forces deep canonicalization)                                  | One-level canonical (children stay compressed)    |
| Fused decompression (Dict-RLE) | Child volunteers via execute_parent                                             | Parent checks for pattern in its own `canonicalize_into` |
| Buffer access in reduce        | Ambiguous                                                                       | Never (strict, works on any device including GPU) |

### Migration path

The migration is designed so that every step leaves the codebase in a working state. The old
and new execution paths coexist until the new path covers all encodings, at which point the
old path is deleted. Each phase is independently shippable.

**Phase 1: Foundation types.** Introduce `Selection` (with `compose`, `count`, `apply`,
`apply_in_place`), `CanonicalBuilder` enum, and `Array::peel_selection()`. All additive, no
behavior changes.

**Phase 2: Add `canonicalize_into` with fallback default.** Add `canonicalize_into` to `VTable` with a
default implementation that bridges to the old execute path:

```rust
fn canonicalize_into(
    array: &Self::Array,
    builder: &mut CanonicalBuilder,
    selection: &Selection,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    // Fallback: use existing execute-to-canonical, then apply selection
    let canonical = Canonical::execute(array.as_array_ref().clone(), ctx)?;
    let selected = selection.apply_to_array(canonical.into_array())?;
    builder.extend_from_array(&selected)?;
    Ok(())
}
```

Wire up `canonicalize()` using the three-phase pipeline. Both old (`execute::<Canonical>`) and
new (`canonicalize()`) paths work. Integration tests assert equivalence.

**Phase 3: Migrate canonical encodings.** One encoding per PR: Null, Bool, Primitive, Decimal,
VarBinView, Struct, List/FixedSizeList, Extension, Constant. Each writes directly into the
builder with fused selection. Independent and parallelizable.

**Phase 4: Migrate selection wrappers.** Filter, Slice, Chunked, Masked compose into
`Selection` instead of executing independently.

**Phase 5: Migrate compressed encodings.** Selection-fused decompression for BitPacked,
RunEnd, Dict, FSST, ALP, FoR, ZStd, Sparse, and others. Start with highest-value encodings.
Each encoding is an independent PR.

**Phase 6: Move `execute_parent` → `reduce_parent`.** Compressed-domain rewrites (Dict +
Compare, ALP + Compare, FoR + Compare, FSST + Compare) become `reduce_parent` rules.
Canonical-type kernels (Primitive + Compare, Bool + FillNull) move into `ScalarFn::execute`.
Once empty, remove `execute_parent` from VTable.

**Phase 7: Delete old execution path.** Delete all selection adaptor traits and their ~40+
per-encoding implementations. Delete `execute` VTable method, the executor loop, and the
`Executable` trait. All callers of `execute::<Canonical>` migrate to `canonicalize()` (the
free function that runs the full three-phase pipeline).

**Phase 8: Replace `scalar_at`.** `scalar_at` becomes a free function using
`Selection::Index`. Remove `OperationsVTable::scalar_at` and per-encoding implementations.

**Phase 9: Wire up exporters.** DuckDB exporter uses optimize → peel → inspect → export.
Add decompression cache to `ExecutionCtx`.

**Ordering and parallelism:**

```
Phase 1: Foundation types
    │
Phase 2: canonicalize_into with fallback
    │
    ├── Phase 3: Canonical encodings (independent per encoding)
    ├── Phase 4: Selection wrappers
    ├── Phase 5: Compressed encodings (independent per encoding)
    └── Phase 6: execute_parent → reduce_parent (independent per rule)
         │
    Phase 7: Delete old path (after 3-6 complete)
         │
    ├── Phase 8: scalar_at → Selection::Index
    └── Phase 9: Exporter updates
```

Phases 3-6 can proceed in parallel across contributors. The fallback default in Phase 2 ensures
nothing breaks while encodings are migrated at their own pace. Phase 7 is the cleanup gate.

## Compatibility

This RFC does not change the file format or wire format. All changes are internal to the
execution engine.

**Public API breakage:**

- `Executable` trait and `execute::<T>()` are removed. Callers migrate to `canonicalize()` or
  call `canonicalize_into` directly.
- `OperationsVTable::scalar_at` is removed. Callers use the free function `scalar_at()` which
  delegates to `canonicalize_into` with `Selection::Index`.
- Third-party encodings that implement `execute`, `execute_parent`, or selection adaptors must
  migrate to `canonicalize_into`. The fallback default (Phase 2) provides a bridge during
  migration.

**Performance implications:**

- Selection-fused decompression should improve performance for filtered/sliced reads by
  avoiding intermediate materialization.
- Single-element access via `Selection::Index` may be marginally slower than today's
  `scalar_at` for encodings that currently fast-path scalar access. These encodings can
  fast-path `Selection::Index` in their `canonicalize_into` if needed.
- The decompression cache avoids redundant work for shared sub-expressions and ZStd blocks.

## Drawbacks

- **Migration effort.** The migration touches every encoding in the codebase (~33 total). The
  phased approach with a fallback default mitigates this, but it is still significant work
  spread across many PRs.

- **Fused pattern detection moves to the parent.** In the current model, a child encoding can
  volunteer to handle its parent's execution via `execute_parent` (e.g., RunEnd volunteers for
  Dict-RLE). In the new model, the parent must check for the pattern in its own
  `canonicalize_into`. This means the parent must know about the child encoding, which is a
  slight inversion of responsibility. In practice, the known fused patterns (Dict-RLE) involve
  encodings that are both Arrow-supported and live in the builtin `vortex-array` crate, so the
  cross-encoding dependency is not a crate boundary issue.

- **Selection composition complexity.** `Selection::compose` must handle all variant
  combinations correctly (Mask + Range, Indices + Mask, etc.). This is concentrated complexity
  in one well-tested type, but getting it wrong would cause subtle data corruption.

- **`Indices` conflates selection with gather.** The first four `Selection` variants are pure
  row subsetting (output <= input, preserves order, no nulls introduced). `Indices` is a
  gather: it can repeat rows, reorder them, and introduce nulls. Bundling both concepts into
  one enum is a pragmatic compromise -- the alternative (a separate gather API every encoding
  must implement) recreates the combinatorial explosion this RFC aims to eliminate -- but it
  means `Selection` is not a clean abstraction.

## Alternatives

### Keep the iterative execution model

We could keep the current `execute`/`execute_parent` loop and address individual pain points
(e.g., adding selection fusion as a new adaptor). This avoids the migration cost but perpetuates
the unclear distinction between reduce and execute, the combinatorial adaptor explosion, and the
difficulty of reasoning about the executor loop. The VTable surface stays at 5 methods.

### Use a single ExpressionArray instead of per-node ScalarFnArray

Instead of one ScalarFnArray per operation, a single ExpressionArray could hold an entire
expression tree. This would be simpler in some ways but prevents `reduce_parent` from
interacting with individual expression nodes. The per-node approach is what makes Dict's
`reduce_parent` able to push `compare` into dict values -- the key optimization for
dictionary-encoded predicates.

### Use `dyn ArrayBuilder` instead of `CanonicalBuilder` enum

The builder could be a trait object (`&mut dyn ArrayBuilder`) instead of an enum. This is more
extensible but prevents encodings from matching on the concrete builder type. FSST's ability to
push directly into `VarBinViewBuilder` (avoiding an intermediate string allocation) depends on
knowing the builder variant at compile time.

### Separate `scalar_at` from `Selection`

We could keep `scalar_at` as a separate VTable method for single-element access, rather than
subsuming it into `Selection::Index`. This avoids any performance regression for scalar access
but adds a fourth VTable method and requires every encoding to implement it separately.

## Prior Art

- **DataFusion** has `PhysicalExpr::evaluate_selection`, which takes an explicit selection
  (equivalent to our `Selection`) and threads it through expression evaluation. This is the
  closest analogue to threading `Selection` through `canonicalize_into`.

## Unresolved Questions

- **Selection composition correctness**: Composing Mask + Range + Indices needs careful
  implementation and thorough testing, especially for `Indices` which introduces repetition,
  reordering, and nullability. This is a well-tested utility type vs N x 3 adaptor
  implementations, so the tradeoff is favorable, but it needs to be right.

- **CanonicalBuilder design**: The exact interface for one-level-deep building, especially for
  List and Extension types, needs design work.

- **Aggregate and window functions**: This RFC focuses on scalar functions. AggregateFnArray
  and WindowFnArray will need their own materialization paths, which are explicitly out of scope
  for this RFC.

- **`Indices` semantics mismatch**: `Indices` is a gather (repetition, reordering, nullable)
  while the other `Selection` variants are pure subsetting. This means `compose`, `count`, and
  `apply` must handle two fundamentally different operations. Whether this tension causes
  practical problems (e.g., confusing behavior in compose chains) needs to be evaluated during
  implementation.

## Future Possibilities

### Selection pull-up

In the new model, selection wrappers compose naturally into the `Selection` parameter during
the recursive descent. However, when a selection is nested inside an expression, it cannot be
peeled:

```
scalar_fn(upper, [filter(dict(...), mask)])
```

`peel_selection()` sees ScalarFnArray at the root, not FilterArray. If `reduce_parent` has no
rule for this function + encoding combination, the tree stays as-is and the filter gets
composed in during descent. `upper` runs on N selected rows, producing a flat canonical result.
The dictionary structure is lost.

A future optimization could **pull selections out of expressions** to the root:

```
filter(scalar_fn(upper, [dict(...)]), mask)
```

With the selection at the root, `peel_selection()` extracts it, and subsequent optimizations
can proceed. More optimal decisions can be made when the selection is available at the top and
pushed down at execution time, rather than being trapped inside an expression.

In practice, most scans apply filters at the top level of the array tree, so this edge case is
uncommon. But for complex expression trees with embedded selections, pull-up would enable
strictly better execution plans.

### Aggregate and window functions

The execution model is designed to support additional function types beyond scalar functions.
`AggregateFnArray` (for sum, min, max, count) and `WindowFnArray` will follow a similar
deferred pattern, using the same `reduce`/`reduce_parent` optimization strategy and
`Selection`-based materialization.

### GPU kernel fusion

The strict metadata-only constraint on `reduce`/`reduce_parent` means the optimized tree can
be shipped to a GPU context. A GPU-aware `canonicalize_into` could fuse the entire optimized tree
into a single kernel launch, reducing memory traffic and kernel overhead.
