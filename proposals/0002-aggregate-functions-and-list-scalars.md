- Start Date: 2026-02-26
- RFC PR: [vortex-data/rfcs#0000](https://github.com/vortex-data/rfcs/pull/0000)
- Tracking Issue: [vortex-data/vortex#0000](https://github.com/vortex-data/vortex/issues/0000)

## Summary

Introduce `AggregateFnVTable` to Vortex as a trait for defining aggregate operations, and a
single `ListAggregate` scalar function that applies any aggregate to a list column. There is
no dedicated `AggregateFnArray` — `ListAggregate` is a `ScalarFnVTable` implementation whose
lazy evaluation is handled by the existing `ScalarFnArray` infrastructure. Child encodings
match on `ExactScalarFn<ListAggregate>` in their reduce_parent rules for aggregate-specific
optimizations.

The key insight is the **aggregate/list-scalar duality**: a list column stored as
`(offsets, elements)` is a pre-materialized grouping. Computing `list_sum(list_column)` is
literally a grouped `sum` over the flat elements array partitioned by offsets. Rather than
implementing N separate list scalar functions (`list_sum`, `list_min`, `list_max`, ...),
we implement N aggregate functions and a single `ListAggregate` scalar function that applies
any aggregate to a list column.

The `Accumulator` trait is the core primitive. It processes one group at a time via
`accumulate(batch)` / `flush()` / `finish()`, supporting streaming through chunked elements,
very large lists, and ungrouped column-level aggregation. `execute_grouped` is a convenience
built on top of the accumulator.

Streaming ordered GROUP BY falls out naturally: the query engine constructs a `ListArray` from
`(flat_column, group_offsets)` per batch — a zero-copy operation — and wraps it in
`ScalarFnArray(ListAggregate(Sum), [list_array])`.

This RFC also evaluates whether the canonical list encoding should switch from `ListViewArray`
(offsets+sizes) to `ListArray` (offsets-only), since the choice directly affects how cleanly
the duality works.

## Motivation

### Aggregate functions exist but in the old framework

Vortex has two compute systems:

- **`ScalarFnVTable`**: element-wise functions (1 input row → 1 output row) with lazy
  `ScalarFnArray` evaluation, expression trees, and reduce/reduce_parent optimization.
- **`ComputeFn`**: kernel-based operations like `sum`, `min_max`, `nan_count`. Standalone,
  not part of the expression tree, dispatched to per-encoding kernels via `inventory`.

Aggregate operations like `sum` already exist, but as `ComputeFn` kernels. They cannot
participate in the expression tree or optimizer. This means:

- No lazy evaluation that defers computation and can be optimized via reduce_parent.
- No way to push aggregate operations through Dict values, RunEnd runs, or other encodings
  via the reduce rule framework.
- No unified framework for list scalar functions and grouped aggregates.

### List scalar functions are redundant with aggregate functions

The only list-specific scalar function today is `list_contains`. Adding `list_sum`,
`list_min`, `list_max`, `list_count`, `list_mean`, etc. as individual `ScalarFnVTable`
implementations would duplicate the aggregation logic that already exists in `ComputeFn`
kernels. If Vortex had first-class aggregate functions, each list scalar function would
simply be "apply this aggregate to each list" — one `ListAggregate` scalar function
parameterized by the aggregate, rather than N separate implementations.

### The aggregate/list-scalar duality

Consider a list column `[[1,2,3], [4,5]]` stored as:

```
elements: [1, 2, 3, 4, 5]
offsets:  [0, 3, 5]
```

Computing `list_sum` yields `[6, 9]`. This is identical to computing a grouped `sum` over
elements partitioned by offset ranges `[0..3, 3..5]`. Every aggregate function has a
corresponding list scalar function:

| Aggregate    | List scalar            | Grouped operation                 |
|--------------|------------------------|-----------------------------------|
| `sum(col)`   | `list_sum(list_col)`   | Sum elements per group            |
| `count(col)` | `list_count(list_col)` | Count non-null elements per group |
| `min(col)`   | `list_min(list_col)`   | Min element per group             |
| `max(col)`   | `list_max(list_col)`   | Max element per group             |
| `mean(col)`  | `list_mean(list_col)`  | Mean of elements per group        |

### Streaming ordered GROUP BY

For databases with sorted storage, GROUP BY over a sorted key is a streaming operation:
groups arrive in order and don't cross batch boundaries. This maps directly to the
aggregate/list-scalar duality:

1. For each batch, compute group boundaries from the sorted key (run boundaries → offsets).
2. Construct `ListArray::new(value_column, group_offsets, ...)` — zero-copy.
3. Wrap in `ScalarFnArray(ListAggregate(Sum), [list_array])`.
4. Execute → one result per group.

No hash tables, no shuffling. The same `execute_grouped` path serves both list scalar
functions and streaming ordered GROUP BY.

### Expensive list interop

The current canonical list encoding is `ListViewArray` (offsets+sizes, out-of-order allowed).
Converting between `ListArray` and `ListViewArray` is expensive — see the rebuild
infrastructure with modes `MakeZeroCopyToList`, `TrimElements`, `MakeExact`, and the
`is_zero_copy_to_list` flag. This complexity exists primarily to support out-of-order
offsets, which complicates the aggregate duality since `execute_grouped` needs monotonic
offsets.

## Design

### `Accumulator` — the core primitive

The `Accumulator` trait is the fundamental aggregation interface. It processes one group
at a time: the caller feeds element batches, then flushes to finalize the group and start
the next one. The accumulator owns its output buffer and returns all results at the end.

```rust
pub trait Accumulator: Send + Sync {
    /// Feed a batch of element values for the current group.
    /// Can be called multiple times per group (e.g., chunked elements).
    fn accumulate(&mut self, batch: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()>;

    /// Finalize the current group: push its result into the output buffer
    /// and reset internal state for the next group.
    fn flush(&mut self, ctx: &mut ExecutionCtx) -> VortexResult<()>;

    /// Return all flushed results as a single array.
    /// Length = number of flush() calls.
    fn finish(self: Box<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>;
}
```

This handles all aggregation cases uniformly:

**Case 1 — List scalar with contiguous elements:**
```rust
let mut acc = aggregate.accumulator(element_dtype)?;
for i in 0..n_lists {
    let group_elements = elements.slice(offsets[i]..offsets[i+1])?;
    acc.accumulate(&group_elements, ctx)?;
    acc.flush(ctx)?;
}
acc.finish(ctx)  // → ArrayRef of length n_lists
```

**Case 2 — Large lists with chunked elements:**
A list's elements may be a `ChunkedArray`. The accumulator handles this by feeding
each chunk separately before flushing:
```rust
let mut acc = aggregate.accumulator(element_dtype)?;
for i in 0..n_lists {
    let group_elements = elements.slice(offsets[i]..offsets[i+1])?;
    for chunk in iter_chunks(&group_elements) {
        acc.accumulate(&chunk, ctx)?;
    }
    acc.flush(ctx)?;
}
acc.finish(ctx)
```

**Case 3 — Ungrouped full-column aggregation:**
One group, fold across chunks:
```rust
let mut acc = aggregate.accumulator(dtype)?;
for chunk in chunked_array.chunks() {
    acc.accumulate(&chunk, ctx)?;
}
acc.flush(ctx)?;
acc.finish(ctx)  // → 1-element array
```

**Case 4 — ListView with scattered elements:**
Canonicalize to ListArray first (rebuilding to sorted form), then use Case 1 or 2.
The accumulator itself always sees contiguous element batches.

### `AggregateFnVTable`

A new trait parallel to `ScalarFnVTable`. The `accumulator()` method is the required
core — `execute_grouped` and `execute_scalar` have default implementations built on it:

```rust
pub trait AggregateFnVTable: 'static + Sized + Clone + Send + Sync {
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    fn id(&self) -> AggregateFnId;

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>>;
    fn deserialize(
        &self,
        metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<Self::Options>;

    /// Arity is always 1. For multi-column aggregates (e.g., covariance), the input
    /// should be a struct array containing the columns.
    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    /// Result dtype for each group (not a list — the scalar result type).
    fn return_dtype(
        &self,
        options: &Self::Options,
        input_dtypes: &[DType],
    ) -> VortexResult<DType>;

    /// Format in SQL style, e.g. `sum(col)`.
    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result;

    /// Create an accumulator for streaming aggregation.
    ///
    /// This is the core primitive. All other execution methods have default
    /// implementations built on this.
    fn accumulator(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Box<dyn Accumulator>>;

    /// One-shot grouped execution over contiguous elements + monotonic offsets.
    ///
    /// Default: loop over groups, slice elements, accumulate each, flush, finish.
    /// Override for vectorized one-shot paths (e.g., SIMD segmented sum).
    fn execute_grouped(
        &self,
        options: &Self::Options,
        input: &ArrayRef,
        offsets: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let n_groups = offsets.len() - 1;
        let mut acc = self.accumulator(options, input.dtype())?;
        for i in 0..n_groups {
            let start = scalar_at(offsets, i)?;
            let end = scalar_at(offsets, i + 1)?;
            acc.accumulate(&input.slice(start..end)?, ctx)?;
            acc.flush(ctx)?;
        }
        acc.finish(ctx)
    }

    /// Ungrouped full-column aggregation returning a scalar.
    ///
    /// Default: single-group execute_grouped with offsets [0, n].
    /// Replaces standalone `ComputeFn` kernels like `compute::sum()`.
    fn execute_scalar(
        &self,
        options: &Self::Options,
        input: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let n = input.len();
        let offsets: ArrayRef = buffer![0u64, n as u64].into_array();
        let result = self.execute_grouped(options, input, &offsets, ctx)?;
        result.scalar_at(0)
    }
}
```

Key differences from `ScalarFnVTable`:

| Property | `ScalarFnVTable` | `AggregateFnVTable` |
|----------|-------------------|----------------------|
| Row semantics | 1 input → 1 output | N inputs → 1 output per group |
| Core primitive | `execute(args) -> ArrayRef` | `accumulator() -> Box<dyn Accumulator>` |
| Input shape | All children same length | Flat elements + offsets defining variable-size groups |
| Streaming | N/A | accumulate/flush/finish across chunks |

### Streaming AVG example

To illustrate how the accumulator works for a stateful aggregate, consider `Mean`:

```rust
struct MeanAccumulator {
    running_sum: f64,
    running_count: u64,
    output: Vec<f64>,
}

impl Accumulator for MeanAccumulator {
    fn accumulate(&mut self, batch: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        self.running_sum += Sum.execute_scalar(&EmptyOptions, batch, ctx)?.as_f64()?;
        self.running_count += Count.execute_scalar(&EmptyOptions, batch, ctx)?.as_u64()?;
        Ok(())
    }

    fn flush(&mut self, _ctx: &mut ExecutionCtx) -> VortexResult<()> {
        let mean = if self.running_count > 0 {
            self.running_sum / self.running_count as f64
        } else {
            f64::NAN
        };
        self.output.push(mean);
        self.running_sum = 0.0;
        self.running_count = 0;
        Ok(())
    }

    fn finish(self: Box<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::from_vec(self.output).into_array())
    }
}
```

For a list column `[[1.0, 2.0, 3.0], [4.0, 5.0]]` where the first list's elements are
chunked as `[1.0, 2.0]` and `[3.0]`:

1. `accumulate([1.0, 2.0])` → running_sum=3.0, running_count=2
2. `accumulate([3.0])` → running_sum=6.0, running_count=3
3. `flush()` → push 2.0, reset
4. `accumulate([4.0, 5.0])` → running_sum=9.0, running_count=2
5. `flush()` → push 4.5, reset
6. `finish()` → `[2.0, 4.5]`

### Built-in aggregates

The initial set, each implementing `AggregateFnVTable`:

```rust
pub struct Sum;           // sum of elements per group
pub struct Count;         // count of non-null elements per group
pub struct Min;           // minimum element per group
pub struct Max;           // maximum element per group
pub struct Mean;          // mean of elements per group (returns f64)
pub struct Any;           // logical OR per group (bool input)
pub struct All;           // logical AND per group (bool input)
pub struct CollectList;   // identity: collect elements into list (returns list)
```

**Migration strategy**: `execute_scalar` replaces standalone `ComputeFn` kernel dispatch
(e.g., `Sum::execute_scalar` replaces `compute::sum()`). The `accumulator` implementations
can start simple (canonicalize + iterate) and be optimized over time with encoding-aware
fast paths.

### `ListAggregate` scalar function

A single `ScalarFnVTable` implementation that bridges list columns to the aggregate system.
Because `ListAggregate` is a scalar function, wrapping it in an expression produces a
`ScalarFnArray` — reusing the existing lazy evaluation, slicing, and reduce infrastructure
with no new array type needed.

```rust
pub struct ListAggregate;

pub struct ListAggregateOptions {
    pub aggregate_fn: AggregateFnRef,
}

impl ScalarFnVTable for ListAggregate {
    type Options = ListAggregateOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.list.aggregate")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn return_dtype(
        &self,
        options: &Self::Options,
        arg_dtypes: &[DType],
    ) -> VortexResult<DType> {
        let element_dtype = arg_dtypes[0].as_list_element()
            .ok_or_else(|| vortex_err!("ListAggregate input must be a list"))?;
        options.aggregate_fn.return_dtype(&[element_dtype.clone()])
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: ExecutionArgs,
    ) -> VortexResult<ArrayRef> {
        let list_input = &args.inputs[0];
        let ctx = args.ctx;
        // Execute the child into a ListArray, decompose, and aggregate.
        let list = list_input.to_list()?;
        let elements = list.elements();
        let offsets = list.offsets();

        // If elements are chunked, use the accumulator to stream through chunks.
        // Otherwise, use one-shot execute_grouped.
        if elements.is::<ChunkedArray>() {
            let mut acc = options.aggregate_fn.accumulator(elements.dtype())?;
            let n_groups = offsets.len() - 1;
            for i in 0..n_groups {
                let group_elements = elements.slice(
                    offset_at(&offsets, i)..offset_at(&offsets, i+1)
                )?;
                for chunk in iter_chunks(&group_elements) {
                    acc.accumulate(&chunk, ctx)?;
                }
                acc.flush(ctx)?;
            }
            acc.finish(ctx)
        } else {
            options.aggregate_fn.execute_grouped(&elements, &offsets, ctx)
        }
    }

    // ...
}
```

Expression-level sugar:

```rust
pub fn list_sum(list: Expression) -> Expression {
    ListAggregate.new_expr(
        ListAggregateOptions { aggregate_fn: Sum.bind(EmptyOptions) },
        [list],
    )
}

pub fn list_count(list: Expression) -> Expression {
    ListAggregate.new_expr(
        ListAggregateOptions { aggregate_fn: Count.bind(EmptyOptions) },
        [list],
    )
}

// list_min, list_max, list_mean, list_any, list_all analogously
```

This is one scalar function parameterized by the aggregate — not N separate functions.
Analogous to DuckDB's `list_aggregate(list, 'sum')`.

**Why this works without a dedicated array type:**

1. **Lazy evaluation.** `ScalarFnArray(ListAggregate(Sum), [list_col])` defers computation
   until `execute()` is called, just as it does for any other scalar function.
2. **Slicing.** `ScalarFnArray` already slices its children. Slicing the list child
   correctly adjusts both grouping and elements.
3. **Reduce rules.** `ListAggregate` can implement `ScalarFnVTable::reduce` for self-reduce
   optimizations (constant folding, statistics). Encoding-specific reduce_parent rules
   use `ExactScalarFn<ListAggregate>` to match on the parent (see below).
4. **Type safety.** The single child has `DType::List` — enforced by `return_dtype`.

### Streaming ordered GROUP BY

For databases with sorted storage, GROUP BY over a sorted key maps directly to the
aggregate/list-scalar duality. The query engine processes batches where:

1. Groups are defined by a sorted key column (e.g., `customer_id`).
2. Groups are ordered and complete within each batch (no cross-batch groups).
3. No hash tables or shuffling needed.

**Physical execution per batch:**

```rust
fn execute_ordered_group_by(
    value_column: &ArrayRef,
    key_column: &ArrayRef,     // sorted
    aggregate: &AggregateFnRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Step 1: Compute group offsets from run boundaries in the key column.
    // E.g., key = [A, A, A, B, B, C] → offsets = [0, 3, 5, 6]
    let group_offsets = compute_run_offsets(key_column)?;

    // Step 2: Wrap as a list — zero-copy, just references.
    let list = ListArray::new(value_column.clone(), group_offsets, Validity::NonNullable);

    // Step 3: Create lazy ScalarFnArray with ListAggregate.
    let scalar_fn = ListAggregate.bind(ListAggregateOptions {
        aggregate_fn: aggregate.clone(),
    });
    let result = ScalarFnArray::try_new(scalar_fn, vec![list.into_array()])?;

    // Step 4: Execute (or let the optimizer reduce first).
    result.into_array().execute(ctx)
}
```

The same `execute_grouped` serves both list scalar functions (where the list already exists
in the schema) and GROUP BY (where the list is constructed from a flat column +
sort-derived offsets).

### Reduce rules

Since `ListAggregate` is a `ScalarFnVTable`, its reduce rules use the existing
`ScalarFnArray` optimization infrastructure.

**Self-reduce rules (via `ScalarFnVTable::reduce`):**

`ListAggregate` implements `ScalarFnVTable::reduce` to handle aggregate-specific
optimizations when the list child's structure is known:

- **Constant list folding**: If the list child is a `ConstantArray` of list scalars,
  compute the aggregate directly. `ListAggregate(Sum, Constant([[1,2,3]], 1000))`
  → `Constant(6, 1000)`.

- **Count from metadata**: `ListAggregate(Count, list)` can use list sizes from metadata
  without touching element data.

- **Min/Max from statistics**: If element-level statistics are available, `Min`/`Max` can
  short-circuit without decompression.

- **Sum of constant elements**: `ListAggregate(Sum, list)` where elements are constant
  → `constant_value * group_size` per group.

**Parent-reduce rules (encoding-specific, via `ExactScalarFn<ListAggregate>`):**

Child encodings register reduce_parent rules that match on `ExactScalarFn<ListAggregate>`.
The `ExactScalarFn` matcher provides typed access to `ListAggregateOptions`, from which the
specific `AggregateFnRef` can be inspected:

```rust
impl ArrayParentReduceRule<DictVTable> for DictAggregateRule {
    type Parent = ExactScalarFn<ListAggregate>;

    fn reduce_parent(
        &self,
        array: &DictArray,
        parent: ScalarFnArrayView<'_, ListAggregate>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let aggregate_fn = &parent.options.aggregate_fn;

        // Min/Max over Dict elements → Min/Max(values)
        if aggregate_fn.is::<Min>() || aggregate_fn.is::<Max>() {
            // Min/max of dictionary values is the global min/max.
            let values = array.values();
            return Ok(Some(
                ListAggregate.new_expr_with(
                    parent.options.clone(),
                    // Rebuild list with dict values as elements
                    // ...
                ).into_array()
            ));
        }

        Ok(None)
    }
}
```

Example encoding-specific rules:

- **Dict**: `ListAggregate(Min, List(Dict(codes, values)))` → `ListAggregate(Min, List(values))`.
  `Max` similarly. `Sum` cannot push down without frequency weighting.

- **RunEnd**: `ListAggregate(Sum, List(RunEnd(values, run_ends)))` → weighted sum.
  `ListAggregate(Min/Max, List(RunEnd))` → `ListAggregate(Min/Max, List(values))`.

These rules operate on the **elements** within the list child. The list's offsets define
the grouping; the elements are what the encoding wraps. A list child like
`ListArray(elements: DictArray, offsets: ...)` allows the Dict reduce_parent rule to fire
on the elements when the `ScalarFnArray(ListAggregate)` is being optimized.

### Interaction with `list_contains`

The existing `list_contains` function is NOT an aggregate — it's a true scalar function
(takes a list + needle, returns bool per row). It stays as-is. `ListAggregate` is
specifically for operations that reduce list elements to a scalar per list.

## Canonical list representation

The choice of canonical list encoding directly affects how cleanly the aggregate duality
works. Currently, `ListViewArray` is canonical for `DType::List`. This section evaluates
whether to switch to `ListArray`.

### Option A: Switch canonical to `ListArray` (offsets-only)

Change `Canonical::List` from `ListViewArray` to `ListArray`.

`ListArray` stores elements + an (n+1)-length monotonically increasing offsets array.
List `i` spans `elements[offsets[i]..offsets[i+1]]`.

**Advantages for the aggregate duality:**

- `execute_grouped` receives offsets directly from the list — no conversion needed.
- Offsets are monotonic by construction, matching the `execute_grouped` contract.
- One array (n+1 integers) vs two arrays (2n integers) — less memory.
- Offset deltas (list sizes) compress extremely well with delta/FoR encoding.
- Direct match to Arrow `ListArray` and DuckDB's list format — zero-copy Arrow export.

**Disadvantages:**

- **Breaking change.** All code calling `to_canonical()` on list data gets `ListArray`
  instead of `ListViewArray`. Migration required.
- **No out-of-order offsets.** Operations like filter/take that scramble row order must
  rebuild the elements array (or produce a `ListViewArray` as a non-canonical intermediate).
- **Sequential dependency in offsets.** Reading `offsets[i+1]` depends on `offsets[i]` for
  computing list size — slightly less SIMD-friendly than separate sizes.

**Migration path:**

1. Add `to_list()` on `ToCanonical` alongside `to_listview()`.
2. Change `Canonical::List` to hold `ListArray`.
3. `ListViewArray` becomes a non-canonical encoding (like `VarBinArray` for strings).
4. Update `list_contains` and all list compute paths.

### Option B: Keep `ListViewArray` as canonical

Keep the current design where `ListViewArray` (offsets + sizes, out-of-order allowed)
is canonical for `DType::List`.

**Advantages:**

- No breaking change.
- SIMD-friendly: sizes can be read independently without sequential offset dependency.
- Out-of-order offsets enable better compression when list order doesn't match element order.
- Filter/take produce valid `ListViewArray` directly without rebuilding.

**Disadvantages for the aggregate duality:**

- `execute_grouped` requires monotonic offsets. Out-of-order `ListViewArray` must be
  rebuilt to sorted form first (the `MakeZeroCopyToList` path), which copies data.
- The ZCTL flag, rebuild modes, and validation code remain as ongoing complexity.
- Arrow/DuckDB export still requires conversion.
- Two arrays (2n integers) instead of one (n+1 integers).

**Impact on `ListAggregate::execute`:**

`ListAggregate::execute` would need to handle the non-sorted case:

```rust
fn execute(&self, options: &Self::Options, args: ExecutionArgs) -> VortexResult<ArrayRef> {
    let list_input = &args.inputs[0];
    let ctx = args.ctx;
    let listview = list_input.to_listview()?;
    if listview.is_zero_copy_to_list() {
        // Fast path: offsets are sorted, use directly
        let offsets = build_list_offsets_from_list_view(&listview);
        options.aggregate_fn.execute_grouped(&listview.elements(), &offsets, ctx)
    } else {
        // Slow path: rebuild to sorted form first
        let list = list_from_list_view(listview)?;
        options.aggregate_fn.execute_grouped(&list.elements(), &list.offsets(), ctx)
    }
}
```

### ListView as direct aggregate input

An alternative to canonicalizing to ListArray before executing is to support `ListView`
directly in `execute_grouped`. A ListView with out-of-order offsets represents
variable-length groups that may reference scattered regions of the elements array.

**Advantages:**

- More powerful — supports aggregation over non-contiguous element ranges.
- Avoids the expensive rebuild-to-sorted-form step for scrambled ListViews.

**Disadvantages:**

- **Complexity in aggregate implementations.** Each aggregate must handle potentially
  scattered, non-contiguous element access patterns instead of simple contiguous slices.
- **Unpredictable decompression cost.** A ListView with sparse, scattered offsets may
  trigger excessive random-access decompression of the elements array. In the worst case
  this is more expensive than one-pass decompression of all elements into contiguous form.
- **SIMD unfriendly.** Contiguous offset ranges map to sequential memory access patterns
  that vectorize well. Scattered access does not.

For now, this RFC assumes `execute_grouped` takes monotonic (n+1) offsets. Aggregates
over ListView canonicalize to sorted form first. Supporting scattered access as an
optimization can be explored in the future.

### Comparison

| Criterion | ListArray (Option A) | ListViewArray (Option B) |
|-----------|---------------------|--------------------------|
| Aggregate duality | Direct — offsets define groups | Indirect — must ensure sorted or rebuild |
| Arrow/DuckDB export | Zero-copy | Requires conversion |
| Memory | n+1 integers | 2n integers |
| Filter/take | Must rebuild elements | Produces valid array directly |
| SIMD for size access | Sequential offset dependency | Independent size reads |
| Offset compression | Delta/FoR on monotonic data | May be non-monotonic |
| Breaking change | Yes | No |

## Compatibility

This RFC does not change the file format or wire format. `ListAggregate` produces a
`ScalarFnArray` at runtime (like any other scalar function). It is not persisted to disk.

**Public API additions:**

- `Accumulator` trait — core streaming aggregation primitive.
- `AggregateFnVTable` trait and built-in implementations (Sum, Count, Min, Max, Mean,
  Any, All, CollectList).
- `ListAggregate` scalar function.
- Expression constructors: `list_sum()`, `list_count()`, `list_min()`, `list_max()`,
  `list_mean()`, `list_any()`, `list_all()`.

**If canonical list changes (Option A):**

- `Canonical::List` changes from `ListViewArray` to `ListArray`.
- `to_canonical()` on list data returns `ListArray` instead of `ListViewArray`.
- `ListViewArray` remains available as a non-canonical encoding.
- Migration: callers using `to_listview()` should transition to `to_list()`.

## Drawbacks

- **New trait surface area.** `AggregateFnVTable` and `Accumulator` are new traits to
  learn, though they closely mirror `ScalarFnVTable`.

- **Reduce rule coverage.** Not all encoding × aggregate combinations will have optimized
  reduce_parent rules initially. The fallback (canonicalize list, then accumulator loop)
  is correct but may be slower.

- **Canonical list change (if Option A).** Breaking change affecting all code that calls
  `to_canonical()` on list data.

## Alternatives

### List scalar functions as separate `ScalarFnVTable` implementations

Implement `ListSum`, `ListCount`, `ListMin`, etc. as individual scalar functions
(like `list_contains`), each operating directly on list arrays without an aggregate
abstraction.

**Rejected because:** Duplicates logic across N functions. No shared optimization rules.
No path to GROUP BY. Each function must independently handle list decomposition. This is
exactly the redundancy that motivates this RFC.

### Aggregates as `ComputeFn` only

Keep aggregates in the kernel-based `ComputeFn` system. Implement list scalar functions
by dispatching to `ComputeFn` with offset ranges.

**Rejected because:** No lazy evaluation, no expression tree participation, no
reduce_parent optimization. This is the status quo for `sum`/`min_max` and has the
limitations this RFC addresses.

### Dedicated `AggregateFnArray` array type

Introduce a new array type `AggregateFnArray` parallel to `ScalarFnArray`, wrapping
a single list child and an `AggregateFnRef`:

```rust
struct AggregateFnArray {
    aggregate_fn: AggregateFnRef,
    dtype: DType,
    child: ArrayRef,     // must have DType::List
    stats: ArrayStats,
}
```

With a `ListAggregate` scalar function that returns an `AggregateFnArray` from its
`execute` method, creating two lazy wrappers in sequence:
`ScalarFnArray(ListAggregate) → AggregateFnArray → actual computation`.

**Rejected because:** `AggregateFnArray` is structurally identical to
`ScalarFnArray` with one child — it duplicates the lazy evaluation, slicing, and reduce
infrastructure. Child encodings can already match on specific scalar functions using
`ExactScalarFn<ListAggregate>`, which provides typed access to `ListAggregateOptions`
and the embedded `AggregateFnRef`. No new array type is needed.

### No accumulator — one-shot `execute_grouped` only

Make `execute_grouped` the only execution method, with no streaming accumulation.

**Rejected because:** Doesn't handle chunked elements. A list column may have elements
stored as a `ChunkedArray`. Without an accumulator, all elements must be materialized
into a single contiguous array before aggregation, defeating the purpose of chunked
storage. Additionally, aggregates like `Mean` need `running_sum` and `running_count`
state across element batches, and numerically stable summation (Kahan) requires ordered
accumulation. The existing `sum_with_accumulator` pattern in `ComputeFn` demonstrates
this need.

### Grouped accumulator tracking N groups simultaneously

An accumulator that manages all groups at once: `accumulate(input, offsets)` where offsets
has N+1 entries per call, and sub-offsets are split at chunk boundaries.

**Rejected because:** Over-engineered for our use cases. For list scalar functions and
ordered GROUP BY, groups are processed sequentially — each group's elements are fully
available before the next group starts. The per-group flush model is simpler and handles
all cases. If vectorized all-groups-at-once processing proves necessary, it can be added
as a separate `GroupsAccumulator` trait (see Future Possibilities).

## Prior Art

- **Apache Arrow**: Separates `ListArray` (offsets-only) from `ListView` (offsets+sizes).
  Arrow Compute has aggregate kernels separate from scalar kernels. ListView is a recent
  addition, not widely adopted.

- **DuckDB**: Uses offsets-only lists. Has `list_aggregate(list, 'func_name')` — a single
  function parameterized by the aggregate name. Separates scalar and aggregate function
  registries.

- **Apache DataFusion**: Two-tier accumulator design. `AggregateUDFImpl` has a required
  `accumulator()` factory method (like ours). The `Accumulator` trait handles one group at
  a time with `update_batch()` → `evaluate()`, plus `state()` and `merge_batch()` for
  partial aggregation (serializing intermediate state for distributed execution). The
  optional `GroupsAccumulator` trait manages all groups simultaneously with
  `group_indices: &[usize]` per row for vectorized hash-based grouping, plus
  `evaluate(EmitTo::First(n))` for streaming emission. Our `accumulate/flush/finish`
  corresponds to DataFusion's per-group `Accumulator` with built-in output buffering.
  Their `state/merge` and `GroupsAccumulator` are future extensions for us (see below).

- **Velox**: `AggregateFunction` with `addRawInput`, `addIntermediateResults`,
  `extractValues` — full streaming accumulator model. Uses row-major (offset, size)
  pairs for lists.

- **Polars**: Separate scalar and aggregate expression systems. List namespace functions
  like `list.sum()`, `list.min()` internally dispatch to grouped aggregation over the
  list's flat values.

## Unresolved Questions

- **`list_contains` integration**: Should `list_contains` be reimplemented as a
  `ListAggregate` with a `Contains` aggregate, or does it remain a separate scalar
  function? It doesn't fit the "reduce N to 1" pattern cleanly since it takes both a
  list and a needle.

- **Multi-column aggregates**: Some aggregates operate on multiple columns
  (e.g., covariance, weighted sum). With arity fixed at 1, these require a struct array
  as input. Is this ergonomic enough, or do we need a separate mechanism?

- **Window functions**: Window functions (rolling sum, rank) operate on sliding windows
  rather than fixed groups. They share the "operate within parent boundaries" property
  but have different execution semantics. Should they be a separate trait or an
  extension of `AggregateFnVTable`?

- **Canonical list decision**: Option A (ListArray) vs Option B (ListViewArray). This
  can be decided independently of the aggregate framework and implemented as a
  preparatory or follow-up change.

- **`CollectList` semantics**: `CollectList` is the identity aggregate — it returns
  the elements as a list. For the list-scalar case, `list_collect(list_col)` is a no-op.
  For GROUP BY, it materializes the grouped elements into a list column. Should this
  be a built-in aggregate or handled separately?

- **Null list handling**: When a list entry is null, should `flush()` push null into
  the output? Or should the caller skip accumulation and use a separate mechanism to
  mark null outputs? The accumulator currently has no way to signal "this group is null
  without receiving any elements."

## Future Possibilities

- **Partial aggregation with `state()` / `merge()`**: Adding `state()` and
  `merge_batch()` methods to `Accumulator` (like DataFusion) for distributed aggregation
  where partial results must be serialized and merged across nodes. Not needed for
  the sequential processing model in this RFC.

- **`GroupsAccumulator`**: A separate trait (like DataFusion's) that manages all groups
  simultaneously with `group_indices` per row, enabling vectorized hash-based grouping.
  Our per-group flush model handles ordered GROUP BY; this extension would handle
  unordered GROUP BY with hash tables.

- **Aggregate push-down in file scanning**: Using `ListAggregate` reduce rules to
  push aggregates into `LayoutReader`, computing aggregates during file scan without
  materializing full columns.

- **`list_distinct`, `list_sort`, `list_reverse`**: List transformation functions that
  don't reduce elements to a scalar. These are not aggregates and would remain as
  `ScalarFnVTable` implementations.

- **Nested aggregation**: `list_sum(list_of_lists)` producing a list of sums at the
  inner level. The duality recurses naturally.

- **GPU execution**: `execute_grouped` with monotonic offsets maps cleanly to GPU
  segmented reduction primitives (e.g., `cub::DeviceSegmentedReduce`).

- **ListView scatter optimization**: If profiling shows that rebuilding scrambled
  ListViews to sorted form is a bottleneck, aggregate implementations could accept
  `(offsets, sizes)` pairs for scattered access. This would be opt-in per aggregate.
