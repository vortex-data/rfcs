- Start Date: 2026-02-26
- Tracking Issue: [vortex-data/vortex#6719](https://github.com/vortex-data/vortex/issues/6719)

## Summary

Introduce first-class aggregate functions to Vortex via an `AggregateFnVTable` trait and
`Accumulator` interface. A single `ListAggregate` scalar function bridges list columns to the
aggregate system, replacing what would otherwise be N separate list scalar functions.

## Motivation

Vortex has aggregate operations (`sum`, `min_max`, `is_constant`, `is_sorted`, `nan_count`)
implemented as standalone `ComputeFn` kernels. These cannot participate in expression trees, benefit from lazy
evaluation, or be optimized via reduce/reduce_parent rules. Meanwhile, list scalar functions
like `list_sum`, `list_min`, etc. don't yet exist — and implementing each one separately would
duplicate the underlying aggregation logic.

The key observation is that a list column stored as `(offsets, elements)` is a pre-materialized
grouping. Computing `list_sum(list_col)` is a grouped `sum` over the flat elements partitioned
by offsets. Every aggregate function has a corresponding list scalar function for free:

| Aggregate   | List scalar                | Operation                   |
| ----------- | -------------------------- | --------------------------- |
| `sum`       | `list_sum(list_col)`       | Sum elements per list       |
| `min`       | `list_min(list_col)`       | Min element per list        |
| `max`       | `list_max(list_col)`       | Max element per list        |
| `count`     | `list_count(list_col)`     | Count non-null per list     |
| `mean`      | `list_mean(list_col)`      | Mean of elements per list   |
| `nan_count` | `list_nan_count(list_col)` | Count NaN elements per list |

Since Vortex does not support shuffling, grouped aggregates only apply to pre-existing groups.
These are naturally represented by List or ListView encodings as a view over the elements array.
By implementing N aggregate functions and one `ListAggregate` scalar function, we get both
ungrouped column-level aggregation and grouped list-scalar operations from a single framework.

## Design

### `Accumulator`

The `Accumulator` is the single execution interface for all aggregation. It processes one group
at a time: the caller feeds element batches via `accumulate`, then calls `flush` to finalize
the group and begin the next. Encodings can short-circuit by producing partial state (via
`aggregate` / `aggregate_list` on the array vtable) that is merged into the accumulator.
The accumulator owns an output buffer and returns all results at the end.

```rust
pub trait Accumulator: Send + Sync {
    /// Feed a batch of elements for the currently open group.
    /// May be called multiple times per group (e.g., chunked elements).
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()>;

    /// Accumulate all groups defined by a ListView in one call.
    /// Default: for each group, accumulate its elements then flush.
    /// Override for vectorized fast paths (e.g., segmented sum over the flat
    /// elements + offsets without per-group slicing).
    fn accumulate_list(&mut self, list: &ListViewArray) -> VortexResult<()> {
        for i in 0..list.len() {
            self.accumulate(&list.list_elements_at(i)?)?;
            self.flush()?;
        }
        Ok(())
    }

    /// Merge pre-computed partial state into the currently open group.
    /// The scalar's dtype must match the aggregate's `state_dtype`.
    /// This is equivalent to having processed raw elements that would produce
    /// this state — used by encoding-specific optimizations (see aggregate).
    fn merge(&mut self, state: &Scalar) -> VortexResult<()>;

    /// Merge an array of pre-computed states, one per group, flushing each.
    /// The array's dtype must match the aggregate's `state_dtype`.
    /// Default: merge + flush for each element.
    fn merge_list(&mut self, states: &ArrayRef) -> VortexResult<()> {
        for i in 0..states.len() {
            self.merge(&states.scalar_at(i)?)?;
            self.flush()?;
        }
        Ok(())
    }

    /// Whether the currently open group's result is fully determined.
    /// When true, callers may skip further accumulate/merge calls and proceed
    /// directly to flush. Resets to false after flush().
    /// Examples: IsConstant after seeing two distinct values, All after seeing false.
    fn is_saturated(&self) -> bool { false }

    /// Finalize the currently open group: push its result to the output buffer
    /// and reset internal state for the next group.
    ///
    /// Flushing a group with zero accumulated elements produces the aggregate's
    /// identity value (e.g., 0 for Sum, u64::MAX for Min) or null if no identity
    /// exists. If accumulation fails mid-group, the accumulator is left in an
    /// unspecified state — callers should not flush after an error.
    fn flush(&mut self) -> VortexResult<()>;

    /// Return all flushed results as a single array.
    /// Length = number of flush() calls made over the accumulator's lifetime.
    fn finish(self: Box<Self>) -> VortexResult<ArrayRef>;
}
```

Usage across all aggregation patterns:

```rust
// Grouped (list scalar): fast path processes all groups at once
let mut acc = aggregate.accumulator(element_dtype)?;
acc.accumulate_list(&list_view)?;
acc.finish()  // ArrayRef of length n_lists

// Ungrouped (full-column): single group, fold across chunks
let mut acc = aggregate.accumulator(dtype)?;
for chunk in chunked_array.chunks() {
    if acc.is_saturated() { break; }
    acc.accumulate(&chunk)?;
}
acc.flush()?;
acc.finish()  // 1-element ArrayRef
```

#### Accumulator state

Each aggregate declares a `state_dtype` — the type of its intermediate accumulator state.
State is a single `Scalar` whose dtype matches this declaration. For aggregates with multiple
fields, use a struct dtype:

| Aggregate    | `state_dtype`                            | Example state value                       |
| ------------ | ---------------------------------------- | ----------------------------------------- |
| `Sum`        | `i64` (or widened input type)            | `Scalar(42)` — overflow saturates to null |
| `Count`      | `u64`                                    | `Scalar(7)`                               |
| `NanCount`   | `u64`                                    | `Scalar(2)`                               |
| `Min`        | input element type                       | `Scalar(3)`                               |
| `Mean`       | `Struct { sum: f64, count: u64 }`        | `Scalar({sum: 10.0, count: 5})`           |
| `IsConstant` | `Struct { value: T, is_constant: bool }` | `Scalar({value: 5, is_constant: true})`   |
| `IsSorted`   | `Struct { last: T, is_sorted: bool }`    | `Scalar({last: 9, is_sorted: true})`      |

The `merge` method on `Accumulator` combines a partial state scalar into the currently open
group. For Sum, this is addition. For IsConstant, this checks whether the incoming value
matches the seen value. The `merge_list` method handles multiple groups at once.

This enables encoding-specific optimization (see below) and also lays the groundwork for
partial/distributed aggregation where intermediate state must be serialized and merged
across nodes.

### `AggregateFnVTable`

A new trait parallel to `ScalarFnVTable`:

```rust
pub trait AggregateFnVTable: 'static + Sized + Clone + Send + Sync {
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    fn id(&self) -> AggregateFnId;

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>>;
    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options>;

    /// Result dtype per group.
    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType>;

    /// DType of the intermediate accumulator state.
    /// Use a struct dtype when multiple fields are needed (e.g., Mean: {sum: f64, count: u64}).
    fn state_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType>;

    /// Create an accumulator for streaming aggregation.
    fn accumulator(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Box<dyn Accumulator>>;
}
```

The `Accumulator` is the single execution interface. Grouped aggregation uses
`accumulate_list`; ungrouped aggregation uses `accumulate`/`flush`/`finish` directly.
Encodings can short-circuit by producing partial state (via `aggregate`/`aggregate_list` on
the array vtable) that is merged into the accumulator via `merge`/`merge_list`. There is no
need for `execute_grouped` or `execute_scalar` methods on the vtable — the accumulator
handles both paths, and its `accumulate_list` override is where vectorized fast paths live.

### Built-in aggregates

The initial set, each implementing `AggregateFnVTable`:

```rust
pub struct Sum;       // sum of elements per group (overflow saturates to null)
pub struct Count;     // count of non-null elements per group
pub struct NanCount;  // count of NaN elements per group (float input)
pub struct Min;       // minimum element per group
pub struct Max;       // maximum element per group
pub struct Mean;      // mean of elements per group (returns f64)
pub struct Any;       // logical OR per group (bool input)
pub struct All;       // logical AND per group (bool input)
```

All built-in aggregates use `EmptyOptions` as their `Options` type. These replace the
standalone `ComputeFn` kernels (e.g., `Sum` replaces `compute::sum()`).

### Encoding-specific optimization

Arrays can short-circuit accumulation by producing partial state directly, avoiding
decompression. This follows the `execute_parent` pattern: the array sees the aggregate
being applied and returns pre-computed state.

Two new methods on the Array VTable:

```rust
/// Produce partial accumulator state for the given aggregate, treating the
/// entire array as a single group.
/// Returns None to fall back to element-by-element accumulation.
fn aggregate(
    &self,
    array: &Self::Array,
    aggregate_fn: &AggregateFnRef,
) -> VortexResult<Option<Scalar>>;

/// Produce partial accumulator state for each group defined by a ListView
/// over this array. Returns an array of state values (one per group) with
/// dtype = aggregate_fn.state_dtype() and length = list.len().
/// Returns None to fall back to per-group accumulation.
fn aggregate_list(
    &self,
    elements: &Self::Array,
    list: &ListViewArray,
    aggregate_fn: &AggregateFnRef,
) -> VortexResult<Option<ArrayRef>>;
```

**Ungrouped examples** (`aggregate` returns `Option<Scalar>`):

| Encoding                 | Aggregate  | Returns                                |
| ------------------------ | ---------- | -------------------------------------- |
| Constant(5, n=100)       | Sum        | `Some(Scalar(500))` — value \* len     |
| Constant(5, n=100)       | IsConstant | `Some({value: 5, is_constant: true})`  |
| RunEnd([1,5,3], [2,5,8]) | Sum        | `Some(Scalar(26))` — weighted sum      |
| RunEnd(...)              | Min        | `Some(Scalar(1))` — min of run values  |
| Primitive                | Sum        | `None` — no shortcut, process elements |

**Grouped examples** (`aggregate_list` returns `Option<ArrayRef>`):

| Elements encoding   | Aggregate  | Optimization                             |
| ------------------- | ---------- | ---------------------------------------- |
| Constant(5)         | Sum        | `constant * list.sizes()` — one multiply |
| Constant(5)         | IsConstant | All groups constant with same value      |
| Dict(codes, values) | Min        | Min code per group → look up value       |
| Dict(codes, values) | Max        | Max code per group → look up value       |

The accumulator wires these into its methods:

```rust
// In accumulate():
if let Some(state) = batch.aggregate(&self.aggregate_fn)? {
    return self.merge(&state);
}
// ... fall back to canonical processing

// In accumulate_list() default:
if let Some(states) = list.elements().aggregate_list(list, &self.aggregate_fn)? {
    return self.merge_list(&states);
}
// ... fall back to per-group slice + accumulate + flush
```

The encoding doesn't need to know accumulator internals — it produces state matching the
aggregate's declared `state_dtype`. The accumulator knows how to merge it.

### `ListAggregate` scalar function

A single `ScalarFnVTable` that bridges list columns to the aggregate system. Because it is a
scalar function, wrapping it in an expression produces a `ScalarFnArray` — reusing the
existing lazy evaluation, slicing, and reduce infrastructure with no new array type.

```rust
pub struct ListAggregate;

pub struct ListAggregateOptions {
    pub aggregate_fn: AggregateFnRef,
}

impl ScalarFnVTable for ListAggregate {
    type Options = ListAggregateOptions;

    fn execute(&self, options: &Self::Options, args: ExecutionArgs) -> VortexResult<ArrayRef> {
        let list = args.inputs[0].to_listview()?;
        let agg = &options.aggregate_fn;

        // Try encoding-specific fast path first.
        if let Some(states) = list.elements().aggregate_list(&list, agg)? {
            let mut acc = agg.accumulator(list.elements().dtype())?;
            acc.merge_list(&states)?;
            return acc.finish();
        }

        // Fall back to accumulator-driven execution.
        let mut acc = agg.accumulator(list.elements().dtype())?;
        acc.accumulate_list(&list)?;
        acc.finish()
    }

    // return_dtype delegates to aggregate_fn.return_dtype over the list element type.
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
// list_min, list_max, list_count, list_nan_count, list_mean, list_any, list_all analogously
```

This is one scalar function parameterized by the aggregate, analogous to DuckDB's
`list_aggregate(list, 'sum')`.

### Reduce rules

Since `ListAggregate` is a `ScalarFnVTable`, it participates in the existing reduce/reduce_parent
optimization framework:

**Self-reduce** (`ScalarFnVTable::reduce`): constant list folding, count from list sizes,
min/max from statistics, sum of constant elements.

**Parent-reduce** (encoding-specific): child encodings match on `ExactScalarFn<ListAggregate>`
to optimize specific aggregate + encoding combinations. For example:

- **Dict**: `ListAggregate(Min/Max, List(Dict(codes, values)))` pushes down to values.
- **RunEnd**: `ListAggregate(Sum, List(RunEnd))` becomes a weighted sum over run values.

### Aggregate push-down

Aggregate reduce rules can push computation into the Scan API, allowing aggregates to be
computed during file scanning without materializing full columns. For example, `Count` can
be resolved from row group metadata alone; `Min`/`Max` can use column-chunk statistics.
The details of scan-level push-down are out of scope for this RFC.

## Compatibility

No file format or wire format changes. `ListAggregate` produces a `ScalarFnArray` at runtime
and is not persisted. Public API additions:

- `Accumulator` trait
- `AggregateFnVTable` trait and built-in implementations
- `ListAggregate` scalar function
- Expression constructors: `list_sum()`, `list_count()`, `list_nan_count()`, `list_min()`,
  `list_max()`, `list_mean()`, `list_any()`, `list_all()`

## Drawbacks

- **New trait surface area.** `AggregateFnVTable` and `Accumulator` are new traits, though
  they closely mirror existing `ScalarFnVTable` patterns.

- **Reduce rule coverage.** Not all encoding x aggregate combinations will have optimized
  reduce_parent rules initially. The fallback (canonicalize + accumulator loop) is correct
  but slower.

## Alternatives

### Separate list scalar functions

Implement `ListSum`, `ListMin`, etc. as individual `ScalarFnVTable` implementations.
Rejected: duplicates logic across N functions, no shared optimization, no path to reuse
for ungrouped aggregation.

### Keep aggregates as `ComputeFn` only

Rejected: no lazy evaluation, no expression tree participation, no reduce_parent optimization.

### Dedicated `AggregateFnArray`

A new array type wrapping an aggregate + list child, parallel to `ScalarFnArray`.
Rejected: structurally identical to `ScalarFnArray` with one child — duplicates existing
lazy evaluation and reduce infrastructure. `ExactScalarFn<ListAggregate>` provides the same
typed matching without a new array type.

## Future Possibilities

- **Partial aggregation** (`state()` / distributed `merge`): the `state_dtype` and `merge`
  infrastructure enables serializing intermediate state for distributed execution. A
  `state()` export method on `Accumulator` would complete this.

- **Aggregate push-down in Scan**: using reduce rules to push aggregates into `LayoutReader`,
  computing results during file scan without materializing full columns.

- **Window functions**: sliding-window operations share the "operate within boundaries"
  property but have different execution semantics. A separate trait or extension is more
  appropriate.
