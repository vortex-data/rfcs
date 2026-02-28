- Start Date: 2026-02-26
- RFC PR: [vortex-data/rfcs#0020](https://github.com/vortex-data/rfcs/pull/0020)
- Tracking Issue: [vortex-data/vortex#0000](https://github.com/vortex-data/vortex/issues/0000)

## Summary

Introduce first-class aggregate functions to Vortex via an `AggregateFnVTable` trait and
`Accumulator` interface. A single `ListAggregate` scalar function bridges list columns to the
aggregate system, replacing what would otherwise be N separate list scalar functions.

## Motivation

Vortex has aggregate operations (`sum`, `min_max`, `is_constant`, `is_sorted`) implemented as
standalone `ComputeFn` kernels. These cannot participate in expression trees, benefit from lazy
evaluation, or be optimized via reduce/reduce_parent rules. Meanwhile, list scalar functions
like `list_sum`, `list_min`, etc. don't yet exist — and implementing each one separately would
duplicate the underlying aggregation logic.

The key observation is that a list column stored as `(offsets, elements)` is a pre-materialized
grouping. Computing `list_sum(list_col)` is a grouped `sum` over the flat elements partitioned
by offsets. Every aggregate function has a corresponding list scalar function for free:

| Aggregate  | List scalar          | Operation                  |
|------------|----------------------|----------------------------|
| `sum`      | `list_sum(list_col)` | Sum elements per list      |
| `min`      | `list_min(list_col)` | Min element per list       |
| `max`      | `list_max(list_col)` | Max element per list       |
| `count`    | `list_count(list_col)` | Count non-null per list  |
| `mean`     | `list_mean(list_col)` | Mean of elements per list |

Since Vortex does not support shuffling, grouped aggregates only apply to pre-existing groups.
These are naturally represented by List or ListView encodings as a view over the elements array.
By implementing N aggregate functions and one `ListAggregate` scalar function, we get both
ungrouped column-level aggregation and grouped list-scalar operations from a single framework.

## Design

### `Accumulator`

The `Accumulator` trait is the core aggregation primitive. It processes one group at a time:
the caller feeds element batches via `accumulate`, then calls `flush` to finalize the group
and begin the next. The accumulator owns an output buffer and returns all results at the end.

```rust
pub trait Accumulator: Send + Sync {
    /// Feed a batch of elements for the current group.
    /// May be called multiple times per group (e.g., chunked elements).
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()>;

    /// Finalize the current group: push its result to the output buffer and reset
    /// internal state for the next group.
    fn flush(&mut self) -> VortexResult<()>;

    /// Return all flushed results as a single array. Length = number of flush() calls.
    fn finish(self: Box<Self>) -> VortexResult<ArrayRef>;
}
```

Usage across all aggregation patterns:

```rust
// Grouped (list scalar): one group per list element
let mut acc = aggregate.accumulator(element_dtype)?;
for i in 0..n_lists {
    acc.accumulate(&elements.slice(offsets[i]..offsets[i+1])?)?;
    acc.flush()?;
}
acc.finish()  // ArrayRef of length n_lists

// Ungrouped (full-column): single group, fold across chunks
let mut acc = aggregate.accumulator(dtype)?;
for chunk in chunked_array.chunks() {
    acc.accumulate(&chunk)?;
}
acc.flush()?;
acc.finish()  // 1-element ArrayRef
```

#### Accumulator state

Some aggregates require non-trivial intermediate state to process groups across multiple
`accumulate` calls. Two good examples:

**`IsConstant`** — the accumulator must track the value seen so far. If a subsequent batch
contains a different value, the group is not constant:

```rust
struct IsConstantAccumulator {
    seen_value: Option<Scalar>,  // None until first non-null element
    is_constant: bool,
    output: Vec<bool>,
}

impl Accumulator for IsConstantAccumulator {
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()> {
        if !self.is_constant { return Ok(()); }  // already failed
        match &self.seen_value {
            None => {
                // First batch: record the value if constant
                if let Some(true) = is_constant(batch.as_ref())? {
                    self.seen_value = Some(batch.scalar_at(0)?);
                } else {
                    self.is_constant = false;
                }
            }
            Some(val) => {
                // Subsequent batch: check all elements match
                if let Some(true) = is_constant(batch.as_ref())? {
                    if &batch.scalar_at(0)? != val {
                        self.is_constant = false;
                    }
                } else {
                    self.is_constant = false;
                }
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> VortexResult<()> {
        self.output.push(self.is_constant);
        self.seen_value = None;
        self.is_constant = true;
        Ok(())
    }

    fn finish(self: Box<Self>) -> VortexResult<ArrayRef> {
        Ok(BoolArray::from_iter(self.output).into_array())
    }
}
```

**`IsSorted`** — the accumulator must track the last value seen to compare against the first
element of the next batch:

```rust
struct IsSortedAccumulator {
    last_value: Option<Scalar>,
    is_sorted: bool,
    output: Vec<bool>,
}

impl Accumulator for IsSortedAccumulator {
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()> {
        if !self.is_sorted { return Ok(()); }
        // Check batch is internally sorted
        if is_sorted(batch.as_ref())? != Some(true) {
            self.is_sorted = false;
            return Ok(());
        }
        // Check continuity with previous batch
        if let Some(prev) = &self.last_value {
            let first = batch.scalar_at(0)?;
            if first < *prev {
                self.is_sorted = false;
                return Ok(());
            }
        }
        self.last_value = Some(batch.scalar_at(batch.len() - 1)?);
        Ok(())
    }

    fn flush(&mut self) -> VortexResult<()> {
        self.output.push(self.is_sorted);
        self.last_value = None;
        self.is_sorted = true;
        Ok(())
    }

    fn finish(self: Box<Self>) -> VortexResult<ArrayRef> {
        Ok(BoolArray::from_iter(self.output).into_array())
    }
}
```

These examples show why `accumulate`/`flush`/`finish` is the right decomposition: stateful
aggregates need to carry intermediate values across chunked input for a single group, then
reset cleanly at group boundaries. A simpler `execute_grouped(elements, offsets)` one-shot
API cannot handle chunked elements or streaming input.

Intermediate accumulator state could in the future be stored as typed Vortex arrays or scalars,
enabling serialization for partial/distributed aggregation (see Future Possibilities).

### `AggregateFnVTable`

A new trait parallel to `ScalarFnVTable`. The `accumulator()` method is the required core;
`execute_grouped` and `execute_scalar` have default implementations built on it:

```rust
pub trait AggregateFnVTable: 'static + Sized + Clone + Send + Sync {
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    fn id(&self) -> AggregateFnId;

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>>;
    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options>;

    /// Result dtype per group.
    fn return_dtype(&self, options: &Self::Options, input_dtypes: &[DType]) -> VortexResult<DType>;

    /// Create an accumulator for streaming aggregation.
    fn accumulator(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Box<dyn Accumulator>>;

    /// One-shot grouped execution over elements + monotonic offsets.
    /// Default: loop over groups using the accumulator.
    /// Override for vectorized fast paths (e.g., SIMD segmented reduction).
    fn execute_grouped(
        &self,
        options: &Self::Options,
        elements: &ArrayRef,
        offsets: &ArrayRef,
    ) -> VortexResult<ArrayRef> { /* default impl using accumulator */ }

    /// Ungrouped full-column aggregation returning a scalar.
    /// Default: single-group execute_grouped with offsets [0, n].
    fn execute_scalar(
        &self,
        options: &Self::Options,
        input: &ArrayRef,
    ) -> VortexResult<Scalar> { /* default impl */ }
}
```

### Built-in aggregates

The initial set, each implementing `AggregateFnVTable`:

```rust
pub struct Sum;       // sum of elements per group
pub struct Count;     // count of non-null elements per group
pub struct Min;       // minimum element per group
pub struct Max;       // maximum element per group
pub struct Mean;      // mean of elements per group (returns f64)
pub struct Any;       // logical OR per group (bool input)
pub struct All;       // logical AND per group (bool input)
```

`execute_scalar` replaces standalone `ComputeFn` kernels (e.g., `Sum::execute_scalar` replaces
`compute::sum()`). Accumulator implementations can start simple (canonicalize + iterate) and
gain encoding-aware fast paths over time.

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
        let list = args.inputs[0].to_list()?;
        let elements = list.elements();
        let offsets = list.offsets();
        options.aggregate_fn.execute_grouped(elements, offsets)
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
// list_min, list_max, list_count, list_mean, list_any, list_all analogously
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

## Migration

`execute_scalar` on each `AggregateFnVTable` replaces the equivalent `ComputeFn` kernel:

| Current `ComputeFn`       | New `AggregateFnVTable`      |
|---------------------------|------------------------------|
| `compute::sum()`          | `Sum::execute_scalar()`      |
| `compute::min_max()`      | `Min/Max::execute_scalar()`  |
| `compute::is_constant()`  | `IsConstant::execute_scalar()` |
| `compute::is_sorted()`    | `IsSorted::execute_scalar()` |

Existing `ComputeFn` APIs can be kept as thin wrappers during transition.

## Compatibility

No file format or wire format changes. `ListAggregate` produces a `ScalarFnArray` at runtime
and is not persisted. Public API additions:

- `Accumulator` trait
- `AggregateFnVTable` trait and built-in implementations
- `ListAggregate` scalar function
- Expression constructors: `list_sum()`, `list_count()`, `list_min()`, `list_max()`,
  `list_mean()`, `list_any()`, `list_all()`

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

- **Partial aggregation** (`state()` / `merge()`): serialize intermediate accumulator state
  for distributed execution. Accumulator state stored as typed Vortex scalars/arrays would
  enable this naturally.

- **Aggregate push-down in Scan**: using reduce rules to push aggregates into `LayoutReader`,
  computing results during file scan without materializing full columns.

- **Window functions**: sliding-window operations share the "operate within boundaries"
  property but have different execution semantics. A separate trait or extension is more
  appropriate.
