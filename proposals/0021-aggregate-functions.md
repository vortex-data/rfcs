- Start Date: 2026-02-26
- Tracking Issue: [vortex-data/vortex#6719](https://github.com/vortex-data/vortex/issues/6719)

## Summary

Introduce first-class aggregate functions to Vortex via an `AggregateFnVTable` trait with an
associated `GroupState` type, and a monomorphized `Accumulator<V>` that handles all
orchestration. A single `ListAggregate` scalar function bridges list columns to the aggregate
system, replacing what would otherwise be N separate list scalar functions.

## Motivation

Vortex has aggregate operations (`sum`, `min_max`, `is_constant`, `is_sorted`, `nan_count`)
implemented as standalone `ComputeFn` kernels. These cannot participate in expression trees,
benefit from lazy evaluation, or be optimized via reduce/reduce_parent rules. List scalar
functions like `list_sum`, `list_min`, etc. don't yet exist — and implementing each one
separately would duplicate the underlying aggregation logic.

A list column stored as `(offsets, elements)` is a pre-materialized grouping. Computing
`list_sum(list_col)` is a grouped `sum` over the flat elements partitioned by offsets. Every
aggregate function has a corresponding list scalar function for free:

| Aggregate   | List scalar                | Operation                   |
| ----------- | -------------------------- | --------------------------- |
| `sum`       | `list_sum(list_col)`       | Sum elements per list       |
| `min`       | `list_min(list_col)`       | Min element per list        |
| `max`       | `list_max(list_col)`       | Max element per list        |
| `count`     | `list_count(list_col)`     | Count non-null per list     |
| `mean`      | `list_mean(list_col)`      | Mean of elements per list   |
| `nan_count` | `list_nan_count(list_col)` | Count NaN elements per list |

By implementing N aggregate functions and one `ListAggregate` scalar function, we get both
ungrouped column-level aggregation and grouped list-scalar operations from a single framework.

## Design

### `AggregateFnVTable`

The vtable defines only the pure per-aggregate math via an associated `GroupState` type.
It does not construct accumulators or handle encoding dispatch.

```rust
pub trait AggregateFnVTable: 'static + Sized + Clone + Send + Sync {
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;
    type GroupState: Send;

    fn id(&self) -> AggregateFnId;
    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>>;
    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options>;

    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType>;
    fn state_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType>;

    fn identity(&self, options: &Self::Options, input_dtype: &DType)
        -> VortexResult<Self::GroupState>;

    /// Accumulate a canonical batch into the current group state.
    fn accumulate(
        &self, options: &Self::Options, state: &mut Self::GroupState, batch: &Canonical,
    ) -> VortexResult<()>;

    /// Merge a partial state scalar into the current group state.
    fn merge(
        &self, options: &Self::Options, state: &mut Self::GroupState, partial: &Scalar,
    ) -> VortexResult<()>;

    fn is_saturated(&self, options: &Self::Options, state: &Self::GroupState) -> bool {
        false
    }

    fn finalize(
        &self, options: &Self::Options, state: Self::GroupState,
    ) -> VortexResult<Scalar>;
}
```

The `accumulate` method receives `&Canonical` — `Accumulator<V>` handles canonicalization,
so aggregate authors never deal with encoding dispatch or decompression.

#### Accumulator state

Each aggregate declares a `state_dtype` (Vortex dtype) and a `GroupState` (Rust-native
representation). For multi-field state, use a struct dtype:

| Aggregate    | `state_dtype`                            | `GroupState` example                      |
| ------------ | ---------------------------------------- | ----------------------------------------- |
| `Sum`        | `i64` (or widened input type)            | `SumState::I64(Some(42))`                 |
| `Count`      | `u64`                                    | `u64`                                     |
| `Min`        | input element type                       | `MinState::I32(Some(3))`                  |
| `Mean`       | `Struct { sum: f64, count: u64 }`        | `MeanState { sum: 10.0, count: 5 }`       |
| `IsConstant` | `Struct { value: T, is_constant: bool }` | `IsConstantState { value: .., is: true }` |

The `merge` method combines a partial state `Scalar` (produced by encoding-specific
shortcuts) into the current `GroupState`. This also lays the groundwork for
partial/distributed aggregation where intermediate state must be serialized and merged.

### `DynAccumulator` trait and `Accumulator<V>`

`DynAccumulator` is the type-erased execution interface:

```rust
pub trait DynAccumulator: Send {
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()>;
    fn accumulate_list(&mut self, list: &ListViewArray) -> VortexResult<()>;
    fn merge(&mut self, state: &Scalar) -> VortexResult<()>;
    fn merge_list(&mut self, states: &ArrayRef) -> VortexResult<()>;
    fn is_saturated(&self) -> bool;
    fn flush(&mut self) -> VortexResult<()>;
    fn finish(self: Box<Self>) -> VortexResult<ArrayRef>;
}
```

`Accumulator<V>` is a monomorphized struct that implements `DynAccumulator` for all
aggregate functions, handling encoding dispatch, canonicalization, and output management:

```rust
struct Accumulator<V: AggregateFnVTable> {
    vtable: V,
    options: V::Options,
    input_dtype: DType,
    agg_fn_ref: AggregateFnRef,  // derived from vtable + options, for encoding dispatch
    current: V::GroupState,
    results: Vec<V::GroupState>,
}
```

The `agg_fn_ref` is derived from `vtable` + `options` at construction time. It is needed
because encoding-side dispatch (`aggregate`/`aggregate_list` on `dyn Array`) requires a
type-erased handle for kernel matching.

All vtable calls inside `Accumulator<V>` are static dispatch. The only dynamic dispatch
boundary is the `Box<dyn DynAccumulator>` returned to callers.

#### Accumulation dispatch

```rust
impl<V: AggregateFnVTable> DynAccumulator for Accumulator<V> {
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()> {
        // Try encoding shortcut (Constant, RunEnd, etc.)
        if let Some(state) = batch.aggregate(&self.agg_fn_ref)? {
            return self.merge(&state);
        }
        // Canonicalize and delegate to vtable
        self.vtable.accumulate(&self.options, &mut self.current, &batch.to_canonical()?)
    }

    fn accumulate_list(&mut self, list: &ListViewArray) -> VortexResult<()> {
        // Try encoding-specific grouped kernel on elements
        if let Some(states) = list.elements().aggregate_list(list, &self.agg_fn_ref)? {
            return self.merge_list(&states);
        }
        // Per-group fallback
        for i in 0..list.len() {
            self.accumulate(&list.list_elements_at(i)?)?;
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> VortexResult<()> {
        let identity = self.vtable.identity(&self.options, &self.input_dtype)?;
        let state = std::mem::replace(&mut self.current, identity);
        self.results.push(state);
        Ok(())
    }

    // merge, merge_list, is_saturated, finish delegate to vtable
}
```

Usage:

```rust
// Grouped
let mut acc = aggregate.accumulator(element_dtype)?;
acc.accumulate_list(&list_view)?;
acc.finish()

// Ungrouped
let mut acc = aggregate.accumulator(dtype)?;
for chunk in chunks {
    if acc.is_saturated() { break; }
    acc.accumulate(&chunk)?;
}
acc.flush()?;
acc.finish()
```

### Built-in aggregates

```rust
pub struct Sum;       // overflow saturates to null
pub struct Count;     // non-null elements per group
pub struct NanCount;  // NaN elements per group (float input)
pub struct Min;
pub struct Max;
pub struct Mean;      // returns f64
pub struct Any;       // logical OR (bool input)
pub struct All;       // logical AND (bool input)
```

All use `EmptyOptions`. These replace the standalone `ComputeFn` kernels.

### Encoding-specific optimization

Encodings can short-circuit accumulation by producing partial state directly. Two methods
on the Array VTable:

```rust
fn aggregate(
    &self, array: &Self::Array, aggregate_fn: &AggregateFnRef,
) -> VortexResult<Option<Scalar>>;

fn aggregate_list(
    &self, elements: &Self::Array, list: &ListViewArray, aggregate_fn: &AggregateFnRef,
) -> VortexResult<Option<ArrayRef>>;
```

| Encoding / Elements      | Aggregate | Optimization                                       |
| ------------------------ | --------- | -------------------------------------------------- |
| Constant(5, n=100)       | Sum       | `value * len`                                      |
| RunEnd([1,5,3], [2,5,8]) | Sum       | weighted sum of run values                         |
| Primitive (grouped)      | Sum       | segmented sum: one pass over flat buffer + offsets |
| Constant(5) (grouped)    | Sum       | `value * list.sizes()`                             |
| Dict(codes, values)      | Min       | min code per group → lookup value                  |

#### `aggregate_list` kernel dispatch

Dispatches on the **elements array's encoding**, following the `ParentKernelSet` pattern.
Each kernel matches on the aggregate function type:

```rust
pub trait AggregateListKernel<V: VTable>: Debug + Send + Sync + 'static {
    type Agg: AggregateFnVTable;
    fn aggregate_list(
        &self, array: &V::Array, list: &ListViewArray,
        options: &<Self::Agg as AggregateFnVTable>::Options,
    ) -> VortexResult<Option<ArrayRef>>;
}
```

Encodings register kernels via static kernel sets (e.g., `PrimitiveVTable` registers
segmented sum/min/max, `ConstantVTable` registers algebraic shortcuts).

#### Selectivity trade-offs

`accumulate_list` intentionally does **not** canonicalize the entire elements array. A
ListView can reference a sparse subset of a large elements array (e.g., after filtering
groups). The dispatch:

1. Try `aggregate_list` on the raw elements encoding.
2. Fall back to per-group slicing — each slice canonicalized independently.

For sparse ListViews (100 selected elements out of 10M), this avoids decompressing the
entire elements array. For dense ListViews over compressed encodings without a kernel, the
per-group path performs N small decompressions. The fix is to register an `aggregate_list`
kernel for hot combinations.

### `ListAggregate` scalar function

A single `ScalarFnVTable` bridging list columns to the aggregate system. Produces a
`ScalarFnArray`, reusing existing lazy evaluation, slicing, and reduce infrastructure.

```rust
pub struct ListAggregate;

pub struct ListAggregateOptions {
    pub aggregate_fn: AggregateFnRef,
}

impl ScalarFnVTable for ListAggregate {
    type Options = ListAggregateOptions;

    fn execute(&self, options: &Self::Options, args: ExecutionArgs) -> VortexResult<ArrayRef> {
        let list = args.inputs[0].to_listview()?;
        let mut acc = options.aggregate_fn.accumulator(list.elements().dtype())?;
        acc.accumulate_list(&list)?;
        acc.finish()
    }
}
```

Expression sugar: `list_sum(expr)`, `list_min(expr)`, `list_max(expr)`, etc. — each
constructs a `ListAggregate` expression with the appropriate bound aggregate function.

### Reduce rules

Since `ListAggregate` is a `ScalarFnVTable`, it participates in reduce/reduce_parent:

- **Self-reduce**: constant list folding, count from list sizes, min/max from statistics.
- **Parent-reduce**: child encodings match on `ExactScalarFn<ListAggregate>` (e.g., Dict
  pushes down Min/Max to values, RunEnd converts Sum to weighted sum).

### Aggregate push-down

Reduce rules can push aggregates into the Scan API (e.g., Count from row group metadata,
Min/Max from column-chunk statistics). Details are out of scope for this RFC.

## Compatibility

No file format or wire format changes. Public API additions:

- `DynAccumulator` trait, `Accumulator<V>` implementation
- `AggregateFnVTable` trait with `GroupState` and built-in implementations
- `ListAggregate` scalar function and expression constructors

## Drawbacks

- **New trait surface area.** `AggregateFnVTable` and `DynAccumulator` mirror existing
  `ScalarFnVTable` patterns.
- **Reduce rule coverage.** Not all (encoding, aggregate) pairs will have optimized kernels
  initially. The per-group fallback is correct but slower.
- **`GroupState` as enum.** Aggregates like Sum need an enum over dtypes (I64, F64, Decimal).
  The match is per-batch, so branch cost is negligible.
- **`merge` accepts `&Scalar`, `finalize` returns `Scalar`.** These are at the boundary
  between typed vtable and erased encoding system. Overhead is per-chunk/per-group, not
  per-element.

## Alternatives

### Separate list scalar functions

Individual `ListSum`, `ListMin`, etc. as `ScalarFnVTable` implementations. Duplicates logic
across N functions with no shared optimization path.

### Keep aggregates as `ComputeFn` only

No expression tree participation, no lazy evaluation, no reduce_parent optimization.

### Dedicated `AggregateFnArray`

Structurally identical to `ScalarFnArray` with one child. `ExactScalarFn<ListAggregate>`
provides the same typed matching without a new array type.

### Per-function `dyn DynAccumulator` without `GroupState`

Each aggregate implements `DynAccumulator` directly via an `accumulator()` factory method,
with no `GroupState` associated type. Every aggregate reimplements encoding dispatch,
per-group fallback, canonicalization, and output buffer management. The `GroupState` design
factors all orchestration into `Accumulator<V>`, written once.

## Future Possibilities

- **Partial aggregation**: `state_dtype` and `merge` enable serializing intermediate state
  for distributed execution.
- **Aggregate push-down in Scan**: push aggregates into `LayoutReader` during file scan.
- **`finalize_batch`**: bulk `Vec<GroupState> -> ArrayRef` without per-group `Scalar`
  allocation.
- **Fused selection + aggregation**: process only selected regions of compressed elements
  without full decompression.
- **Window functions**: separate trait with different execution semantics.
