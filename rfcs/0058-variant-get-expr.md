- Start Date: 2026-05-05
- Authors: @AdamGS
- RFC PR: [vortex-data/rfcs#57](https://github.com/vortex-data/rfcs/pull/57)

# VariantGet Expression

## Summary

Introduce a new `VariantGet` expression that extracts useable data from variant arrays.

## Motivation

As described in the [Variant RFC](https://github.com/vortex-data/rfcs/blob/develop/rfcs/0015-variant-type.md),
variants arrays are useful for many use cases, but in order to actually use the data a fully typed array is required.

## Design

### Definition

A new VariantGet expression is required, the expression has two inputs:

1. Path to the required child - similar to JSONPath, but a much stricter subset. Just a combination of names and indexes.
2. Optional dtype, if None - the return type is `None`, the expression's return type is `Variant`.

### Array

The canonical Variant array will add an additional child, representing optional shredded data, it will now have:

1. Validity
2. Core storage - containing the raw unshredded data, which can be encoded in any way the child array's encoding.
3. An optional shredded child - a tree of fully typed arrays for paths that were shredded during
   the array's creation.

The shredded child is an explicit child of the canonical Variant array. It has the same length as
`core_storage`, and its rows must stay aligned with the raw variant rows.

Nested shredded paths can be represented by nesting typed arrays inside struct arrays. For example,
if `$.a.b` is shredded but `$.a.c` is not, the shredded child may contain a field for `a`, whose
own child contains a typed field for `b`. Paths that are not represented by the shredded child are
still read from `core_storage`.

### Execution

When executing the expression on a variant array, it will pull out recursively shredded data until the path is exhausted OR the path reached a child path that isn't shredded. As we traverse the chain of shredded children along the path, we'll need to make sure to keep track of their validity, as the leaf child's validity is an OR of all of them.

At this point, we have 3 possible cases:

1. Perfectly shredded - there's a fully shredded child at this path. If it matches the expected type or can be casted into it, we can just return it. Note that this child might actually be a Variant array with its own shredded children, this just means that we've reached a position where all data is contained within this child, with no relevant data in the "core storage" child.
2. Partially shredded - data for this path exists in both the shredded child AND in some unshredded values, which we can merge according to the expected type.
3. Unshredded - No shredded child at this path, we try and extract the relevant value from the unshredded values which are unchanged from the original array.

The important invariant is that `VariantGet` changes the typed child selected for the requested
path, but it does not rewrite the raw unshredded data. The raw storage continues to represent the
same original variant values and can still be used by later `VariantGet` expressions for paths that
were not shredded.

For example, `VariantGet("$.a.b", i64)` changes only the typed view of the requested path:

```text
Variant array before VariantGet("$.a.b", i64)

+------------------------------------------------------------------------+
| validity                                                               |
| raw unshredded data  ------------------------------ unchanged -------- |
| shredded children                                                      |
|   $.a.b: utf8 / missing / partially materialized                       |
|   $.x.y: bool                                                          |
+------------------------------------------------------------------------+
                                      |
                                      | VariantGet("$.a.b", i64)
                                      v
+------------------------------------------------------------------------+
| validity for rows where $.a.b can be read as i64                       |
| raw unshredded data  ------------------------------ unchanged -------- |
| typed child: i64 values for $.a.b                                      |
|   built from shredded data, raw data, or a merge of both               |
+------------------------------------------------------------------------+
```

### Pushdown, Filter and Slice

The canonical `VariantArray` is the stable execution boundary, but it should not force
`VariantGet` to materialize the whole variant value. When `VariantGet` sees a canonical variant, it
first uses the explicit `shredded` child when that child contains the requested path. If the path is
not fully represented by the shredded child, execution continues against `core_storage` for the
remaining unshredded values. This allows encoding-specific kernels, such as Parquet Variant, to
implement path extraction directly against their raw representation.

This pushdown is a path-extraction pushdown, not predicate pushdown. A predicate over
`VariantGet(v, path, dtype)` is still evaluated over the extracted result. The important part is
that extracting the path does not first decode unrelated paths from the variant value.

`Filter` and `Slice` interact with variants as row-preserving transformations:

1. `Filter(variant, mask)` filters `core_storage` with the same mask.
2. `Slice(variant, range)` slices `core_storage` with the same range.
3. If the variant has a `shredded` child, the same filter or slice is applied to that child.
4. The resulting canonical variant is rebuilt from the transformed `core_storage` and transformed
   `shredded` child.

This keeps the raw unshredded data and the shredded child row-aligned without rewriting the raw
variant payload. For example, `VariantGet(Slice(v, 10..20), "$.a", i64)` first produces a sliced
variant whose `core_storage` and shredded data both cover rows `10..20`; `VariantGet` then extracts
from that sliced shredded child, sliced `core_storage`, or a merge of both. The same applies to
filtered variants: `VariantGet(Filter(v, m), "$.a", i64)` sees only the selected rows, and any
shredded child used for `$.a` has been filtered with the same mask.

If an encoding does not implement `VariantGet` directly, execution can continue by executing the
`core_storage` into a lower-level representation. If no execution step makes progress, the
expression errors rather than silently returning an incorrectly decoded array.

## Compatibility

This extends the canonical `VariantArray` shape, as implemented in
[vortex-data/vortex#7494](https://github.com/vortex-data/vortex/pull/7494). Instead of a single
variant child, the canonical array exposes a required `core_storage` child and an optional logical
`shredded` child.

This does not change the `Variant` dtype semantics or rewrite the raw unshredded values.
Compatibility is limited to code and serialized data that assumes the old canonical variant array
shape (which we've made an effort to make sure doesn't exist). Readers, writers, and array
transformations that handle canonical variants need to use the new `core_storage` and `shredded`
accessors rather than assuming there is only one child.

## Drawbacks

This makes canonical variants more complex than a single raw child. Any code that transforms a
canonical `VariantArray` must preserve both `core_storage` and the optional `shredded` child, and
must keep them row-aligned through filter, slice, take and mask operations.

The expression also pushes complexity into variant encodings. Each encoding can fall back to raw
extraction, but good performance requires encoding-specific `VariantGet` support that understands
its own raw representation and how to merge that with shredded values.

Partial shredding is the highest-risk part of the design. If the same logical path can be served
from both the shredded child and `core_storage`, the implementation has to maintain a clear
precedence rule and test that the merged result is identical to extracting from the original raw
variant values.

## Alternatives

We can make the dtype parameter required, but I do think that the optional one keeps execution more flexible and opens up
opportunities for different usage, which is useful for compute engines that have more flexible type systems or that might want
to process the raw byte data themselves.

## Prior Art

See the [Variant RFC](https://github.com/vortex-data/rfcs/blob/develop/rfcs/0015-variant-type.md).

## Unresolved Questions

- What exact path grammar should `VariantGet` support? This RFC assumes a strict subset of
  JSONPath with field names and list indexes, but still needs to specify escaping, quoted names and
  whether negative indexes or wildcards are out of scope.
- What casts are allowed when `as_dtype` is provided? Numeric widening seems reasonable, but string
  parsing, lossy casts and timestamp/decimal coercions should be decided explicitly.
- What are the exact null semantics for outer nulls, missing paths, `variantnull` values and type
  mismatches? Typed extraction likely returns null for all of these cases, but untyped extraction
  needs to preserve the distinction between a missing result and a present variant null where
  possible.
- How should implementations validate consistency between the shredded child and raw
  `core_storage`? This may be a construction-time invariant, a debug assertion or a checked error
  path when merging partial shredding.
- What shape should the shredded tree use for list indexes and nested variants? Struct fields cover
  object paths naturally, but array indexes and leaves that are themselves `Variant` need a precise
  representation.
- Automatic shredding policy is out of scope for this RFC. The compressor can decide which paths to
  shred later; this RFC only defines how extracted paths are represented and executed once shredded
  data exists.
