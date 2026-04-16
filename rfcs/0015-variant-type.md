- Start Date: 2025-02-25
- Authors: @AdamGS
- RFC PR: [vortex-data/rfcs#15](https://github.com/vortex-data/rfcs/pull/15)

# Variant Type

## Summary

Vortex currently requires a strict schema, but real world data is often only semi-structured and deeply hierarchical. Logs, traces and user-generated data often take the form of many sparse fields.

This proposal introduces a new dtype - `Variant`, which can capture data with row-level schema, while storing it in a columnar form that can compress well while being available for efficient analysis in a columnar format.

## Design

We'll start with a rough description of the variant type, as many different systems define in different ways (see the [Prior Art](#prior-art) section at the bottom of the page).

The variant type can be commonly described as the following rust type:

```rust
enum Variant {
	Value(Scalar),
	List(Vec<Variant>),
	Object(BTreeMap<String, Variant>), // Usually sorted to allow efficient key finding
}
```

Here `variantnull` value inside the variant payload is represented as
`Scalar::null(DType::Null)`. That is distinct from the outer nullability of the
`Variant` dtype itself.

Different systems have different variations of this idea, but at its core its a type that can hold nested data with either a flexible or no schema.

Variant types are usually stored in two ways - values that aren't accessed often in some system-specific binary encoding, and some number of "shredded" columns, where a specific key is extracted from the variant and stored in a dense format with a specific type, allowing for much more performant access. This design can make commonly accessed subfields perform like first-class columns, while keeping the overall schema flexible. Shredding policies differ by system, and can be pre-determined or inferred from the data itself or from usage patterns.

This document proposes adding a new `DType::Variant(Nullability)`, a logical type describing this group of data encodings and behavior, with its own canonical representation (see below).

### Arrow representation

Arrow now has a new [canonical extension type](https://arrow.apache.org/docs/format/CanonicalExtensions.html#parquet-variant) to represent Parquet's variant type. I think supporting this encoding will be a good start, but it requires supporting Arrow extension types.

Supporting extension types requires replacing the target `DataType` and nullability with a `Field`, which also includes metadata like a desired extension type. I believe this change is desired, as Vortex DTypes include more information than a plain `arrow::DataType`.

### Nullability

`Variant` should follow the same top-level nullability model as every other Vortex dtype:
`DType::Variant(Nullability)` can be nullable or non-nullable. A nullable variant allows the
array slot itself to be absent. A non-nullable variant guarantees that the slot is present, but it
does **not** guarantee that extracted paths will be non-null.

This is distinct from the semantic null value inside the variant payload, which I'll call
`variantnull`. A `variantnull` is a present variant value whose payload is
`null`, while an outer null is the absence of the variant value itself.
In scalar form this is the difference between `Scalar::null(DType::Variant(Nullability::Nullable))`
and `Scalar::variant(Scalar::null(DType::Null))`.

Typed extraction from a variant should therefore still return nullable arrays even when the source
variant column is non-nullable. A path can be missing in a given row, have an unexpected type, or
evaluate to `variantnull`, and each of those cases becomes null in the extracted child.

Combined with shredding, handling nulls can still be complex and is encoding dependent (like this
[parquet example](https://github.com/apache/parquet-format/blob/master/VariantShredding.md#arrays)
for handling arrays), but that is separate from whether the outer `Variant` column itself is
nullable.

### Expressions

Variant columns are commonly accessed through a combination of column, path and the desired type, which are all required to extract a column with a known type. Our current `GetItem` has two issues:

1. It assumes the input can be executed into a struct array.
2. Access is only based on name.

I suggest we add a new expression - `get_variant_element(path, dtype)` (name TBD) which will support flexible paths and allow extracting children from variants. I use the `path` argument in this document loosely, but a subset of JSONPath might be appropriate here, see the [prior art](#prior-art) section to see how other systems handle it.

Every variant encoding will need to be able to dispatch these behaviors, returning arrays of the expected type.

### Scalar

While there has been talk for a long time of converting the Vortex scalar system from an enum to
length 1 arrays, I do believe the current system actually works very well for variants. A variant
scalar can simply wrap another row-specific `Scalar`, rather than needing a dedicated scalar enum
just for variants.

That model also makes the null semantics explicit. `Scalar::null(DType::Variant(Nullability::Nullable))`
means the variant scalar itself is missing. `Scalar::variant(Scalar::null(DType::Null))` means the
variant is present and its payload is `variantnull`.

Just like when extracting child arrays, Variant's need to support an additional expression, `get_variant_scalar(idx, path, dtype)` that will indicate the desired dtype.

### Stats and pushdown

Statistics will only be collected for shredded children of the variant array. As all variant expressions are typed, this will allow us to not only use the same type of pushdown we currently support, but for row ranges where specific key might exist but with an unexpected type, we could skip it completely.

### Path to usefulness

A key component of making variants useable will be making sure the experience of writing and using them is as straightforward as possible, without forcing them to go through complex builders or serialization (unless they require it).

I can see multiple things we can do:

1. The compressor should support compressing arrays with the JSON extension type into variant columns, initially with a pre-configured policy and with more complex heuristics in the future, as seen in the [JSON Tiles paper](https://db.in.tum.de/~durner/papers/json-tiles-sigmod21.pdf).
2. Add expression to convert UTF-8 arrays formatted as JSON into variants, and vice versa. This can also include some other parsing and utilities to handle JSON.

Its important to note that while I suggest the canonical encoding will be basically opaque with regards to the specific encoding of the child array, we could still compress the children using our hierarchical compressor.

## Prior Art

Many systems have a `Variant` type or similar concept, and they generally differ from each other both in implementation and meaning, I've tried to summarize some of the common ones, but I suggested reading the linked sources, especially [Clickhouse's](#clickhouse) blogpost about their variant, dynamic and JSON types.

### Parquet/Arrow

The full details can be found in the [encoding](https://github.com/apache/parquet-format/blob/master/VariantEncoding.md) and [shredding](https://github.com/apache/parquet-format/blob/master/VariantShredding.md) specification, but I'll try and capture it here to the best of my understanding.

#### Un-shredded columns

Parquet represents the columns is a group with two binary fields - `metadata` and `value`. The `metadata` array contains type information for arrays and objects, including field names and offsets. The `value` array contains the serialized values, each prefaced with a 1-byte header containing basic type information.
In Parquet - the variant type has its [own type system](https://github.com/apache/parquet-format/blob/master/VariantEncoding.md#encoding-types), as they don't have a "scalar" concept, and that type system is also used when its loaded into arrow to save on serialization.

#### Shredded columns

When shredding columns, the data is stored in an optional `typed_value` column, which can be any type (including a `Variant`).
Depending on the level of nesting of the data, there are many cases that need to be considered to differentiate between null and missing values and support for various types. They are all described in the [Variant Shredding](https://github.com/apache/parquet-format/blob/master/VariantShredding.md) specification.

#### Statistics

Statistics are only stored for the shredded columns, at the file/row group or page level.

#### In-Memory

When loaded into memory, Arrow has defined a [canonical extension type](https://arrow.apache.org/docs/format/CanonicalExtensions.html#parquet-variant) to support Parquet's variant type. Its stored as a struct array, which contains a mandatory `metadata` binary child, an optional binary `value` child, and an optional `typed_value` which can be a "variant primitive", list or struct, allowing for nested shredding.

### Clickhouse

As described in [this](https://clickhouse.com/blog/a-new-powerful-json-data-type-for-clickhouse#building-block-2---dynamic-type) fantastic blogpost, Clickhouse offers multiple features that build on top of each other to support similar data:

1. [Variant](https://clickhouse.com/docs/sql-reference/data-types/variant) - Allows for arbitrary nesting of types. Variant can contain integers, strings and arrays of integers, strings or another variant type (note the lack of the "object" variant). Each leaf (`col_x.str` vs `col_x.int32`) column is stored separately with some additional metadata which points to which one is used by each row. Types have to be declared in advance.
2. [Dynamic](https://clickhouse.com/docs/sql-reference/data-types/dynamic) - Like variant, but types don't have to be declared in advance. Shreds a limited number of columns
3. [JSON](https://clickhouse.com/docs/sql-reference/data-types/newjson) - Builds on top of `Dynamic`, with a few specialized features - allowing users to specify known "typed paths", how many dynamic paths and types to support for untyped paths, and some JSON-specific configuration allowing skipping specific JSON paths on insert.
   The full blogpost is worth reading, but Clickhouse's on-disk model roughly mirrors the arrow in-memory format, and they store some metadata outside of the array.

### Others

- Iceberg seems to support the variant type (as described in [this](https://docs.google.com/document/d/1sq70XDiWJ2DemWyA5dVB80gKzwi0CWoM0LOWM7VJVd8/edit?tab=t.0) proposal), but the docs are minimal.
- Datafusion's variant support is being developed [here](https://github.com/datafusion-contrib/datafusion-variant), its unclear to me how much effort is going into it and whether its going to be merged upstream.
- DuckDB doesn't support a variant type. It does have a [Union](https://duckdb.org/docs/stable/sql/data_types/union) type, but its basically a struct. It also seems to have support for Parquet's shredding, but I can't find any docs and seems like PRs are being merged as I'm looking through their issues.
- Databricks supports some specialized [variant functions](https://docs.databricks.com/gcp/en/sql/language-manual/sql-ref-functions-builtin#variant-functions), and their docs show a [good example](https://docs.databricks.com/aws/en/sql/language-manual/functions/is_variant_null) of null vs variant null.

## Unresolved Questions

- Do we want a JSON extension type that automatically compresses as variant?

- ~How do variant expressions operate over different variant encodings?~ Resolved this, just had to talk the new execution model with @joseph-isaacs.

## Future Possibilities

In the future, we could add a Vortex-native encoding, but at this point in time it seems like 3rd-party integration is a more useful target.

As mentioned above, I believe starting with a simple shredding policy in the compressor is the best way forward, but exploring things like JSON Tiles could prove to be useful.

Integration with query engines will be an ongoing effort, depending on what features they support and how expressive they are.
