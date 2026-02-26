- Start Date: 2025-02-25
- RFC PR: [vortex-data/rfcs#15](https://github.com/vortex-data/rfcs/pull/15)
- Tracking Issue: [vortex-data/vortex#0000](https://github.com/vortex-data/vortex/issues/0000)

# Variant Type

## Summary

Vortex currently requires a strict schema, but real world data is often only semi-structured. Logs, traces and user-generated data often try to capture generally sparse data, which requires some processing to make it useful for most analytical systems.

This proposal introduces a new type - `Variant`, which can capture data with row-level schema, while storing it in a columnar form that can compress well while being available for efficient analysis in a columnar format.

## Design

We'll start with a rough description of the variant type, as many different systems define in different ways (see the [Prior Art] section at the bottom of the page).

The variant can be commonly described as the following rust type:

```rust
enum Variant {
	Value(Scalar),
	List(Vec<Variant>),
	Object(BTreeMap<String, Variant>), // Usually sorted to allow efficient key finding
}
```

Different systems have different variations of this idea, but at its core its a type that can hold nested data with either a flexible or no schema. In addition to this "catch all" column, most system include the concept of "shredding", extracting a key with a specific type out of this column, and storing it in a dense way. This design can make commonly access subfields perform like first-class columns, while keeping the overall schema flexible.

I propose a new dtype - `DType::Variant`. The variant type is always nullable, and its canonical encoding is just an array with a single child array, which is encoded in some specialized variant type.

### Nullability

In order to support data with a changing or unexpected schema, Variant arrays are always nullable, even for a specific key/path, its value might change type between items which will cause null values in shredded children.

Combined with shredding, handling nulls can be complex and is encoding dependent (Like this [parquet example](https://github.com/apache/parquet-format/blob/master/VariantShredding.md#arrays) for handling arrays).

### Expressions

Variant columns are commonly accessed through a combination of column, path and the desired type, which are all required to extract a column with a known type. Our current `GetItem` has two issues:

1. It assumes the input can be executed into a struct array.
2. Access is only based on name.

I suggest we add a new expression - `get_variant_element(path, dtype)` (name TBD) which will support flexible paths and allow extracting children from variants.

Every variant encoding will need to be able to dispatch these behaviors, returning arrays of the expected type.

### Arrow representation

Arrow now has a new [canonical extension type](https://arrow.apache.org/docs/format/CanonicalExtensions.html#parquet-variant) to represent Parquet's variant type. I think supporting this encoding will be a good start, but it requires supporting Arrow extension types.

Supporting extension types requires replacing the target `DataType` and nullability with a `Field`, which also includes metadata like a desired extension type. I belive this change is desired, as Vortex DTypes include more information than a plain `arrow::DataType`.

### Scalar

While there has been talk for a long time of converting the Vortex scalar system from an enum to length 1 arrays, I do believe the current system actually works very well for variants, and the Variant scalar can just be some version of the type described above.

Just like when extracting child arrays, Variant's need to support an additional expression, `get_variant_scalar(idx, path, dtype)` that will indicate the desired dtype.

### Constructing and writing scalars

The API for creating variant arrays is complex, as shredding decisions need to be made either before hand based on data-specific knowledge, or on the fly during writes.

In the medium/long term, I believe the compressor should support a JSON extension type, which will take JSON formatted UTF8 column, and parse it gradually into a binary formatted and typed variant encoding.

## Prior Art

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

1. [Variant](https://clickhouse.com/docs/sql-reference/data-types/variant) - Allows for arbitrary nesting of types. Variant can contains ints, strings and arrays of ints, strings or another variant type (note the lack of the "object" variant). Each leaf (`col_x.str` vs `col_x.int32`) column is stored separately with some additional metadata which points to which one is used by each row. Types have to be declared in advance.
2. [Dynamic](https://clickhouse.com/docs/sql-reference/data-types/dynamic) - Like variant, but types don't have to be declared in advance. Shreds a limited number of columns
3. [JSON](https://clickhouse.com/docs/sql-reference/data-types/newjson) - Builds on top of `Dynamic`, with a few specialized features - allowing users to specify known "typed paths", how many dynamic paths and types to support for untyped paths, and some JSON-specific configuration allowing skipping specific JSON paths on insert.
   The full blogpost is worth reading, but Clickhouse's on-disk model roughly mirrors the arrow in-memory format, and they store some metadata outside of the array.

### Others

- Iceberg seems to support the variant type (as described in [this](https://docs.google.com/document/d/1sq70XDiWJ2DemWyA5dVB80gKzwi0CWoM0LOWM7VJVd8/edit?tab=t.0) proposal), but the docs are minimal.
- Datafusion's variant support is being developed [here](https://github.com/datafusion-contrib/datafusion-variant), its unclear to me how much effort is going into it and whether its going to be merged upstream.
- DuckDB doesn't support a variant type. It does have a [Union](https://docs.google.com/document/d/1sq70XDiWJ2DemWyA5dVB80gKzwi0CWoM0LOWM7VJVd8/edit?tab=t.0) type, but its basically a struct. It also seems to have support for Parquet's shredding, but I can't find any docs and seems like PRs are being merged as I'm looking through their issues.
- Databricks supports some specialized [variant functions](https://docs.databricks.com/gcp/en/sql/language-manual/sql-ref-functions-builtin#variant-functions).

## Unresolved Questions

- Do we want a JSON extension type that automatically compresses as variant?
- How do variant expressions operate over different variant encodings?

## Future Possibilities

What natural extensions or follow-on work does this enable? This is a good place to note related ideas that are out of scope for this RFC but worth capturing.
