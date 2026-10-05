- Start Date: 2026-09-24
- Authors: @gatesn
- RFC PR: TBD (renumber this file to match the PR)

# DTypes, Extensions, and Execution Layouts

## Summary

Keep Vortex's core `DType` variants for common value families, and add
`FixedSizeBinary(n)` as a core type. Represent UTF-8, decimal, fixed-size list,
and map as well-known extensions. A UUID extension uses `FixedSizeBinary(16)`
as its portable storage type.

Migrating the four existing variants changes the Rust API, not their dtype wire
format: readers and writers keep their existing tags. FixedSizeBinary is a new
type and needs a wire representation. It canonicalizes to contiguous fixed-width
bytes, not binary views. Other encodings, including 128-bit lanes for UUIDs,
can execute without first producing that canonical form.

## Motivation

We call Vortex dtypes _logical_ and its encodings _physical_. Those words
describe a boundary, not an absolute classification:

```text
engine logical type -> engine storage type / Vortex DType -> Vortex encoding
```

An engine may store timestamps as `i64` values. To the engine, `i64` is storage;
to Vortex, it is a value type that can be represented by primitive buffers, bit
packing, or a datetime-parts encoding. A timestamp extension adds the unit and
temporal meaning to that integer. Likewise, UTF-8 adds a text guarantee to
binary values without choosing between VarBin, VarBinView, and FSST.

Exposing that meaning to Vortex serves two purposes. Other engines can recognize
the same type, and the compressor can use its parameters to consider
domain-specific strategies such as datetime parts. An engine can instead write
storage types and preserve its logical schema separately, as
[Arrow's Parquet integration](https://arrow.apache.org/docs/cpp/parquet.html#roundtripping-arrow-types)
does with `ARROW:schema`. Vortex only sees that domain information if the engine
passes it through a type or an explicit compression hint.

The current enum makes engine-level meanings look like additions to Vortex's
fundamental value families. FixedSizeBinary belongs among those families: its
exact byte width is a value guarantee that multiple engines can use. Its
efficient memory layout is a separate decision.

## Design

### Core types and extensions

The proposed core `DType` variants are:

```text
Null | Bool | Primitive | Binary | FixedSizeBinary(n)
     | List | Struct | Union | Variant | Extension
```

`FixedSizeBinary(n)` is a new core variant, not an extension over `Binary`.
`Binary` admits byte sequences of any length; `FixedSizeBinary(n)` admits exactly
`n` bytes per non-null value. Nullability applies to the whole value. Both have
byte access, equality, and hashing, but only `FixedSizeBinary(n)` promises a width.
Converting between them is explicit, with length validation when constructing
`FixedSizeBinary(n)` from `Binary`.

Move these Rust variants to well-known extensions:

| Former variant  | Extension                | Portable storage           | Contract                                        |
| --------------- | ------------------------ | -------------------------- | ----------------------------------------------- |
| `Utf8`          | `vortex.utf8`            | `Binary`                   | Valid UTF-8 bytes                               |
| `Decimal`       | `vortex.decimal`         | Signed integer             | Precision and scale                             |
| `FixedSizeList` | `vortex.fixed_size_list` | `List(T)`                  | Exactly `n` elements of type `T`                |
| `Map`           | `vortex.map`             | `List(Struct{key, value})` | Key/value types and the existing map guarantees |

These IDs are proposed names. Existing temporal extensions retain their
identities. `Union` stays a core sum type because inactive alternatives and
outer nulls do not have ordinary struct semantics. `Variant` stays pending a
separate proposal.

Each known extension has one specified portable storage type, determined by its
identity and parameters. A UUID extension uses `FixedSizeBinary(16)` with the
canonical UUID byte order. The extension's storage mapping does not require a
particular array encoding. A 128-bit lane representation can implement UUID
equality directly; numeric lane ordering is not automatically UUID ordering.

Known extension identity is its ID and semantic parameters, including nested
type arguments and nullability. Its portable storage type is fixed by that
definition. Unknown extensions retain their complete received descriptors,
including storage, because Vortex cannot infer whether two foreign descriptors
have the same meaning. A chosen array encoding never participates in type
equality.

### Execution layouts

A portable storage type gives a reader a common way to understand values. It
must not force all arrays through one in-memory layout. A UUID array, for example,
may use 128-bit lanes and run a suitable kernel directly.

Keep a default canonical form for callers that need a simple uncompressed array.
Also allow a kernel or exporter to request an acceptable layout for its input.
For byte sequences, the useful layouts are contiguous fixed-width slots, offsets
with a byte buffer, and byte views. Preparation should use an existing
acceptable layout when possible and convert only when necessary. The array keeps
its declared dtype through preparation.

The canonical form for FixedSizeBinary is contiguous fixed-width slots plus
whole-value validity. Routing dense 16-byte values through VarBinView would add
per-value descriptors and, for out-of-line values, a separate payload. An
ordinary `Binary` array whose values happen to have the same length may use the
same layout without acquiring a type-level length guarantee. Keep VarBinView as
the default for variable-length Binary initially: its inline values and prefixes
can make comparisons and row selection cheap. They are useful for binary keys as
well as text. Offset-based storage remains useful for contiguous scans and for
interoperation with systems that use offsets.

This requires changing the current relationship between `DType` and `Canonical`.
Today Binary and UTF-8 canonicalize to VarBinView, and the default binary
builder produces views. FixedSizeBinary needs its own canonical form, while
kernels may request another supported layout. A small set of requested layouts
and preparation rules is sufficient for this RFC; a general query planner is
not required. Primitive arrays remain direct typed lanes; sharing a bit width
with FixedSizeBinary does not require an extension wrapper around byte storage.

### Operations and validation

An extension does not inherit storage comparison, hashing, arithmetic, or
parsing by default. A function may use a storage kernel only when that kernel
implements the function's bound semantics. Case-insensitive text, for example,
cannot use raw byte equality and hashing. UTF-8 LIKE may consume validated byte
views, but the bound call still has UTF-8 arguments.

FixedSizeBinary equality and hashing compare its bytes. A UUID function may
delegate to them when its bound contract agrees. Any UUID ordering must be
specified separately before a byte or integer kernel can implement it.

Constructors and imports establish type guarantees before kernels rely on them.
That includes UTF-8 validity, fixed byte widths, list lengths, and decimal
precision. Structural operations such as slice and take preserve the type when
they preserve its guarantees. A compressed representation can stay compressed
until a consumer needs a particular layout.

Moving Decimal requires integer storage capable of its currently supported
precisions, including wide lanes. Its existing wire Decimal tag must continue to
work; adding an internal lane type must not change old files into new primitive
wire tags. Keep the existing Decimal implementation until an extension-backed
path has the same coverage.

## Compatibility

The Rust change breaks pattern matches and constructors using `DType::Utf8`,
`DType::Decimal`, `DType::FixedSizeList`, and `DType::Map`. Replace them with
constructors and typed extension accessors. Array builders, scalars, Arrow
conversion, and engine adapters must use those accessors rather than matching
the removed variants.

The dtype wire schema is separate from the Rust enum. Keep the existing
FlatBuffers and Protocol Buffers tags and fields for these four types. New
readers translate those tags into the corresponding extensions; new writers
translate the extensions back to the existing tags. Apply the translation
recursively to nested types and preserve nullability and parameters. Existing
array encodings retain their wire contracts. Both old-file reading and new-file
reading by old readers must be demonstrated for the migrated types.

A generic wire `Extension` descriptor is not a substitute for an old dedicated
tag when writing one of these well-known types. FixedSizeBinary needs a new dtype
wire representation and a compatibility decision for old readers. New encodings
or serialized functions also need separate decisions; this RFC does not claim
that unrelated additions are readable by old software.

## Implementation

1. Add `FixedSizeBinary(n)` to the core type, scalar, builder, and dtype wire
   interfaces, with dense fixed-width canonical storage and whole-value validity.
2. Define the well-known extensions, their validation, operations, and portable
   storage mappings. Establish wide integer support before moving Decimal.
3. Add typed constructors and accessors, then migrate callers of the four enum
   variants.
4. Add reader and writer translations for the existing dtype tags. Verify nested
   schemas, scalars, arrays, and both read directions with historical files.
5. Let byte kernels accept appropriate layouts. Demonstrate UUID equality over
   fixed-width bytes and 128-bit lanes without a VarBinView conversion; compare
   fixed-width, offset, and view layouts before choosing broader defaults.
6. Remove the four Rust variants once their extension implementations and
   compatibility paths cover the existing behavior.

## Drawbacks

The extension mechanism becomes responsible for types that core handles directly
today, while FixedSizeBinary adds a core variant and canonical layout. Builders
and kernels need typed dispatch without repeated registry work inside row loops.
Supporting several byte layouts adds preparation paths and kernels; their benefit
must justify their cost. The wire adapters remain after the Rust variants disappear.

## Alternatives

Making FixedSizeBinary an extension over `Binary` saves a core variant, but makes
the fixed-width canonical form an exception to Binary's variable-width default.
Making every primitive an extension over fixed-size bytes is coherent as a type
definition model, but would not justify wrapping typed numeric lanes in byte
arrays during execution. Giving each extension exactly one canonical execution
layout is simpler, but makes a portable storage choice dictate performance for
every workload.

## Unresolved Questions

- Which byte layouts should the first fixed-width and variable-width kernels
  accept directly, and what preparation cost is acceptable? This needs
  measurement.
- Should the requested-layout mechanism extend beyond bytes in this RFC, or
  should other type families adopt it as needed?
- Which existing map guarantees are semantic type parameters, and which describe
  a particular representation? Preserve the current contract during migration
  and settle any change separately.
