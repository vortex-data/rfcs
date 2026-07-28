- Start Date: 2026-07-05
- Authors: @mprammer
- RFC PR: [vortex-data/rfcs#62](https://github.com/vortex-data/rfcs/pull/62)

# Column and File Metadata

## Summary

Vortex has no way to attach identity or annotation to a column. This RFC adds generic, user-defined metadata at two granularities, per-column and file-level, both shaped as an opaque `map<string, [u8]>` whose values the consumer defines and Vortex core does not interpret. Per-column metadata is serialized co-located with each `DType` node but kept out of the in-memory `DType`, so the type stays purely structural and its equality and hashing are unchanged. File-level metadata is a lazily-fetched footer segment.

One mechanism, realized as two small additive wire changes (a `DType`-table field and a `Postscript` segment reference), unblocks three independent needs: stable field identifiers for catalog integrations (Iceberg field IDs, schema evolution), lossless Arrow field-metadata round-tripping, and general file-level sidecar metadata. None of these leaks into the format as a bespoke concept.

## Motivation

Vortex identifies a column only by its name and position. `Struct_` carries `{ names, dtypes, nullable }` and `List` carries `{ element_type, nullable }`, with no per-field metadata slot and no file-level user-metadata segment. Three concrete needs are blocked on this gap:

1. **Catalog field identifiers and schema evolution.** Iceberg and other table formats identify columns by a stable integer field ID rather than by name, which is what makes rename, reorder, add, and drop safe. Without such a slot, a catalog integration must synthesize field IDs and resolve columns by name through a name-mapping fallback, which cannot survive a rename. The need generalizes beyond Iceberg: any catalog or engine that recovers columns by stable identity wants the same thing, namely a per-column identifier carried in the file so that a different reader still recovers the right columns.
2. **Arrow field-metadata round-trip.** Arrow's `Field` carries `custom_metadata`, and the Arrow C Data Interface, Vortex's primary interchange boundary, round-trips it. Vortex's schema conversion (`vortex-array/src/dtype/arrow.rs`, with plugin dispatch in `vortex-array/src/arrow/session.rs`) consumes only the Arrow extension-name key (to reconstruct `Extension` and `Variant`) and drops all other `Field` metadata on import. "Arrow in, Arrow out" should be lossless for field metadata, and today it is not.
3. **File-level sidecar metadata.** Writers want to attach file-wide metadata, such as the originating Arrow schema, provenance, or engine hints, without a bespoke side channel.

Absent a first-class mechanism, each consumer invents its own workaround: name-mapping alone, a separate id sidecar, or a side copy of the Arrow schema. This is the per-integration divergence a format should prevent, and it blocks catalog integrations today.

## Design

One mechanism, two granularities, both opaque:

> **Per-column metadata** and **file-level metadata**, each a `map<string, [u8]>`. Keys are UTF-8 strings; values are opaque bytes the _consumer_ defines. Vortex core stores and round-trips them and interprets nothing.

### Per-column metadata: co-located on the wire, separate in memory

On the wire, an optional metadata map attaches to each `DType` node. Because it rides on the node, it is available at every position that can carry an identifier: struct fields, `List` elements, `FixedSizeList` elements, and, once a map dtype lands, map key and value fields. That placement is the load-bearing property: the identifier is serialized with the field it identifies and addressed structurally, so it survives rename and reorder, which is the entire reason field IDs exist.

FlatBuffers (illustrative), an optional field appended to the existing `DType` table:

```
table KeyValue { key: string (required); value: [ubyte]; }

table DType {
  // existing union / fields ...
  metadata: [KeyValue];   // NEW: optional, per-node, absent when unset
}
```

There is direct in-format precedent: `Extension` already carries a `metadata: [ubyte]` blob in the dtype schema, so metadata riding in the serialized dtype is not a new concept for Vortex.

In memory, metadata is not a member of `DType`. Vortex's in-memory `DType` is a hand-written enum, distinct from the generated FlatBuffers `DType` table, so the reader routes the table's `metadata` field into a separate metadata carrier rather than into the enum. This is what leaves the enum, and its `Eq` and `Hash`, untouched.

The metadata carrier is a schema-level structure parallel to the type tree, addressed by a structural pre-order index over the tree's nodes. This mirrors how Arrow keeps annotations on `Field` and `Schema` and leaves `DataType` structural. Vortex has no `Schema` or `Field` layer today, since a file's schema is its root `DType`, so this introduces a thin schema wrapper or an equivalent index-keyed side table (the exact shape is an implementation choice, see Unresolved Questions). On read, the traversal that parses the `DType` tree populates the carrier by position; on write, the same traversal emits both. Every schema-mutating operation, such as projection or reorder, keeps the two in lockstep. This choice is the crux of the design:

> **`DType` structural equality and hashing are unchanged. Metadata does not participate in type identity.**

`DType` is `Eq + Hash`, and its equality gates load-bearing invariants across the codebase: `ChunkedArray` requires all chunks to share a dtype, compute kernels assert that input and output dtypes match, and `StructArray` validates field dtypes. Making metadata a member of `DType` would force a bad choice. Either it participates in equality, in which case two structurally identical schemas that differ only in `iceberg.id` become unequal, breaking chunking and kernel guards. Or it is excluded from equality, in which case `DType` becomes the sole in-tree annotation outside `Eq`, requiring a hand-written `Hash` to match the existing manual `PartialEq` and a wrapper around the `DType` enum that churns every match site. Keeping metadata out of the in-memory `DType` avoids both. The type stays purely structural; no compute, scan, or chunk code changes. Metadata lives where its consumers use it, at the schema and file boundary (Iceberg scan planning, the Arrow import and export bridge, the file sidecar), rather than in the compute hot path.

Because metadata is not in `DType`, the bare `DType` FlatBuffers serializer (`WriteFlatBuffer for DType`) cannot see it. The writer that owns the carrier passes it alongside, and the schema-boundary serialization emits the dtype and its metadata together. One consequence is worth stating plainly: metadata round-trip is guaranteed only where a carrier is threaded, namely the file's root schema and the Arrow import and export bridge. Bare-`&DType` paths that carry no schema context, such as vortex-ipc's per-array schema messages, drop metadata unless they are extended to thread the carrier; that extension is out of scope here and listed as future work. Since arrays carry a bare `DType`, the carrier is also the first step toward a schema-with-annotations type, if one is later wanted.

### File-level metadata: a lazily-fetched footer segment

File-level metadata is a `map<string, [u8]>` stored as its own segment, referenced by a small `PostscriptSegment` descriptor. This reuses the postscript-referenced-segment structure that `dtype` and `statistics` already use, with one deliberate difference: the current reader folds those two segments into the open-time footer read window and parses them eagerly (`FooterDeserializer`), whereas the metadata segment is excluded from that window and fetched only on demand. The postscript is an evolvable FlatBuffers `table`, so adding one more optional segment reference costs a vtable entry plus a roughly 30-byte descriptor. The metadata values live in the referenced segment rather than in the postscript, so the addition stays well within the postscript size cap (`MAX_POSTSCRIPT_SIZE`, `u16::MAX − 8`). Readers that never touch file metadata pay zero extra I/O.

### Conventions layered on top

Vortex core defines the mechanism; consumers own reserved key namespaces:

- **Iceberg**, modeled on the accepted ORC approach, uses `iceberg.id` as a per-column key. ORC did not add a field-id concept to its format; it added generic type attributes, and Iceberg layered `iceberg.*` on top. We do the same. Vortex carries `nullable` natively, so ORC's companion `iceberg.required` attribute is unnecessary here.
- **Arrow** import maps `Field` `custom_metadata` into per-column metadata, and export maps it back, closing the current drop. One asymmetry applies: arrow-rs `Field::metadata()` is `HashMap<String, String>`, so `Arrow → Vortex → Arrow` is lossless because strings are a subset of bytes, but a Vortex-native non-UTF-8 value has no home on the Arrow boundary. Export must specify lossy-or-error semantics for such values (see Unresolved Questions).
- **Other catalogs and engines** use their own namespaces with no core change.

### Reader semantics

- **Absent.** Existing behavior, such as Iceberg's name-mapping fallback. Old files and non-annotating writers keep working.
- **Partial.** Per-field fallback, matching ORC: a field with `iceberg.id` uses it, and one without falls back to name-mapping.
- **Conflicting.** The core cannot detect a malformed or duplicate id, because values are opaque. Duplicate-`iceberg.id` detection is the consumer's responsibility at read time, not only at write time.

### Worked example

Schema `struct { id: i64, name: utf8, tags: list<utf8> }` annotated with Iceberg field IDs (illustrative; not Vortex's `Display` output):

```
Struct                             names    = ["id", "name", "tags"]
├─ id   : Primitive(I64)           metadata = { "iceberg.id": 1 }
├─ name : Utf8                     metadata = { "iceberg.id": 2 }
└─ tags : List                     metadata = { "iceberg.id": 3 }
   └─ element : Utf8               metadata = { "iceberg.id": 4 }   ← only a per-node carrier reaches the list element
```

Rename `name` to `full_name`: `names[1]` changes, but the node still carries `iceberg.id = 2`, so a reader resolves the same column. The list-element id (`4`) is unreachable by any per-struct scheme, which is the decisive argument for a per-node carrier over a `Struct_`-parallel array (see Alternative 4).

### Design properties

- **Generic.** One primitive serves catalog field IDs, Arrow round-trip, file sidecar, and future consumers, with no further format churn. ORC and Arrow both chose generic attributes for this reason.
- **Stable under schema evolution.** Per-column, the id is serialized at the node and addressed by structure rather than by name or path, so it survives rename and reorder and reaches nested elements.
- **Identity-preserving.** `DType` equality and hashing stay untouched, so there are no compute, scan, or chunk regressions and no risk of annotation leaking into type comparisons.
- **Additive and minimal.** It is an optional FlatBuffers field plus one optional footer segment, forward and backward compatible (see Compatibility).
- **Precedented.** ORC type attributes, Arrow `custom_metadata`, Parquet key-value plus schema field-ids, and Vortex's own `Extension.metadata` and postscript-referenced `dtype` and `statistics` segments all take this shape.
- **Opaque bytes.** Consumers pick integer, string, or binary encodings, and the core stays out of it.

### Crates affected

- `vortex-array`: the `src/dtype` schema, the Arrow import and export bridge, and the schema-level metadata carrier.
- `vortex-flatbuffers`: the `DType` table and the footer `.fbs`.
- `vortex-proto`: the parallel protobuf DType serialization must gain the same field, or metadata silently drops across it.
- `vortex-file`: the footer segment, read and write.
- The bindings that expose schema metadata.

`vortex-ipc` serializes a bare `DType` for per-array schema messages and is out of scope for metadata round-trip in this RFC (see Design). No change to encodings, to the in-memory `DType` enum, or to the scan hot path.

## Compatibility

This changes the wire format. It is designed to be backward- and forward-compatible, consistent with how Vortex adds optional schema fields.

- **Both wire changes are additive optional FlatBuffers fields**: per-column metadata appends a field to the `DType` table, and file-level metadata appends a `PostscriptSegment` reference to the `Postscript` table. Appended optional table fields are skippable by old readers by construction: old readers ignore both, and new readers read old files with the metadata absent. No file format version change is needed; the file's `VERSION` stays `1`, and an old reader's exact-match version check still accepts the file. This is distinct from adding a `union Type` member, which an old reader cannot skip.
- **The parallel `vortex-proto` DType serialization must be updated in lockstep**, or metadata silently drops wherever that path is used. This is a real compat hole, called out here so it is not missed.
- **In-memory `DType` identity is unchanged**, since metadata is not a `DType` member. Chunking, kernel dtype guards, and scan conformance are unaffected, with no dual codepath in the compute layer.
- **Backward compatibility is already covered; forward compatibility is the risk.** `vortex-test/compat-gen` decodes every release's fixtures against the current reader (new reader over old files, the backward direction), and an additive change cannot break it. The forward direction, old released readers over new metadata-bearing files, is safe by FlatBuffers unknown-field skipping but warrants an explicit test that runs released reader binaries against new fixtures. Separately, an old-version tool that rewrites a file drops `iceberg.id` and silently loses column identity; this rewrite-strips-metadata case is a known failure mode in Parquet tooling and should be documented as a consumer hazard.
- **No public-API break.** The schema APIs gain optional metadata accessors.
- **Statistics interaction.** `FileStatistics.field_stats` is positional over root-schema fields, so a field-ID-based projection resolver must still map id to position to reach stats. This is orthogonal to the metadata carrier, but worth stating.
- **Performance.** Per-column metadata is schema-level, absent when unset (leaf types pay nothing), and read with the schema. The file-level segment is lazily fetched and excluded from the initial footer read window, so callers that do not want it pay zero extra I/O. There is no scan-hot-path impact.
- **Testing.** Extend `compat-gen` with fixtures both with and without metadata, add an Arrow field-metadata round-trip test (including a non-UTF-8 value to pin the export semantics), and add the released-reader-on-new-fixture check above.

## Drawbacks

- It is a wire-format change: it touches every binding that reads and writes schema, and both the FlatBuffers and protobuf dtype serializations. The file format version is unchanged, since the change is additive (see Compatibility).
- Two granularities mean two code paths: per-column co-located with the dtype tree, and file-level in a segment.
- The in-memory metadata carrier must stay in sync with the `DType` tree across every construction and traversal that builds a schema, a discipline the single-tree alternative would not need.
- Opaque bytes push validation to consumers. The core will not catch a malformed or duplicate `iceberg.id`, and a convention is only as good as its writers.

## Alternatives

1. **File-level metadata only** (postscript-referenced keyed segments; an earlier file-metadata-segments proposal, public PR [#7954](https://github.com/vortex-data/vortex/pull/7954), took this shape with eagerly-read segments).
   - _Pros:_ simplest; no `DType` change.
   - _Cons:_ to attach an id to a column it needs a field-path-to-id addressing scheme, which is fragile under rename and reorder (the very problem field IDs exist to solve), and it duplicates the schema's structure in a second place that can drift. It is awkward for nested types. Rejected as the sole mechanism, adopted as the file-level half, though lazily fetched rather than eager as in #7954.
2. **Per-column metadata only** (no file-level).
   - _Cons:_ no home for genuinely file-wide metadata, such as a full Arrow schema round-trip or provenance. Insufficient alone; we want both.
3. **Metadata as an in-memory member of `DType`** (participating in, or excluded from, structural equality).
   - _Pros:_ single tree; no parallel carrier to keep in sync; consistent with `nullable` and `names` living inside the dtype.
   - _Cons:_ if it participates in `Eq`, two structurally identical schemas differing only in `iceberg.id` become unequal, breaking `ChunkedArray`, kernel, and `StructArray` dtype invariants. If it is excluded, `DType` becomes the only in-tree annotation outside equality, forcing a hand-written `Hash` (to match the existing manual `PartialEq`) and a wrapper around the `DType` enum that churns every match site. Rejected in favor of wire-co-located, memory-separate, which keeps identity clean at lower blast radius.
4. **A `Struct_`-parallel `field_metadata` array** (instead of per-`DType`-node), mirroring how `names` is carried.
   - _Pros:_ keeps name and metadata adjacent on the parent, like an Arrow `Field`.
   - _Cons:_ does not reach `List` or `FixedSizeList` elements or future map key and value fields without adding a separate slot to each, so the worked example's list-element id (`4`) is unreachable. Rejected, because the per-node carrier is the only uniform shape.
5. **An Iceberg-specific typed `field_id: int` on `Struct_`.**
   - _Pros:_ smallest change; typed and self-validating.
   - _Cons:_ single-purpose, offering nothing for Arrow round-trip, other catalogs, or engine sidecar data, so the format churns again the next time a consumer needs to annotate a column. Precedent is genuinely split (Parquet did add a typed `field_id` to its Thrift `SchemaElement`, while ORC and Arrow chose generic attributes), so this is a real option, rejected on genericity rather than on precedent.
6. **String values (`map<string,string>`, ORC/Arrow-style) instead of bytes.**
   - _Pros:_ human-readable; byte-identical to ORC and Arrow; round-trips cleanly out to the Arrow boundary, with no non-UTF-8 case.
   - _Cons:_ forces every consumer to string-encode inherently-binary metadata, whereas `[u8]` is strictly more general. Chosen: bytes, with UTF-8 strings the expected common case and specified Arrow-export semantics for the non-UTF-8 tail.
7. **Represent metadata as a Vortex array of `map` dtype.**
   - _Cons:_ Vortex has no `map` dtype yet (`DataType::Map` is unsupported on import today), it is heavier machinery for a handful of small values, and it is chicken-and-egg. Deferred: because the file-level segment's bytes are opaque, it can become a Vortex array later with no format change. Captured in Future Possibilities.
8. **Do nothing, and let each consumer keep an out-of-band sidecar.**
   - _Cons:_ every integration invents its own scheme, giving per-integration divergence and metadata unreadable across tools. It leaves the Iceberg field-ID gap and other catalog integrations blocked. The cost of not doing this is concrete: catalog integrations blocked on a missing primitive.

## Prior Art

| System            | Where metadata lives                    | Value type                                                | Field-id convention                                   | Fallback        |
| ----------------- | --------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------- | --------------- |
| ORC               | generic per-type **attributes**         | `string → string`                                         | `iceberg.id` / `iceberg.required` (Iceberg layers on) | name / position |
| Arrow             | `Field.custom_metadata`                 | `string → string`                                         | `PARQUET:field_id` (the Arrow↔Parquet **bridge** key) | —               |
| Parquet           | schema `SchemaElement` + file key-value | typed `field_id: i32` (schema) + `string → string` (file) | native Thrift `field_id`                              | name            |
| Vortex (this RFC) | per-`DType`-node (wire) + file segment  | `string → [u8]`                                           | `iceberg.id` (convention)                             | name-mapping    |

- **ORC** is the established precedent for exactly this problem: generic attributes, `iceberg.*` layered on, and the schema reconstructed from them.
- **Arrow** `Field.custom_metadata` is the direct model for the per-column carrier and for the round-trip Vortex currently drops. Iceberg writes Parquet field IDs into the Thrift `field_id` directly; `PARQUET:field_id` is the Arrow-to-Parquet bridge metadata key, not Iceberg's own mechanism.
- **Vortex** itself already carries a metadata blob in the dtype via `Extension.metadata`, and `dtype` and `statistics` are already optional, postscript-referenced footer segments, which is in-format precedent for both halves.
- Prior Vortex work on **file-level metadata segments** (public PR [#7954](https://github.com/vortex-data/vortex/pull/7954)) explored the file-level half with eager, postscript-referenced segments.

References: [Arrow columnar format, schema and field metadata](https://arrow.apache.org/docs/format/Columnar.html), [Apache Parquet file-format metadata](https://parquet.apache.org/docs/file-format/metadata/), [Apache ORC specification](https://orc.apache.org/specification/), [Apache Iceberg table spec, field IDs and name mapping](https://iceberg.apache.org/spec/).

## Unresolved Questions

- **Concrete in-memory carrier shape.** A thin `Schema { dtype, metadata }` wrapper versus an index-keyed side table on the existing schema-holding types. Both keep `DType` structural; the choice is an implementation detail deferred to the PR.
- **Interaction with lazy `DType` views.** Struct fields deserialize lazily (`FieldDType` / `ViewedDType`), so an eagerly-populated carrier would force a full flatbuffer traversal at open and change the wide-schema cost profile. The carrier may need to be lazy as well.
- **File-level segment granularity.** A single metadata blob versus multiple independently-fetchable keyed segments, trading round-trip cost against all-or-nothing download.
- **Arrow-export semantics for non-UTF-8 values.** Error, lossless-encode (for example base64) into the string map, or drop with a warning? This determines whether `Vortex → Arrow` is total.
- **Reserved-namespace documentation.** Do we register and document conventions (`iceberg.*`, `arrow.*`) in the spec, or keep the carrier fully opaque and let consumers own their namespaces?

## Future Possibilities

- **File-level metadata as a Vortex `map`-dtype array** once that type lands, with no format change needed, since the segment's bytes are already opaque.
- **A cross-implementation conformance corpus** that exercises metadata round-trip (field IDs, Arrow metadata) across every Vortex reader and writer.
- **Additional catalog and engine field-ID conventions** (Delta, others) riding the same mechanism with new namespaces, with no format change.
- **Column-level statistics or encoding hints** reusing the same per-column carrier if ever desired.
- **Threading the metadata carrier through `vortex-ipc`** so per-array schema messages round-trip metadata, closing the bare-`&DType` gap noted in Design.
