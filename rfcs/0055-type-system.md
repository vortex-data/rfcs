- Start Date: 2026-03-06
- Authors: @connortsui20
- RFC PR: [vortex-data/rfcs#55](https://github.com/vortex-data/rfcs/pull/55)

# Vortex Type System

## Summary

Vortex separates logical types (`DType`) from physical encodings, but the boundary between them has
been defined informally. This has led to recurring debates, such as whether `FixedSizeBinary<n>`
warrants a new `DType` variant or is merely `FixedSizeList<u8, n>` under a different name. More
fundamentally, we lack a shared vocabulary for reasoning about what makes two types "different" at
the logical level.

This RFC formalizes the Vortex type system by treating physical arrays that decode to the same
logical values as equivalent, and by requiring operations on logical types to respect that
equivalence. It then uses refinement types to establish a decision framework for when new `DType`
variants are justified. A new logical type requires either semantic distinctness (a genuinely
different domain of values or operation contract) or a refinement predicate that gates operations
unavailable on the parent type. If justified, a second step determines whether the type belongs in
core `DType` or as an extension type.

## Overview

Vortex defines a set of logical types, each of which can represent many physical data encodings, and
a set of `Canonical` encodings that represent the different targets that arrays can decompress into.

### Logical vs. Physical Types

A **logical type** (`DType`) describes what operations are available, what laws those operations
obey, and what the data means independent of how it is stored (e.g., `Primitive(I32)`, `Utf8`,
`List(Primitive(I32))`). A **physical encoding** describes how data is laid out in memory or on disk
(e.g., flat buffer, dictionary-encoded, run-end-encoded, bitpacked). Many physical encodings can
represent the same logical type.

Vortex separates these two concepts so that consumers can write code against logical operations,
while new physical encodings can be added without changing that code. Without
this separation, implementing `M` operations across `N` encodings requires `N * M` implementations.
With it, each encoding only needs to decompress itself and each operation only needs to target
decompressed forms, reducing the cost to `N + M`. See this
[blog post](https://spiraldb.com/post/logical-vs-physical-data-types) for more information.

### What is a `Canonical` encoding?

The `N + M` argument relies on a common decompression target that operations are implemented
against. A **canonical encoding** is a physical encoding chosen as the representation for a logical
type (e.g., `VarBinView` for `Utf8`, `ListView` for `List`). The choice is deliberate and optimized
for the common compute case, but not fundamental: nothing in theory privileges `ListView` over
`List` for list data, for example.

### Extension Types

Vortex's built-in set of logical types will not cover every use case. Extension types allow external
consumers to define their own logical types on top of existing `DType`s without modifying the core
type system. See [RFC #0005](https://github.com/vortex-data/rfcs/pull/5) for the full design.

## Motivation

The separation between logical types and physical encodings described above has mostly worked well
for us. However, several recent discussions have revealed that this loose definition may be
insufficient.

For example, we would like to add a `FixedSizeBinary<n>` type, but it is unclear if this is
necessary when it is mostly equivalent to `FixedSizeList<u8, n>`. Are these actually different
logical types? What does a "different" logical type even mean?

Another discussion we have had is if the choice of a canonical `ListView` is better or worse than a
canonical `List` ([vortex#4699](https://github.com/vortex-data/vortex/issues/4699)). Both have the
exact same logical type (same domain of values), but we are stuck choosing a single "canonical"
encoding that we force every array of type `List` to decompress into.

This RFC formalizes the Vortex type system definitions, and this formalization serves as the
foundation for reasoning about questions like these.

## Type Theory Background

This section introduces the type-theoretic concepts that underpin Vortex's `DType` system and its
relationship to physical encodings. To reiterate, most of the maintainers understand these concepts
intuitively, but there is value in mapping these implicit concepts to explicit theory.

Note that this section made heavy use of LLMs to help research and identify terms and definitions,
as the author of this RFC is notably _not_ a type theory expert.

### Equivalence Classes over Physical Arrays

#### In Theory

An **equivalence relation** `~` on a set `S` is a relation that is reflexive (`a ~ a`), symmetric
(`a ~ b` implies `b ~ a`), and transitive (`a ~ b` and `b ~ c` implies `a ~ c`). An equivalence
relation partitions `S` into disjoint subsets called **equivalence classes**, where each class
contains all elements that are equivalent to one another.

A **quotient type** `A / ~` is formed by taking a type `A` and collapsing it by an equivalence
relation `~`. The elements of the quotient type are the equivalence classes themselves: not
individual values, but entire groups of values that are considered "the same."

The critical property of a quotient type is that operations on it must be **well-defined**: they
must produce the same result regardless of which member of the class you operate on. Formally, if
`f : A → B` respects the equivalence relation (`a ~ a'` implies `f(a) = f(a')`), then `f` descends
to a well-defined function on the quotient `f' : A/~ → B`.

#### In Vortex

Consider the set of all physical array representations / encodings in Vortex: a dictionary-encoded
`i32` array, a run-end-encoded `i32` array, a bitpacked `i32` array, a flat Arrow `i32` buffer, etc.

Two physical arrays are logically equivalent if and only if they produce the same logical sequence
of values when decoded / decompressed. This equivalence relation partitions the space of physical
arrays into equivalence classes, where each class corresponds to a single logical column value.

A Vortex `DType` like `Primitive(I32, NonNullable)` identifies the logical type of many such
equivalence classes. It tells us which domain of values and operations we are working with, but says
nothing about which physical encoding is representing any particular array. Thus, the quotient is
best understood as a quotient over physical array values, while `DType` indexes the logical family
those values belong to.

This quotient structure imposes a concrete requirement: any operation defined on arrays of a
`DType` must produce the same logical result regardless of which physical encoding backs the data.

For example, operations like `filter`, `take`, and `scalar_at` all satisfy this: they depend only on
the logical values, not on how those values are stored. However, an operation like "return the
`ends` buffer" is not well-defined on the quotient type as that only exists for run-end encoding.

### Sections and Canonicalization

Observe that every physical array (a specific encoding combined with actual data) maps back to a
`DType`. A run-end-encoded `i32` array maps to `Primitive(I32)`, as does a dictionary-encoded `i32`
array. A `VarBinView` array can map to either `Utf8` or `Binary`, depending on whether its contents
are valid UTF-8. Call this projection `π : Array → DType`.

A **section** is a right-inverse of this projection: a function `s : DType → Array` that injects
each logical type back into the space of physical encodings, such that projecting back recovers the
original `DType` (`π(s(d)) = d`). In other words, a section answers the question: "given a logical
type, which physical encoding should I use to represent it?"

**In Vortex**, the current `to_canonical` function is a section. For each `DType`, it selects
exactly one canonical physical form:

```rust
/// The different logical types in Vortex.
/// Each logical type indexes a family of logical array values.
pub enum DType {
    Null,
    Bool(Nullability),
    Primitive(PType, Nullability),
    Decimal(DecimalDType, Nullability),
    Utf8(Nullability),
    Binary(Nullability),
    List(Arc<DType>, Nullability),
    FixedSizeList(Arc<DType>, u32, Nullability),
    Struct(StructFields, Nullability),
    Extension(ExtDTypeRef),
}

/// We "choose" the set of representations for each of the logical types.
/// This is the image/result of the `to_canonical` function (where `to_canonical` is the section).
pub enum Canonical {
    Null(NullArray),
    Bool(BoolArray),
    Primitive(PrimitiveArray),
    Decimal(DecimalArray),
    VarBinView(VarBinViewArray), // Note that both `Utf8` and `Binary` map to `VarBinView`.
    List(ListViewArray),
    FixedSizeList(FixedSizeListArray),
    Struct(StructArray),
    Extension(ExtensionArray),
}
```

More formally, `Canonical` enumerates the **image** of the section `to_canonical`. Note that the
section is _not_ a bijection between `DType` and `Canonical`: multiple logical types can share the
same canonical form. For example, both `Utf8` and `Binary` canonicalize to `VarBinView`.

This non-bijection is deliberate. If `DType` and `Canonical` were in bijection, the physical type
system would "leak" into the logical types.

For example, if two logically distinct types coincidentally share the same physical layout, a
bijective section would conflict with having both as separate logical `DType`s since "there is no
physical reason for the second." But this reasoning is backwards: logical types are justified by
their _semantics_ (the operations they gate and the refinement predicates they carry), not by
whether they coincidentally share a physical representation.

`Canonical` also represents several arbitrary **choices**. Nothing in the theory privileges
`ListView` over `List` as the canonical representation for variable-length list data. Both are valid
sections (both pick a representation from the same equivalence class), and both satisfy
`π(s(d)) = d`. Even dictionary encoding or run-end encoding are theoretically valid sections. The
fact that we choose flat, uncompressed forms as canonical is a design choice optimized for compute,
not a theoretical requirement.

### The Church-Rosser Property (Confluence)

A rewriting system has the **Church-Rosser property** (or is **confluent**) if, whenever a term can
be reduced in two different ways, both reduction paths can be continued to reach the same final
result (called a **normal form**). For example, the expression `(2 + 3) * (1 + 1)` can be reduced by
evaluating the left subexpression first (`5 * (1 + 1)`) or the right first (`(2 + 3) * 2`), but both
paths arrive at `10`.

**In current Vortex**, `to_canonical` is confluent by construction. Applying `take`, `filter`, or
`scalar_at` before or after canonicalization produces the same logical values. There is one normal
form per `DType`, and every reduction path reaches it.

A **non-confluent** rewriting system is one where two reduction paths from the same starting point
can arrive at different normal forms. Non-confluent systems are well-studied, and the standard
approach is to define **separate reduction relations**, each of which is internally confluent.

For example, instead of one global set of reduction rules, you define two strategies: strategy A
always reduces to normal form X, and strategy B always reduces to normal form Y. Each strategy
satisfies Church-Rosser independently, the only difference is which normal form they target. This is
relevant for future work: if Vortex were to support multiple canonical targets (e.g., both `List`
and `ListView`), each target would define an internally confluent reduction strategy.

### Refinement Types

A **refinement type** `{ x : T | P(x) }` is a type `T` restricted to values satisfying a predicate
`P`. Refinement types express subtypes without changing the underlying representation, instead they
add constraints that gate operations or impose invariants.

For example, in Vortex, `Utf8` is a refinement of `Binary`:

```
Utf8  ~=  { b : Binary | valid_utf8(b) }
```

Every `Utf8` value is a valid `Binary` value, but not every `Binary` value is valid `Utf8`. The
predicate `valid_utf8` is what justifies the separate `DType` variant: it gates string operations
(like substring, regex matching, case conversion) that are not meaningful on arbitrary binary data.
Without this predicate, `Utf8` and `Binary` would be the same type, and maintaining both would be
redundant.

Similarly, `FixedSizeList` is a refinement of `List`:

```
FixedSizeList<T, n>  ~=  { l : List<T> | len(l) = n }
```

The predicate `len(l) = n` constrains the domain of values, and it does gate certain operations:
knowing the size at the type level enables static indexing, fixed-shape tensor operations, and
reshaping without runtime length checks. These two examples (`Utf8` and `FixedSizeList`) illustrate
how refinement predicates can justify new logical types through operation gating, which is central
to the decision framework detailed in the next section.

Ideally, we would like to support refinement types in Vortex, likely as a special kind of extension
type. Examples of this would include `Uuid` as a refinement over `FixedSizeBinary`, and
`NormalizedVector` as a refinement over `Vector`. We would be able to "push down" operations defined
over the refined type when working with the refinement (for example, we could re-use the
`InnerProduct` implementation of `Vector` for `NormalizedVector`). However, it is not obvious what
the best way to express this in Vortex is.

## What justifies a new logical type?

The formalizations above give us a two-step decision framework. The first step decides whether a new
logical type is justified at all, and the second decides whether it should be a core type (a new
variant of `DType`) or an extension type.

**Step 1: Is a new logical type justified?** A new logical type is justified when it is
distinguishable from existing types by one of the following criteria:

1. **Is it semantically distinct?** (Logical-domain criterion.) The values must have a genuinely
   different domain, semantics, or operation contract, not merely a different physical layout.
2. **Does it have a refinement predicate that gates different operations?** (Refinement type
   criterion.) A predicate that restricts the domain of values _and_ enables or disables specific
   operations justifies a new logical type. For example, `valid_utf8` gates string operations that
   are not meaningful on arbitrary binary data, so `Utf8` is a distinct logical type from `Binary`.

If neither criterion applies, the difference is purely physical and belongs in the encoding layer.

**Step 2: Core type or extension type?** Once a new logical type is justified, the question is
whether it belongs in the core `DType` enum or as an extension type (see
[RFC #0005](https://github.com/vortex-data/rfcs/pull/5)).

This must be a pragmatic decision: if the operations gated by the type are owned by core Vortex
(e.g., string kernels for `Utf8`), it should be a first-class `DType`. If the gated operations are
specific to external consumers (e.g., UUID-specific operations), an extension type suffices (see the
comparison to programming language built-in types below).

### Pragmatic Choices: Core Types in Programming Languages

Every programming language must decide which types are built-in primitives and which are
user-defined. This is a universal design decision, and different languages draw the boundary in
different places:

- **OCaml**: `int`, `float`, `char`, `string`, `bool`, `unit`, `list`, `array`, `option`, `ref`, and
  tuples are built-in. Notably, `char` is not a refinement of `int` at the language level even
  though it is represented as one.
- **Haskell**: `Int`, `Integer`, `Float`, `Double`, `Char`, `Bool`, tuples, lists, `Maybe`,
  `Either`, `IO`. `String` is defined as `[Char]` (a type alias, not a built-in), yet it is
  pervasive in the language ecosystem.
- **Rust**: `i8` through `i128`, `u8` through `u128`, `f32`, `f64`, `bool`, `char`, `str`, tuples,
  arrays, slices, and references. Rust distinguishes each integer width as a separate type rather
  than having a single `Integer` type parameterized by bit-width.
- **Agda, Coq, Lean**: Proof assistants based on dependent type theory with essentially no built-in
  data types. `Nat`, `Bool`, `List`, and everything else are defined inductively in standard
  libraries. However, even these systems pragmatically add primitive types for performance: Coq
  added `Int63` and `Float64` as kernel primitives, and Lean has opaque runtime types (`Nat`,
  `UInt8`-`UInt64`, `Float`, `String`) that bypass the inductive definitions. The type theory is
  maximally minimal, but practical implementations end up adding built-in representations anyway.

Several observations are relevant to the Vortex type system:

- Numeric types are universally built-in (even the proof assistants add them back), even though they
  are derivable from more primitive constructs. The justification is performance and ergonomics.
  This parallels Vortex having `Primitive(PType)` as a first-class `DType` rather than encoding
  integers as `List<u8>` with a width constraint.
- String types are almost universally built-in, even though they are refinements of byte sequences.
  Haskell's `String` is `[Char]`, and Rust's `str` is `[u8]` with a UTF-8 invariant. The
  justification is the same as Vortex's `Utf8` over `Binary`: the `valid_utf8` predicate gates
  enough core operations to warrant a dedicated type.
- Rust has separate types for `i8`, `i16`, `i32`, `i64`, and `i128` rather than a single `Integer`
  type with a bit-width refinement. This is analogous to Vortex having separate `PType` variants for
  each primitive width: each width gates different operations (e.g., SIMD lanes, overflow behavior)
  and has different performance characteristics.

The takeaway is that _every_ type system could in principle be collapsed to a minimal core (as the
proof assistants demonstrate), but no practical system does this.

The question of "should X be a core type or a user-defined type?" is always answered by pragmatics:
does the type gate enough core operations and is it central enough to the system's compute model to
justify built-in support? Vortex's decision framework above is how we answer this question for
`DType` variants.

### Should `FixedSizeList` be a type?

We can apply this framework to an existing type, `FixedSizeList`:

**Is it semantically distinct from an existing `DType`?** No. `FixedSizeList` has a different
_physical_ layout (no offsets buffer), but this is a physical difference, not a logical one.
Logically, it is a refinement type over `List` (as shown in the Refinement Types section above).

**Does it gate different query operations?** Yes. Knowing the size at the type level enables
operations like static indexing, fixed-shape tensor reshaping, and compile-time size checks that are
not available on variable-size lists.

By the decision framework, `FixedSizeList` is a justified logical type (it has a refinement
predicate that gates operations).

The remaining question is Step 2: should it be a core `DType` or an extension type? The argument for
a core type (which is what we decided in the past) is that the fixed-size invariant is such a
pervasive feature in types (for example, fixed-shape tensors like vectors and matrices) that it
doesn't make sense to ship `FixedSizeList` as a second-class extension type.

### Should `FixedSizeBinary` be a type?

A similar question applies to `FixedSizeBinary<n>` vs. `FixedSizeList<u8, n>`:

**Is it semantically distinct from an existing `DType`?** This is debatable. `FixedSizeBinary<n>`
and `FixedSizeList<u8, n>` have the same physical layout (a flat buffer of `n`-byte elements), but
semantically a "4-byte opaque blob" (e.g., a UUID prefix or hash) is arguably different from "a list
of 4 individual bytes." Whether this semantic distinction is strong enough to constitute a different
equivalence class is not obvious.

**Does it have a refinement predicate that gates different operations?** This is unclear.
`FixedSizeBinary<n>` signals "opaque binary data" (UUIDs, hashes, IP addresses) whereas
`FixedSizeList<u8, n>` signals "a list of bytes that happens to have a fixed length." One could
argue that `FixedSizeBinary` carries the invariant that individual bytes are not independently
meaningful, which would gate byte-level list operations. However, this invariant is weaker than
something like `valid_utf8`, and it is not obvious that the core Vortex library would ship any
operations gated by it.

Both Step 1 criteria are inconclusive for `FixedSizeBinary`. There is a plausible semantic
distinction and a plausible (but weak) refinement predicate, but neither is clear-cut. If the answer
to either is **yes**, then we move to Step 2: does Vortex core own enough operations gated by this
type to justify a first-class `DType`, or should it be an extension type? This question remains
unresolved. See [Unresolved Questions](#unresolved-questions).

## Prior Art

- **Arrow** has a physical type system, defining both `List` and `ListView` (and
  `LargeList`/`LargeListView`) as separate type IDs in its columnar format. Arrow also distinguishes
  `FixedSizeBinary` from `FixedSizeList<uint8>` at the type level.
- **Parquet** also has a physical type system, with `FIXED_LEN_BYTE_ARRAY` as a distinct primitive
  type separate from repeated groups. This is a nominal distinction similar to the `FixedSizeBinary`
  question.

## Unresolved Questions

- Should `FixedSizeBinary<n>` be a `DType` variant (refinement type) or extension type metadata? See
  the [analysis above](#should-fixedsizebinary-be-a-type) for the case for and against. It is not so
  easy to claim one argument here is better than the other. Comments would be appreciated!
- How should we express refinement types in the Vortex type system?

## Future Possibilities

- We can have extension types be a generalization of the refinement type pattern: for example we
  could enforce user-defined predicates that gate custom operations on existing `DType`s.
- Using the decision framework to audit existing `DType` variants and determine if any should be
  consolidated or split, as well as to make decisions about logical types we want to add (namely
  `FixedSizeBinary` and `Variant`).

## Further Reading

- **Equivalence classes and partitions.**
  [Wikipedia: Equivalence class](https://en.wikipedia.org/wiki/Equivalence_class).
- **Quotient types in type theory.**
  [nLab: quotient type](https://ncatlab.org/nlab/show/quotient+type). Altenkirch, Anberree, Li,
  "Quotient Types for Programmers" ([arXiv:1901.01006](https://arxiv.org/abs/1901.01006)).
- **Sections in category theory.**
  [Wikipedia: Section (category theory)](<https://en.wikipedia.org/wiki/Section_(category_theory)>).
- **Church-Rosser property and confluence.**
  [Wikipedia: Church-Rosser theorem](https://en.wikipedia.org/wiki/Church%E2%80%93Rosser_theorem).
  [Wikipedia: Confluence](<https://en.wikipedia.org/wiki/Confluence_(abstract_rewriting)>). Baader &
  Nipkow, _Term Rewriting and All That_ (Cambridge University Press, 1998).
- **Refinement types.** [Wikipedia: Refinement type](https://en.wikipedia.org/wiki/Refinement_type).
  Rondon, Kawaguci, Jhala, "Liquid Types"
  ([DOI:10.1145/1375581.1375602](https://doi.org/10.1145/1375581.1375602)).
- **Abstract types and existential quantification.** Mitchell & Plotkin, "Abstract Types Have
  Existential Type" ([DOI:10.1145/44501.45065](https://doi.org/10.1145/44501.45065)).
- **Type theory textbook.** Pierce, _Types and Programming Languages_ (MIT Press, 2002). Chapters on
  existential types, subtyping, and type equivalence.
- **Arrow columnar format.**
  [Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html).
