- Start Date: 2026-03-06
- Authors: Connor Tsui
- RFC PR: [vortex-data/rfcs#29](https://github.com/vortex-data/rfcs/pull/29)

# Vortex Type System

## Summary

This RFC formalizes the Vortex type system by grounding `DType` as a quotient type over physical
encodings and establishes a decision framework (based on refinement types) for when new `DType`
variants are justified.

## Overview

Vortex defines a set of logical types, each of which can represent many physical data encodings, and
a set of `Canonical` encodings that represent the different targets that arrays can decompress into.

### Logical vs. Physical Types

A **logical type** (`DType`) describes what the data means, independent of how it is stored (e.g.,
`Primitive(I32)`, `Utf8`, `List(Primitive(I32))`). A **physical encoding** describes how data is
laid out in memory or on disk (e.g., flat buffer, dictionary-encoded, run-end-encoded, bitpacked).
Many physical encodings can represent the same logical type.

Vortex separates these two concepts so that encodings and compute can evolve independently. Without
this separation, implementing `M` operations across `N` encodings requires `N * M` implementations.
With it, each encoding only needs to decompress itself and each operation only needs to target
decompressed forms, reducing the cost to `N + M`. See this
[blog post](https://spiraldb.com/post/logical-vs-physical-data-types) for more information.

### What is a `Canonical` encoding?

The `N + M` argument relies on a common decompression target that operations are implemented
against. A **canonical encoding** is a physical encoding chosen as this representative for a logical
type (e.g., `VarBinView` for `Utf8`, `ListView` for `List`). The choice is deliberate and
optimized for the common compute case, but not fundamental: nothing in theory privileges `ListView`
over `List` for list data, for example.

### Extension Types

Vortex's built-in set of logical types will not cover every use case. Extension types allow external
consumers to define their own logical types on top of existing `DType`s without modifying the core
type system. See [RFC 0005](./0005-extension.md) for the full design.

## Motivation

This definition has mostly worked well for us. However, several recent discussions have revealed
that this loose definition may be insufficient.

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

### Equivalence Classes and `DType` as a Quotient Type

#### In Theory

An **equivalence relation** `~` on a set `S` is a relation that is reflexive (`a ~ a`), symmetric
(`a ~ b` implies `b ~ a`), and transitive (`a ~ b` and `b ~ c` implies `a ~ c`). An equivalence
relation partitions `S` into disjoint subsets called **equivalence classes**, where each class
contains all elements that are equivalent to one another.

A **quotient type** is a data type that falls under the general class of algebraic data types.
Formally, a quotient type `A / ~` is formed by taking a type `A` and collapsing it by an equivalence
relation `~`. The elements of the quotient type are the equivalence classes themselves: not
individual values, but entire groups of values that are considered "the same."

The critical property of a quotient type is that operations on it must be **well-defined**: they
must produce the same result regardless of which member of the class you operate on. Formally, if
`f : A → B` respects the equivalence relation (`a ~ a'` implies `f(a) = f(a')`), then `f` descends
to a well-defined function on the quotient `f' : A/~ → B`.

#### In Vortex

Consider the set of all physical array representations / encodings in Vortex: a dictionary-encoded
`i32` array, a run-end-encoded `i32` array, a bitpacked `i32` array, a flat Arrow `i32` buffer,
etc.

Two physical encodings are logically equivalent if and only if they produce the same logical
sequence of values when decoded / decompressed. This equivalence relation partitions the space of
all physical encodings into equivalence classes, where each class corresponds to a single logical
column of data.

A Vortex `DType` like `Primitive(I32, NonNullable)` **names** one of these equivalence classes. It
tells us what logical data we are working with, but says nothing about which physical encoding is
representing it. Thus, we can say that logical types in Vortex form equivalence classes, and `DType`
is the set of equivalence classes. More formally, `DType` is the quotient type over the space of
physical encodings, collapsed by the decoded / decompressed equivalence relation.

This quotient structure imposes a concrete requirement: any operation defined on `DType` must
produce the same result regardless of which physical encoding backs the data.

For example, operations like `filter`, `take`, and `scalar_at` all satisfy this: they depend only on
the logical values, not on how those values are stored. However, an operation like "return the
`ends` buffer" is not well-defined on the quotient type as that only exists for run-end encoding.

### Sections and Canonicalization

Observe that every physical array (a specific encoding combined with actual data) maps back to a
`DType`. A run-end-encoded `i32` array maps to `Primitive(I32)`, as does a dictionary-encoded `i32`
array. A `VarBinView` array can map to either `Utf8` or `Binary`, depending on whether its contents
are valid UTF-8. Call this projection `π : Array → DType`.

A **section** is a function going the other direction: `s : DType → Encoding`, that picks one
specific physical encoding for each logical type, such that projecting back gives you the original
`DType` (`π(s(d)) = d`). In other words, a section answers the question: "given a logical type,
which physical encoding should I use to represent it?"

**In Vortex**, the current `to_canonical` function is a section. For each `DType`, it selects
exactly one canonical physical form. Observe how the `Canonical` enum is essentially identical to
the `DType` enum (with the exception of `VarBinView` with `Utf8` and `Binary`):

```rust
/// The different logical types in Vortex (the different equivalence classes).
/// This is the quotient type!
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

/// We "choose" the set of representatives of each of the logical types.
/// This is the image/result of the `to_canonical` function (where `to_canonical` is the section).
pub enum Canonical {
    Null(NullArray),
    Bool(BoolArray),
    Primitive(PrimitiveArray),
    Decimal(DecimalArray),
    VarBinView(VarBinViewArray), // Note that `VarBinView` maps to both `Utf8` and `Binary`.
    List(ListViewArray),
    FixedSizeList(FixedSizeListArray),
    Struct(StructArray),
    Extension(ExtensionArray),
}
```

More formally, `Canonical` enumerates the **image** of the section function `to_canonical`.

The critical insight is that `Canonical` represents several arbitrary **choices**. For example,
nothing in the theory privileges `ListView` over `List` as the canonical representative for
variable-length list data. Both are valid sections (since both pick a representative from the same
equivalence class), and both satisfy `π(s(d)) = d`. The current system in Vortex simply hardcodes
one particular section.

Note that even dictionary encoding or run-end encoding are theoretically valid sections (they
satisfy `π(s(d)) = d`). The fact that we choose flat, uncompressed forms as canonical is a design
choice optimized for compute, not a theoretical requirement.

### The Church-Rosser Property (Confluence)

A rewriting system has the **Church-Rosser property** (or is **confluent**) if, whenever a term can
be reduced in two different ways, both reduction paths can be continued to reach the same final
result (called a **normal form**). For example, the expression `(2 + 3) * (1 + 1)` can be reduced
by evaluating the left subexpression first (`5 * (1 + 1)`) or the right first (`(2 + 3) * 2`), but
both paths arrive at `10`.

**In current Vortex**, `to_canonical` is confluent by construction. Applying `take`, `filter`, or
`scalar_at` before or after canonicalization produces the same logical values. There is one normal
form per `DType`, and every reduction path reaches it.

A **non-confluent** rewriting system is one where two reduction paths from the same starting point
can arrive at different normal forms. Non-confluent systems are well-studied, and the standard
approach is to define **separate reduction relations**, each of which is internally confluent.

For example, instead of one global set of reduction rules, you define two strategies: strategy A
always reduces to normal form X, and strategy B always reduces to normal form Y. Each strategy
satisfies Church-Rosser independently, the only difference is which normal form they target. This
is relevant for future work: if Vortex were to support multiple canonical targets (e.g., both `List`
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

## What justifies a new type?

With the formalizations above, we have a framework that gives us a set of questions to guide whether
a new `DType` variant is justified:

1. **Does it gate different query operations?** If yes, does Vortex core own those operations? If
   so, it should be a first-class `DType` (refinement type). If only external consumers need them,
   an extension type suffices (see [RFC 0005](./0005-extension.md)).
2. **Is it structurally distinct from an existing `DType`?** If yes, there may be a case for a new
   `DType` variant. However, this depends on whether the structural difference provides enough
   practical benefit to justify the added complexity.
3. If neither, it should just be a new physical encoding.

These decisions are mostly design choices rather than strict theoretical requirements. For example,
`Utf8` could theoretically be an extension type over `Binary` with a `valid_utf8` predicate. The
main reason it is a first-class `DType` is because we want to have optimized string kernels in the
core Vortex library.

### Should `FixedSizeList` be a type?

We can apply this framework to an existing type, `FixedSizeList`:

**Does it gate different query operations?** No. A fixed-size list is a list with the additional
constraint that every element has the same length `n`, but `scalar_at`, `filter`, `take`, etc. all
behave identically regardless of whether the list is fixed-size.

**Is it structurally distinct from an existing `DType`?** Yes. `FixedSizeList` has a different
physical layout (no offsets buffer, since all elements have the same size). However, this does not
automatically justify a new `DType` variant. A `List` whose offsets are a constant stride would
compress extremely well (a `SequenceArray`), and encodings in Vortex are designed to exploit exactly
this kind of redundancy.

The argument for keeping `FixedSizeList` as its own `DType` is that it makes the fixed-size
invariant explicit at the type level, which simplifies downstream consumers that want to rely on
it (e.g., fixed-shape tensors). The argument against is that it adds a variant to the `DType` enum
that is logically equivalent to `List` with a constraint.

Ultimately, we decided in the past that the structural difference merits its own `DType` variant,
but an argument can be made that it does not warrant one.

### Should `FixedSizeBinary` be a type?

A similar question applies to `FixedSizeBinary<n>` vs. `FixedSizeList<u8, n>`:

**Does it gate different query operations?** This is unclear. `FixedSizeBinary<n>` signals "opaque
binary data" (UUIDs, hashes, IP addresses) whereas `FixedSizeList<u8, n>` signals "a list of bytes
that happens to have a fixed length." One could argue that `FixedSizeBinary` carries the invariant
that individual bytes are not independently meaningful, which would gate byte-level list operations.
However, this invariant is weaker than something like `valid_utf8`, and it is not obvious that the
core Vortex library would ship any operations gated by it.

If the answer is **yes** (it gates operations), then the next question is whether Vortex core owns
those operations. If so, `FixedSizeBinary` is a first-class refinement type. If not, it should be
an extension type.

If the answer is **no** (it does not gate operations), then: **is it structurally distinct from an
existing `DType`?** `FixedSizeBinary<n>` has the same physical layout as `FixedSizeList<u8, n>` (a
flat buffer of `n`-byte elements), so the answer is no. By the decision framework, this means it
should be modeled as a canonical form or extension type metadata rather than a new `DType` variant.

This question remains unresolved. See [Unresolved Questions](#unresolved-questions).

## Prior Art

- **Arrow** has a physical type system, defining both `List` and `ListView` (and
  `LargeList`/`LargeListView`) as separate type IDs in its columnar format. Arrow also
  distinguishes `FixedSizeBinary` from `FixedSizeList<uint8>` at the type level.
- **Parquet** also has a physical type system, with `FIXED_LEN_BYTE_ARRAY` as a distinct primitive
  type separate from repeated groups. This is a nominal distinction similar to the
  `FixedSizeBinary` question.

## Unresolved Questions

- Should `FixedSizeBinary<n>` be a `DType` variant (refinement type) or extension type metadata?
  See the [analysis above](#should-fixedsizebinary-be-a-type) for the case for and against. It is
  not so easy to claim one argument here is better than the other. Comments would be appreciated!

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
  [nLab: quotient type](https://ncatlab.org/nlab/show/quotient+type).
  Altenkirch, Anberree, Li, "Quotient Types for Programmers"
  ([arXiv:1901.01006](https://arxiv.org/abs/1901.01006)).
- **Sections in category theory.**
  [Wikipedia: Section (category theory)](<https://en.wikipedia.org/wiki/Section_(category_theory)>).
- **Church-Rosser property and confluence.**
  [Wikipedia: Church-Rosser theorem](https://en.wikipedia.org/wiki/Church%E2%80%93Rosser_theorem).
  [Wikipedia: Confluence](<https://en.wikipedia.org/wiki/Confluence_(abstract_rewriting)>).
  Baader & Nipkow, _Term Rewriting and All That_ (Cambridge University Press, 1998).
- **Refinement types.**
  [Wikipedia: Refinement type](https://en.wikipedia.org/wiki/Refinement_type).
  Rondon, Kawaguci, Jhala, "Liquid Types"
  ([DOI:10.1145/1375581.1375602](https://doi.org/10.1145/1375581.1375602)).
- **Abstract types and existential quantification.**
  Mitchell & Plotkin, "Abstract Types Have Existential Type"
  ([DOI:10.1145/44501.45065](https://doi.org/10.1145/44501.45065)).
- **Type theory textbook.**
  Pierce, _Types and Programming Languages_ (MIT Press, 2002). Chapters on existential types,
  subtyping, and type equivalence.
- **Arrow columnar format.**
  [Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html).
