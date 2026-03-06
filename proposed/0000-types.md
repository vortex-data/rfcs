- Start Date: 2026-03-06
- RFC PR: [vortex-data/rfcs#0000](https://github.com/vortex-data/rfcs/pull/0000)

# Formalize the Vortex Type System

## Summary

One paragraph explanation of the proposed change.

## Motivation

Many of the Vortex maintainers have a good understanding of how the Vortex type system works: we
define a set of logical types, each of which can represent many physical data encodings. We
additionally define a set of `Canonical` encodings that represent the different targets that arrays
can decompress into.

This definition has mostly worked well for us. However, several recent discussions have revealed
that this loose definition may be insufficient.

For example, we would like to add a `FixedSizeBinary<n>` type, but it is unclear if this is
necessary when it is mostly equivalent to `FixedSizeList<u8, n>`. Are these actually different
logical types? What does a "different" logical type even mean?

Another discussion we have had is if the choice of a canonical `ListView` is better or worse than a
canonical `List` ([vortex#4699](https://github.com/vortex-data/vortex/issues/4699)). Both have the
exact same logical type (same domain of values), but we are stuck choosing a single "canonical"
encoding that we force every array of type `List` to decompress into. Is forcing everyone to
decompress into the same physical encoding really what we want?

This RFC makes 2 proposals. The first is a more formalized definition of the Vortex type system, and
this serves to justify the second proposal.

The second proposal is to relax (or extend) the concept of a "canonical" type from choosing a unique
physical encoding for every logical type (a unique normal form) to allowing many possible canonical
targets (multiple normal forms).

# Type Theory Background

This section introduces the type-theoretic concepts that underpin Vortex's `DType` system and its
relationship to physical encodings. To reiterate, most of the maintainers understand these concepts
intuitively, but there is value in mapping these implicit concepts to explicit theory.

Note that this section made heavy use of LLMs to help research and identify terms and definitions,
as the author of this RFC is notably _not_ a type theory expert.

## Equivalence Classes and `DType` as a Quotient Type

### In Theory

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

### In Vortex

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
physical encodings, collapsed by decoded / decompressed equivalence relation.

This quotient structure imposes a concrete requirement: any operation defined on `DType` must
produce the same result regardless of which physical encoding backs the data.

For example, operations like `filter`, `take`, and `scalar_at` all satisfy this: they depend only on
the logical values, not on how those values are stored. However, an operation like "return the
`ends` buffer" is not well-defined on the quotient type as that only exists for run-end encoding.

## Sections and Canonicalization

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
`DType` enum (with the exception of `VarBinView` with `Utf8` and `Binary`):

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
one particular section. The second proposal in this RFC is to allow _multiple sections_.

## Design

Describe the proposed design in enough detail that someone familiar with Vortex could implement it. This should cover:

- New or modified APIs, traits, or vtable entries.
- How this interacts with existing components (encodings, layouts, scan, file format, etc.).
- Key implementation details and corner cases.
- Why is this the best approach in the space of possible designs?
- Which crates are affected and how the dependency graph changes, if at all.

Use code examples and diagrams where they might help, like this:

```rust
pub fn main() {
    let x = f32::to_bits(100.0f32);
    dbg!(x);
}
```

## Compatibility

- Does this change the file format or wire format? Is it backward/forward compatible?
- Does this break any public APIs? If so, what is the migration path?
- Are there performance implications?

If there are no compatibility concerns, briefly state why.

## Drawbacks

- Why should we _not_ do this?
- What is the maintenance cost of this change?
- Does this add complexity that could be avoided?

## Alternatives

- What other designs were considered and why were they rejected?
- What is the cost of **not** doing this?
- Is there a simpler approach that gets us most of the way there?

## Prior Art

How have other systems solved this or similar problems? Consider:

- Other columnar formats (Parquet, Arrow, etc.).
- Database internals (DuckDB, DataFusion, Velox, etc.).
- Relevant academic papers or blog posts.

This section helps frame the design in a broader context. If there is no relevant prior art, that is fine.

## Unresolved Questions

- What parts of the design need to be resolved during the RFC process?
- What is explicitly out of scope for this RFC?
- Are there open questions that can be deferred to implementation?

## Future Possibilities

What natural extensions or follow-on work does this enable? This is a good place to note related ideas that are out of scope for this RFC but worth capturing.

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
