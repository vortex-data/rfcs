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

## The Church-Rosser Property (Confluence)

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
satisfies Church-Rosser independently, the only difference is which normal form they target. 

In Vortex, a similar scenario would be defining multiple strategies of canonicalization, where one
strategy could target `List` as a canonical target (normal form) for list data, and another strategy
could target `ListView`. See the [`List` vs. `ListView`](#list-vs-listview) section for more info.

## Refinement Types and the DType Decision Framework

A **refinement type** `{ x : T | P(x) }` is a type `T` restricted to values satisfying a predicate
`P`. Refinement types express subtypes without changing the underlying representation, instead they
add constraints that gate operations or impose invariants.

For example in Vortex, `Utf8` is a refinement of `Binary`:

```
Utf8  ~=  { b : Binary | valid_utf8(b) }
```

Every `Utf8` value is a valid `Binary` value, but not every `Binary` value is valid `Utf8`. The
predicate `valid_utf8` is what justifies the separate `DType` variant: it gates string operations
(like substring, regex matching, case conversion) that are not meaningful on arbitrary binary data.
Without this predicate, `Utf8` and `Binary` would be the same type, and maintaining both would be
redundant.

This gives us a concrete decision tree for whether a new `DType` variant is justified:

```
                        Does it gate different query operations?
                                    │
                          Yes ──────┼────── No
                           │                │
                     Add to DType     Is it structurally distinct
                   (refinement type)  from an existing DType?
                                            │
                                  Yes ──────┼────── No
                                   │                │
                             Add to DType     Model as canonical form
                          (new structure)     or encoding
```

## Should `FixedSizeBinary` Be a `DType`?

Applying the decision framework above to `FixedSizeBinary<n>` vs. `FixedSizeList<u8, n>`:

### The Case For (Refinement / Nominal Argument)

`FixedSizeBinary<n>` could be justified as a refinement type if it carries semantic distinctions
that `FixedSizeList<u8, n>` does not:

- **Intent signaling.** `FixedSizeBinary<n>` says "this is opaque binary data" (UUIDs, hashes, IP
  addresses), while `FixedSizeList<u8, n>` says "this is a list of bytes that happens to have a
  fixed length."
- **Schema compatibility.** Arrow, Parquet, and other formats distinguish these types. A
  `FixedSizeBinary` `DType` makes round-tripping schemas lossless.
- **Potential invariant.** `FixedSizeBinary` could carry the invariant that elements are not
  individually addressable or meaningful. This is weaker than `valid_utf8` but still semantic.

Under this reading, `FixedSizeBinary` is a lightweight refinement type, and the nominal distinction
earns its place in `DType`.

### The Case Against (Canonical Form / Section Argument)

`FixedSizeBinary` could instead be modeled as a canonical form (section target) of
`FixedSizeList<u8, n>`, or as extension type metadata:

- **No gating predicate.** If no operations are meaningful on `FixedSizeBinary` that are not also
  meaningful on `FixedSizeList<u8, n>`, then the predicate is empty and the refinement is trivial.
- **Leaky abstraction risk.** Every query engine function that handles list types would need to
  additionally handle `FixedSizeBinary`, or we would need coercion rules. If the handling is always
  identical, the `DType` distinction adds complexity without semantic payoff.
- **Schema mapping as metadata.** The information "this came from a Parquet `FixedSizeBinary`
  column" could live in extension type metadata rather than in the core `DType` enum, keeping the
  logical layer minimal.
- **Complexity.** Adding yet another variant to the `DType` enum has a large surface area of change.

Under this reading, `FixedSizeBinary` is an encoding or canonical form, not a logical type.

### Decision

It is somewhat hard to decide which is the right way to go. However, this section provides some more
structure to the discussions we have been holding.

## List vs. ListView

There is also a separate question about whether the current `Canonical` system is the most ideal.
The relationship between `List` and `ListView` ties all of the above concepts together and most
directly motivates a multi-section (multiple normal form) proposal.

`List` and `ListView` represent exactly the same logical data: a sequence of variable-length
sub-arrays. Given an array of type `List(Int32)`, element `i` is a variable-length sequence of
`Int32` values. This is true regardless of the physical layout.

The distinction is entirely in the buffer layout:

- **`List`** stores a single offsets buffer where `offsets[i]..offsets[i+1]` defines the range for
  element `i`. Offsets are monotonically increasing. The child values buffer is contiguous and
  non-overlapping, and every byte belongs to exactly one logical element.
- **`ListView`** stores separate offsets and sizes buffers, where
  `offsets[i]..offsets[i] + sizes[i]` defines the range for element `i`. This allows overlapping
  views (two logical elements can share backing data) and gaps (regions of the values buffer that
  belong to no element).

This is a purely physical distinction, as no query operation can observe the difference.
For example, `scalar_at(i)` will always returns the same list, and `filter`, `take`, and `slice` all
produce logically identical results.

However, this physical distinction has massive performance implications. Converting from `ListView`
to `List` requires rebuilding the entire array to eliminate overlaps and gaps. On the other hand,
converting from `List` to `ListView` is trivial (sizes are just offset deltas).

This asymmetry is notable: the section that targets `List` is more expensive to compute but produces
a form with stronger structural guarantees (no aliasing, no gaps). The section that targets
`ListView` is cheaper and permits aliasing and faster random access.

Many of our consumers (particularly Arrow FFI boundaries and consumers of DataFusion) prefer `List`
because Arrow has is adding support for `ListView` slowly. Other consumers prefer `ListView` because
some operations (namely random access and potentially dependent operations) are faster.

It is highly unfortunate that we cannot canonicalize into different targets, and we are forced to
always decompress into `ListView`. We have the same constraint on `VarBinView` vs `VarBin`, and
while we haven't seen as many performance problems, it feels like a constraint that is too
restrictive in Vortex.

## Design

The proposal is to add a new `to_canonical_target` function that accepts a `CanonicalTarget`
parameter, while keeping the existing `to_canonical` as a convenience that uses the default target.
This minimizes breaking changes.

```rust
/// Canonicalize using the default target. Existing behavior, no breaking change.
fn to_canonical(array: &Array) -> VortexResult<Canonical> {
    to_canonical_target(array, CanonicalTarget::Default)
}

/// Canonicalize into a specific target.
fn to_canonical_target(array: &Array, target: CanonicalTarget) -> VortexResult<Canonical>;
```

Where `CanonicalTarget` selects from a family of valid canonical forms:

```rust
/// A canonicalization target (selects which section to use).
enum CanonicalTarget {
    /// The default canonical forms. This is what the current system uses.
    /// For lists: ListView. For strings: VarBinView.
    Default,
    /// Contiguous canonical forms, motivated by Arrow FFI boundaries and
    /// consumers that need stronger structural guarantees.
    /// For lists: List (monotonic offsets). For strings: VarBin (contiguous data).
    Contiguous,
}
```

The `Canonical` enum would use inner enums for the types where multiple canonical forms exist:

```rust
pub enum CanonicalList {
    List(ListArray),
    ListView(ListViewArray),
}

pub enum CanonicalVarBin {
    VarBin(VarBinArray),
    VarBinView(VarBinViewArray),
}

pub enum Canonical {
    ...
    VarBin(CanonicalVarBin),     // Maps to both Utf8 and Binary DTypes.
    List(CanonicalList),         // Maps to List DType.
    ...
}

array.to_canonical()?.scalar_at(i) == array.to_canonical_target(target)?.scalar_at(i)
```

Each target defines an internally confluent reduction strategy. Within `CanonicalTarget::Default`,
all paths converge to `ListView`/`VarBinView`. Within `CanonicalTarget::Contiguous`, all paths
converge to `List`/`VarBin`. The choice of target is thus made by the consumer (query engine,
FFI boundary, serializer), not by the encoded array.

For `DType`s where the distinction does not apply (e.g., `Primitive`, `Bool`, `Null`), both targets
produce the same canonical form. The parameterization only has an effect where multiple valid
canonical forms exist.

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

- Should `FixedSizeBinary<n>` be a `DType` variant (refinement type) or extension type metadata?
  See the [analysis above](#should-fixedsizebinary-be-a-dtype) for the case for and against.
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
