# Theoretical Background: Logical and Physical Types in Vortex

## Purpose

This document provides the type-theoretic foundations for reasoning about Vortex's `DType` system, its relationship to physical encodings, and the design space for canonicalization. It is intended as background material for an RFC on evolving how the Vortex type system handles canonical representations — and on whether additions like `FixedSizeBinary` belong in `DType`.

---

## 1. Core Concepts

### 1.1 Equivalence Classes

An **equivalence class** is the set of all elements that are "the same" under some equivalence relation. An equivalence relation `~` on a set `S` must be reflexive (`a ~ a`), symmetric (`a ~ b ⟹ b ~ a`), and transitive (`a ~ b` and `b ~ c` ⟹ `a ~ c`).

**In Vortex:** Consider the set of all physical array representations — dictionary-encoded int32 arrays, run-length-encoded int32 arrays, bitpacked int32 arrays, flat Arrow int32 arrays, etc. Two representations are equivalent if and only if they produce the same logical sequence of values when decoded. This equivalence relation partitions the space of all physical arrays into equivalence classes, where each class corresponds to a single logical column of data.

The key insight is that a Vortex `DType` like `Primitive(I32, NonNullable)` _names_ one of these equivalence classes. It tells you what logical data you're working with, but says nothing about which physical representative you hold.

**Further reading:**

- Partition of a set / equivalence class: [https://en.wikipedia.org/wiki/Equivalence_class](https://en.wikipedia.org/wiki/Equivalence_class)

### 1.2 Quotient Types

A **quotient type** `A / ~` is a type formed by taking a base type `A` and collapsing it by an equivalence relation `~`. Elements of the quotient type are the equivalence classes themselves. Operations on the quotient type must be _well-defined_ — they can't depend on which representative you pick from within a class.

Formally, if `f : A → B` is a function that respects the equivalence relation (`a ~ a' ⟹ f(a) = f(a')`), then `f` descends to a well-defined function on the quotient `f' : A/~ → B`.

**In Vortex:** `DType` is a quotient type over the space of physical representations. The equivalence relation is decode-equality ("these two arrays decode to the same logical values"). Query operations like `filter`, `take`, `scalar_at` are well-defined on the quotient: their results depend only on the logical data, not on the encoding. This is exactly the requirement for a function to descend to the quotient.

An operation that _doesn't_ descend to the quotient — say, "return the offset buffer" — is one that depends on the specific representative (ListView has offset-size pairs; List has monotonic offsets; dictionary encoding has neither). Such operations are physical-layer concerns and must live below the DType abstraction.

**Further reading:**

- Quotient types in type theory: [https://ncatlab.org/nlab/show/quotient+type](https://ncatlab.org/nlab/show/quotient+type)
- Quotient types in programming languages (Altenkirch, Anberree, Li): [https://arxiv.org/abs/1901.01006](https://arxiv.org/abs/1901.01006)

### 1.3 Sections

Given a surjection (a function that maps from a detailed representation to a collapsed one), a **section** is a right inverse: a function that picks one representative from each equivalence class.

Concretely, if `π : Encoding → DType` is the projection that sends each physical encoding to its logical type, a section is a function `s : DType → Encoding` such that `π(s(d)) = d` for all DTypes `d`. In other words, `s` picks a specific physical representation for each logical type, and that representation actually has the right logical type.

A section is also sometimes called a **choice function** or a **splitting** of the projection.

**In Vortex:** The current `to_canonical` function is a section. It calls `execute` on a lazy array tree that defines the execution plan to reach a canonical form. For each `DType`, it selects exactly one physical form:

| DType               | Current canonical form (section)                                                         |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `Primitive(I32, _)` | Flat Arrow-compatible buffer                                                             |
| `Utf8(_)`           | VarBin with UTF-8 bytes                                                                  |
| `List(T, _)`        | ListView (offset-size pairs)                                                             |
| `Struct(fields, _)` | Struct with canonicalized top-level layout (children remain in their existing encodings) |

Note that canonicalization is _shallow_ — it applies only to the outermost array, not recursively to children. A canonical `Struct` has its top-level field pointers in canonical form, but each child field may still be dictionary-encoded, run-length-encoded, etc. This is an important design choice: it means canonicalization is a local operation, not a full tree traversal.

The critical property is that this is a _choice_. Nothing in the theory privileges ListView over List as the canonical representative for variable-length list data. Both are valid sections. The current system hardcodes one.

**Further reading:**

- Section in algebra / category theory: [https://en.wikipedia.org/wiki/Section\_(category_theory)](<https://en.wikipedia.org/wiki/Section_(category_theory)>)
- Splitting lemma (analogous structure in group theory): [https://en.wikipedia.org/wiki/Splitting_lemma](https://en.wikipedia.org/wiki/Splitting_lemma)

### 1.4 The Church-Rosser Property

A rewriting system has the **Church-Rosser property** (or is **confluent**) if, whenever a term can be reduced in two different ways, both reduction paths can be continued to reach the same normal form. Equivalently: every term has at most one normal form.

This is the property that makes "just keep simplifying until you're done" a well-defined strategy. If a system is Church-Rosser, the order in which you apply reduction rules doesn't matter — you always arrive at the same result.

**In Vortex (current state):** The current `to_canonical` pathway is confluent by construction. No matter how deeply nested the encodings are (e.g., dictionary-of-run-length-of-bitpacked), executing the lazy canonicalization plan converges to the same canonical form. There's one normal form per DType, and every reduction path reaches it. This is clean and simple.

**In Vortex (if we add multiple canonical targets):** We would be moving to a _non-confluent_ rewriting system. A `List(Int32)` encoded as dictionary-of-run-length could reduce to _either_ a flat `List` (contiguous monotonic offsets) or a `ListView` (offset-size pairs with potential sharing), depending on which reduction strategy is applied. The system no longer has a unique normal form per logical type.

This isn't pathological — it's well-studied. The standard approach is to define **separate reduction relations**, each of which is internally confluent. For example, you might have one reduction strategy that targets ListView as the normal form for list types, and another that targets List. Each strategy independently satisfies Church-Rosser — within a given strategy, all paths converge. The strategies simply converge to _different_ normal forms. The choice of strategy is made by the consumer (the caller of `to_canonical`), not by the encoded array itself.

**Further reading:**

- Church-Rosser theorem: [https://en.wikipedia.org/wiki/Church%E2%80%93Rosser_theorem](https://en.wikipedia.org/wiki/Church%E2%80%93Rosser_theorem)
- Confluence in rewriting systems: [https://en.wikipedia.org/wiki/Confluence\_(abstract_rewriting)](<https://en.wikipedia.org/wiki/Confluence_(abstract_rewriting)>)
- Term rewriting and all that (Baader, Nipkow): [https://www.cambridge.org/core/books/term-rewriting-and-all-that/71768055C83EA8B18A58B8B09BEF3AB5](https://www.cambridge.org/core/books/term-rewriting-and-all-that/71768055C83EA8B18A58B8B09BEF3AB5)

### 1.5 Representation Independence and Existential Types

**Representation independence** is the principle that clients of an abstract data type cannot observe which concrete representation is in use. Mitchell and Plotkin (1988) proved that abstract data types correspond to **existential types** in System F:

```
∃Repr. {
    encode : LogicalBuffer → Repr,
    decode : Repr → LogicalBuffer,
    ops    : Repr → Results
}
```

The `Repr` type variable is hidden from the client. The client can only interact with the data through the provided operations, which are guaranteed to respect the logical semantics. Two implementations that expose the same interface and produce the same results are interchangeable.

**In Vortex:** Each encoding (dictionary, RLE, bitpacked, etc.) is an existential package. The query engine is a client that interacts through `scalar_at`, `slice`, `filter`, `take`, etc. The physical layout is the hidden `Repr`. Representation independence is what makes the whole layered architecture work — the query engine doesn't branch on encoding type.

**Further reading:**

- Mitchell, Plotkin, "Abstract Types Have Existential Type": [https://doi.org/10.1145/44501.45065](https://doi.org/10.1145/44501.45065)
- Existential types in type theory: [https://ncatlab.org/nlab/show/existential+type](https://ncatlab.org/nlab/show/existential+type)

---

## 2. Structural vs. Nominal Typing in DType

### 2.1 The Distinction

In a **structural** type system, two types are the same if they have the same structure. In a **nominal** type system, two types are the same only if they have the same name (declaration), even if their structures are identical.

**Further reading:**

- Structural vs. nominal type systems: [https://en.wikipedia.org/wiki/Nominal_type_system](https://en.wikipedia.org/wiki/Nominal_type_system)

### 2.2 Where Vortex's DType is Nominal

Several DType variants are structurally isomorphic but intentionally kept distinct:

| Type A                 | Type B                              | Structurally identical?                         | Semantically distinct?                                                      |
| ---------------------- | ----------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------- |
| `Utf8`                 | `Binary`                            | Yes (both are variable-length byte sequences)   | **Yes** — `Utf8` carries a validity invariant and enables string operations |
| `FixedSizeList(U8, n)` | (hypothetical) `FixedSizeBinary(n)` | Yes (both are fixed-width byte windows)         | **Unclear** — this is the open question                                     |
| `List(T)`              | `ListView(T)` (if it were in DType) | Yes (both are variable-length sequences of `T`) | **No** — same logical semantics, different physical form                    |

The first case (Utf8 vs Binary) is a clear **refinement type**: `Utf8` is `Binary` refined by the predicate `valid_utf8(bytes)`. This predicate gates operations (string functions) and imposes invariants. The nominal distinction is load-bearing.

The third case (List vs ListView) is clearly _not_ a nominal type distinction — it's a physical encoding difference that should not appear in DType. This is already handled correctly.

The second case (FixedSizeList<u8> vs FixedSizeBinary) is the interesting question, analyzed in Section 3.

### 2.3 List vs ListView: The Canonical Example of a Section Choice

The relationship between List and ListView deserves detailed treatment because it is the cleanest example of how the logical/physical split works in Vortex — and the example that most directly motivates the multi-section proposal.

**Logical equivalence.** List and ListView represent exactly the same logical data: a sequence of variable-length sub-arrays. Given an array of type `List(Int32)`, element `i` is a variable-length sequence of `Int32` values. This is true regardless of whether the physical layout uses monotonic offsets (List) or offset-size pairs (ListView).

**Physical difference.** The distinction is entirely in the buffer layout:

- **List** stores a single offsets buffer where `offsets[i]..offsets[i+1]` defines the range for element `i`. Offsets are monotonically increasing. The child values buffer is contiguous and non-overlapping — every byte belongs to exactly one logical element.
- **ListView** stores separate offsets and sizes buffers, where `offsets[i]..offsets[i]+sizes[i]` defines the range for element `i`. This allows _overlapping ranges_ (two logical elements can share backing data) and _gaps_ (regions of the values buffer that belong to no element).

**Why this is purely physical.** No query operation can observe the difference. `scalar_at(i)` returns the same sub-array. `filter`, `take`, `slice` all produce logically identical results. The overlapping and gaps in ListView are properties of the physical layout, not the logical data. A consumer that reads element `i` gets the same list of values regardless of which representation backs it.

This is precisely the quotient type property: List and ListView are two representatives of the same equivalence class. Any function that "descends to the quotient" — that is well-defined on the logical type — produces the same result on both.

**Why it matters for canonicalization.** The current system chooses ListView as the canonical form for `List(T)`. This is a valid section. But it is not the _only_ valid section. Some consumers (particularly Arrow FFI boundaries) would prefer List, because Arrow's canonical layout uses monotonic offsets. Other consumers might prefer ListView because it avoids the compaction cost of eliminating sharing.

This is the motivating case for parameterized sections: the same logical type, two valid canonical forms, and different consumers with different preferences. The theory says this is fine — you just need each section to be internally consistent (Church-Rosser within itself).

**Compaction as a morphism.** Converting from ListView to List requires _compaction_ — eliminating overlaps and gaps by copying values into a contiguous buffer and recomputing monotonic offsets. Converting from List to ListView is trivial (sizes are just offset deltas). This asymmetry is interesting: the section that targets List is more expensive to compute, but produces a form with stronger structural guarantees (no aliasing, no gaps). The section that targets ListView is cheaper but permits aliasing. Both are correct.

### 2.4 Refinement Types

A **refinement type** `{ x : T | P(x) }` is a type `T` restricted to values satisfying predicate `P`. Refinement types let you express subtypes without changing the underlying representation.

**In Vortex:**

```
Utf8  ≅  { b : Binary | valid_utf8(b) }
```

This is a proper refinement: every Utf8 value is a valid Binary value, but not every Binary value is valid Utf8. The predicate `valid_utf8` is what justifies the separate DType variant. Without it, Utf8 and Binary would be the same type, and maintaining both would be redundant.

**The test for whether a new DType variant is justified:** Does it carry a predicate that gates operations or imposes invariants that the base type does not? If yes, it's a refinement type and belongs in DType. If no, it may be better modeled as a canonical form (section), an encoding, or a type alias.

**Further reading:**

- Refinement types: [https://en.wikipedia.org/wiki/Refinement_type](https://en.wikipedia.org/wiki/Refinement_type)
- Liquid types (Rondon, Kawaguci, Jhala): [https://doi.org/10.1145/1375581.1375602](https://doi.org/10.1145/1375581.1375602)

---

## 3. Analysis: Should FixedSizeBinary Be a DType?

### 3.1 The Case For (Refinement / Nominal Argument)

FixedSizeBinary could be justified if it carries a semantic distinction that `FixedSizeList(U8, n)` does not:

- **Intent signaling:** FixedSizeBinary says "this is opaque binary data" (UUIDs, hashes, IP addresses), while FixedSizeList<u8> says "this is a list of bytes that happens to have a fixed length." The operations you'd want on each might differ (hex encoding, byte-order operations for FixedSizeBinary vs. element-wise list operations for FixedSizeList).
- **Schema compatibility:** Arrow, Parquet, and other formats distinguish these types. A FixedSizeBinary DType makes round-tripping schemas lossless.
- **Potential invariant:** FixedSizeBinary could carry the invariant "elements are not individually addressable / meaningful" — an opacity predicate. This is weaker than `valid_utf8` but still semantic.

Under this reading, FixedSizeBinary is a lightweight refinement type, and the nominal distinction earns its place.

### 3.2 The Case Against (Canonical Form / Section Argument)

FixedSizeBinary could instead be a canonical form (section target) of `FixedSizeList(U8, n)`:

- **No gating predicate:** If no operations are meaningful on FixedSizeBinary that aren't on FixedSizeList<u8>, then the predicate is empty and the refinement is trivial.
- **Leaky abstraction risk:** Every query engine function that handles list types would need to additionally handle FixedSizeBinary, or you need coercion rules. If the handling is always identical, the DType distinction adds complexity without semantic payoff.
- **Schema mapping as metadata:** The "this came from a Parquet FixedSizeBinary column" information could live in extension type metadata rather than in the core DType enum, keeping the logical layer minimal.

Under this reading, FixedSizeBinary is an encoding or canonical form, not a logical type.

### 3.3 Decision Framework

The following questions can help determine where a proposed type belongs:

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

For FixedSizeBinary: if it gates operations → DType. If it's structurally and operationally identical to FixedSizeList<u8> → canonical form or extension metadata.

---

## 4. The Multi-Section Architecture

### 4.1 Current State: Single Section

Today, `to_canonical` is a single global section:

```
to_canonical : EncodedArray → CanonicalArray
```

It executes a lazy array tree (the canonicalization plan) to map every encoded array to a unique canonical form determined by its DType. The system is confluent (Church-Rosser) — all reduction paths converge. Canonicalization is shallow: only the outermost layer is reduced to canonical form.

### 4.2 Proposed State: Parameterized Sections

The system would support multiple named sections, each targeting a different canonical form:

```
to_canonical : EncodedArray × Target → CanonicalArray
```

where `Target` selects from a family of valid canonical forms for the given DType.

**Concrete example for list types:**

| DType               | `Target::A`                                     | `Target::B`                            |
| ------------------- | ----------------------------------------------- | -------------------------------------- |
| `List(T, _)`        | ListView (offset-size pairs, potential sharing) | List (monotonic offsets, contiguous)   |
| `Utf8(_)`           | VarBin (offsets + sizes + data)                 | Flat UTF-8 (offsets + contiguous data) |
| `Primitive(I32, _)` | Flat buffer                                     | Flat buffer (same in both)             |

The targets are not necessarily "Vortex" vs "Arrow" — the naming and number of targets is a design choice. A target might exist for a specific FFI boundary, a serialization format, or an optimization preference. The point is that each target defines an internally confluent reduction. The choice of target is made by the consumer (query engine, FFI boundary, serializer), not by the encoded array.

### 4.3 Formal Structure

The layered architecture becomes a commutative diagram:

```
                    EncodedArray
                   /     |      \
                  /      |       \
          section_a   section_b   section_c  ...
                /        |          \
               /         |           \
        ListView       List      (other form)
               \         |          /
                \        |         /
                 ─── DType (quotient) ───
```

All paths through the diagram commute: decoding any canonical form to logical values yields the same result. This is exactly the universal property of the quotient — any two representatives in the same equivalence class decode to the same data.

### 4.4 Implementation Sketch

```rust
/// A canonicalization target — selects which section to use.
///
/// The specific variants here are illustrative. The actual set of targets
/// is a design decision based on which canonical forms consumers need.
enum CanonicalTarget {
    /// Default canonical forms (e.g., ListView for lists, VarBin for strings).
    Default,
    /// Alternative canonical forms (e.g., List for lists, contiguous strings).
    /// Could be motivated by FFI boundaries, serialization, or optimization.
    Contiguous,
}

/// Canonicalize an encoded array into the target representation.
///
/// Executes the lazy array tree to reduce the outermost encoding to a
/// canonical form selected by `target`. Children are not recursively
/// canonicalized.
fn to_canonical(array: &Array, target: CanonicalTarget) -> VortexResult<Array> {
    // Each target is an internally confluent reduction strategy.
    // The logical DType is preserved regardless of target.
    match target {
        CanonicalTarget::Default => to_canonical_default(array),
        CanonicalTarget::Contiguous => to_canonical_contiguous(array),
    }
}
```

The invariant to maintain is that for any `array`, `target_a`, `target_b`:

```
decode(to_canonical(array, target_a)) == decode(to_canonical(array, target_b))
```

That is: different canonical forms of the same data decode to the same logical values. This is the quotient compatibility condition.

---

## 5. Summary of Theoretical Landscape

| Concept                           | Role in Vortex                                                                      | Key Reference                                                                          |
| --------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Equivalence class**             | All physical arrays with the same logical data form a class                         | Standard set theory                                                                    |
| **Quotient type**                 | `DType` is the quotient over encodings by decode-equality                           | [nLab: quotient type](https://ncatlab.org/nlab/show/quotient+type)                     |
| **Section**                       | `to_canonical` picks one representative per class                                   | [Section (category theory)](<https://en.wikipedia.org/wiki/Section_(category_theory)>) |
| **Church-Rosser**                 | Single-section canonicalization is confluent; multi-section is confluent per-target | [Confluence](<https://en.wikipedia.org/wiki/Confluence_(abstract_rewriting)>)          |
| **Existential types**             | Each encoding hides its representation behind the DType interface                   | [Mitchell & Plotkin 1988](https://doi.org/10.1145/44501.45065)                         |
| **Representation independence**   | Query engine can't observe which encoding is in use                                 | Follows from existential typing                                                        |
| **Refinement types**              | Utf8 refines Binary with `valid_utf8`; justifies separate DType                     | [Liquid Types](https://doi.org/10.1145/1375581.1375602)                                |
| **Structural vs. nominal typing** | DType is nominal — structurally isomorphic types can be distinct                    | [Nominal type systems](https://en.wikipedia.org/wiki/Nominal_type_system)              |

---

## 6. Further Reading

### Foundational

- **Types and Programming Languages** (Pierce, 2002) — Chapters on existential types, subtyping, and type equivalence. The standard graduate textbook for type theory.
- **Term Rewriting and All That** (Baader & Nipkow, 1998) — Rigorous treatment of confluence, normal forms, and the Church-Rosser property.
- **Abstract Types Have Existential Type** (Mitchell & Plotkin, 1988) — The original paper connecting data abstraction to existential quantification. [DOI](https://doi.org/10.1145/44501.45065)

### On Quotient Types in Programming

- **Quotient Types for Programmers** (Angiuli, Coquand, 2021) — Accessible introduction to quotient types from an HoTT/cubical perspective. [nLab](https://ncatlab.org/nlab/show/quotient+type)
- **Observational Equality, Now!** (Altenkirch, McBride, 2007) — Explores when two representations should be considered "the same" in type theory. [PDF](http://www.cs.nott.ac.uk/~psztxa/publ/obseqnow.pdf)

### On Refinement Types

- **Refinement Types for ML** (Freeman & Pfenning, 1991) — The original paper on refinement types.
- **Liquid Types** (Rondon, Kawaguci, Jhala, 2008) — Decidable refinement type inference. [DOI](https://doi.org/10.1145/1375581.1375602)

### On Representation Independence

- **Abstraction and Specification in Program Development** (Liskov & Guttag, 1986) — Practical treatment of representation independence in software engineering.
- **Parametricity and Representation Independence** (Plotkin & Abadi, 1993) — Formalizes how parametric polymorphism guarantees representation independence.

### Columnar Format Context

- **The BtrBlocks Paper** — Direct ancestor of Vortex's compression approach.
- **Apache Arrow Specification** — The de facto standard for canonical columnar layout; relevant for understanding which canonical forms matter in practice. [https://arrow.apache.org/docs/format/Columnar.html](https://arrow.apache.org/docs/format/Columnar.html)
