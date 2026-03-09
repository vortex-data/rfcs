- Start Date: 2026-03-06
- RFC PR: TBD

# Multiple Canonical Targets

## Summary

This RFC proposes extending canonicalization from a single target (one canonical "normal form" per
`DType`) to parameterized targets via `to_canonical_target`, allowing consumers to choose among
multiple valid canonical representations (e.g., `List` vs. `ListView`). This builds on the type
system formalization in [RFC 0029](./0029-types.md).

## Motivation

There is an open question about whether the current `Canonical` system is the most ideal. The
relationship between `List` and `ListView` ties the concepts from
[RFC 0029](./0029-types.md) together and most directly motivates a multi-section (multiple normal
form) proposal.

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
For example, `scalar_at(i)` will always return the same list, and `filter`, `take`, and `slice` all
produce logically identical results.

However, this physical distinction has massive performance implications. Converting from `ListView`
to `List` requires rebuilding the entire array to eliminate overlaps and gaps. On the other hand,
converting from `List` to `ListView` is trivial (sizes are just offset deltas).

This asymmetry is notable: the section that targets `List` is more expensive to compute but produces
a form with stronger structural guarantees (no aliasing, no gaps). The section that targets
`ListView` is cheaper and permits aliasing and faster random access.

Many of our consumers (particularly Arrow FFI boundaries and consumers of DataFusion) prefer `List`
because Arrow is adding support for `ListView` slowly. Other consumers prefer `ListView` because
some operations (namely random access and potentially dependent operations) are faster.

It is highly unfortunate that we cannot canonicalize into different targets, and we are forced to
always decompress into `ListView`. We have the same constraint on `VarBinView` vs `VarBin`, and
while we haven't seen as many performance problems, it feels like a constraint that is too
restrictive in Vortex.

## Background

As described in [RFC 0029](./0029-types.md), Vortex's `to_canonical` function is a **section** on
the quotient type `DType`: it picks one canonical physical encoding for each logical type. The
current system is **confluent** (Church-Rosser): there is one normal form per `DType`, and every
reduction path reaches it.

A **non-confluent** rewriting system is one where two reduction paths from the same starting point
can arrive at different normal forms. The standard approach for handling non-confluent systems is to
define **separate reduction relations**, each of which is internally confluent. For example, instead
of one global set of reduction rules, you define two strategies: strategy A always reduces to normal
form X, and strategy B always reduces to normal form Y. Each strategy satisfies Church-Rosser
independently, the only difference is which normal form they target.

In Vortex, a similar scenario would be defining multiple strategies of canonicalization, where one
strategy could target `List` as a canonical target (normal form) for list data, and another strategy
could target `ListView`.

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

Because `DType` is a quotient type and canonicalization is a section on that quotient, supporting
multiple canonical targets is theoretically sound (adding this feature will not weaken any invariant
of the Vortex type system). Because each `CanonicalTarget` defines an internally confluent reduction
strategy, the quotient structure of `DType` guarantees that all well-defined operations produce
the same logical results regardless of which target is chosen.

```
Logical type        (DType: purely semantic, not physical)
        ▲
        │  section 1: Default    (ListView, VarBinView)
        │  section 2: Contiguous (List, VarBin)
        │  section n: ...        (future targets)
        ▼
Canonical form      (a specific "normal form" encoding chosen by the section)
        ▲
        │  encode
        │  decode
        ▼
Physical encoding   (dictionary, REE, bitpacked, etc.)
```

## Compatibility

There shouldn't be any compatibility concerns here because even under a specific `DType`, the array
tree is fully serialized, and consumers can always convert back and forth between `List` and
`ListView` if they really need to.

## Drawbacks

The drawback is extra complexity in supporting multiple canonical targets. However, we've also had
to spend time making optimizations and fixes (`is_zero_copy_to_list` for `ListView`, see
[vortex#5129](https://github.com/vortex-data/vortex/pull/5129)) because we were forced to always
canonicalize into a single target. So there is an obvious tradeoff here.

## Alternatives

The alternative is to just not do this. We continue to find workarounds when canonical encodings do
not fit the use case.

## Prior Art

- **Arrow** defines both `List` and `ListView` (and `LargeList`/`LargeListView`) as separate type
  IDs in its columnar format. Arrow's canonical layout uses monotonic offsets (`List`), and
  `ListView` support is being added incrementally across implementations.
- **DuckDB** uses a single canonical form per logical type and does not support multiple
  canonicalization targets.

## Unresolved Questions

- What is the concrete set of `CanonicalTarget` variants? This RFC proposes `Default` and
  `Contiguous`, but there may be other useful targets. Theoretically, we could have a target that
  would allow us to return compressed arrays as a `Canonical` type!

## Future Possibilities

- User-defined or plugin-defined canonical targets for custom FFI boundaries or serialization
  formats.
