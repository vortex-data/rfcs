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
