- Start Date: YYYY-MM-DD (today's date)
- Authors: @username1, @username2
- RFC PR: [vortex-data/rfcs#0000](https://github.com/vortex-data/rfcs/pull/0000)

# RFC Template

## Summary

One paragraph explanation of the proposed change.

## Motivation

What problem does this solve? Include concrete use cases where possible.

- What specific use cases does this enable or improve?
- What workflows or operations are painful, slow, or impossible today?

## Design

Describe the proposed design in enough detail that someone familiar with Vortex could implement it.
This should cover:

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

This section helps frame the design in a broader context.
If there is no relevant prior art, that is fine.

## Unresolved Questions

- What parts of the design need to be resolved during the RFC process?
- What is explicitly out of scope for this RFC?
- Are there open questions that can be deferred to implementation?

## Future Possibilities

What natural extensions or follow-on work does this enable? This is a good place to note related
ideas that are out of scope for this RFC but worth capturing.
