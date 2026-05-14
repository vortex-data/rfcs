- Start Date: 2026-05-14
- Authors: @gatesn
- RFC PR: [vortex-data/rfcs#57](https://github.com/vortex-data/rfcs/pull/57)

# AI-Assisted Development Workflow

## Summary

Vortex should make AI-assisted development easier to review by changing repository workflow rather
than adding human paperwork. The main idea is to make diffs smaller, give bots enough structure to
route work, strengthen automated checks, and keep core code on a stricter path than leaf code.

The proposals below are separable. Each can be accepted, rejected, or deferred independently.

## Motivation

LLMs make it cheap to produce large amounts of code. That is useful, but it can overwhelm review if
the project keeps treating every PR as the same kind of work.

The project should optimize for two things:

- small, reviewable PRs
- high confidence around core Vortex behavior

The workflow should not depend on authors writing declarations that no tool can use. If metadata is
needed, it should be inferred from changed paths, labels, branch structure, CI results, or bot
comments.

## Goals

- Reduce review latency without lowering the bar for core correctness.
- Encourage stacked, focused PRs for large AI-assisted changes.
- Give bots enough structure to route checks and reviewers.
- Keep humans focused on design, correctness, compatibility, performance, and maintainability.
- Avoid ceremonies that depend on contributors manually writing process notes.

## Non-Goals

- This RFC does not require contributors to use LLMs.
- This RFC does not allow AI tools to approve or merge code on behalf of maintainers.
- This RFC does not define exact GitHub Actions, prompts, labels, or branch protection rules.
- This RFC does not change Vortex runtime behavior, file compatibility, or public API policy.

## Proposal 1: Prefer stacked PRs for large changes

Large changes should usually be split into a stack of small PRs.

Recommended splits include:

- mechanical refactor before behavior change
- tests before implementation
- public API shape before broad migration
- compatibility scaffolding before format behavior changes
- core change separate from leaf integration changes

The project should explicitly encourage use of the `gh stack` skill for contributors working with
Codex. Other stacking tools are fine; the requirement is the shape of the work, not the tool.

Implementation hint: document the preferred stack shape in `AGENTS.md` and contributor docs. Add a
bot warning for unusually large PRs that suggests splitting or stacking.

## Proposal 2: Bot-generated review briefs

Every PR should get an automated review brief before human review.

The brief should summarize:

- changed crates and paths
- likely risk area
- relevant owners
- CI and benchmark status
- whether public API, unsafe code, serialization, or performance-sensitive paths appear touched
- suggested reviewer focus areas

The brief is advisory. It should not approve, reject, or block a PR by itself.

Implementation hint: a GitHub Action can inspect the diff, apply path rules, and post or update a
single bot comment. An LLM can help summarize the diff, but path-based facts should come from
deterministic tooling.

## Proposal 3: Route review by crate and path

Vortex should classify crates and paths by expected review depth.

Suggested buckets:

- **Core:** format, dtype semantics, array invariants, encodings, layouts, serialization, unsafe
  code, public APIs, and performance-critical execution paths.
- **Shared:** IO, planning, reusable internals, and utilities used across multiple crates.
- **Leaf:** bindings, integrations, examples, documentation, benchmarks, and tests that do not
  change production behavior.
- **Incubating:** code that is not yet part of the stable production surface.

Core changes should require review from an owner of the affected area. Leaf changes can use a
lighter path when checks pass. Incubating code can move quickly while it stays isolated, but moving
it into the stable production path should be treated as a core or shared change.

Implementation hint: use `CODEOWNERS`, path labels, and branch protection rules. The bot-generated
review brief should explain why it assigned a bucket.

## Proposal 4: Group crates by stability inside the repository

The repository layout should make review expectations visible.

For example, crates could move toward directories such as:

- `crates/core/`
- `crates/shared/`
- `crates/integrations/`
- `crates/incubating/`
- `crates/benchmarks/`

The exact names are not part of this RFC. The intent is that path structure should reflect
stability and blast radius, so humans and bots can reason about review requirements.

Implementation hint: start with new crates and opportunistic moves. Do not block this workflow on a
large repository reorganization.

## Proposal 5: Treat incubation-to-production as promotion

Incubating code should not silently become production code.

Any change that wires incubating code into default execution, public APIs, stable file behavior, or
release artifacts should be treated as a promotion. Promotion should require the same review depth
as changing the destination area directly.

A promotion should show that the code has the checks expected of its new home. For example, code
moving into core execution should have core-style tests and benchmarks, not only incubating-area
coverage.

Implementation hint: define path rules that detect imports, feature-default changes, public API
exports, and release packaging changes involving `incubating` crates. The action can label the PR
as `promotion` and require the destination owners.

## Proposal 6: Expand machine confidence checks

The project should keep moving routine correctness work from human review into automated checks.

High-value checks for Vortex include:

- array and encoding roundtrip tests
- property tests for dtype, validity, offsets, and slicing behavior
- compatibility corpus tests for serialized files
- fuzzing for readers, decoders, and unsafe boundaries
- API diff checks for public Rust and language-binding surfaces
- targeted benchmark comparisons for performance-sensitive paths
- formatting and linting checks that remove style debate from reviews

Implementation hint: choose checks by changed paths and labels. Expensive checks can run for core
or promotion PRs, on demand, or before release rather than on every documentation change.

## Proposal 7: Allow a narrow fast path

Some PRs should move without waiting behind core review work.

Candidates include:

- documentation
- examples
- tests that do not change production behavior
- benchmark harness additions
- isolated integration fixes
- mechanical changes with strong automated validation

Fast-path PRs still need passing checks and clear ownership. The point is to avoid spending scarce
core-review attention on changes whose blast radius is low.

Implementation hint: a `fast-path` label should be assigned by path rules or maintainers. The
rules should exclude core, shared, promotion, public API, serialization, unsafe, and
performance-sensitive changes.

## Proposal 8: Measure review health

The project should track whether the workflow improves review throughput and quality.

Useful signals include:

- time to first review
- time waiting on review
- PR size
- number of PRs per stack
- PRs by bucket
- CI failure rate after review starts
- revert rate
- benchmark regression rate

Implementation hint: start with GitHub metadata and bot-produced labels. The goal is to identify
workflow bottlenecks, not score individual contributors.

## Compatibility

This RFC changes project workflow rather than Vortex runtime behavior.

It may affect repository layout, `CODEOWNERS`, labels, branch protection, CI policy, and contributor
documentation. It does not change the file format, wire format, public APIs, or release
compatibility guarantees.

## Drawbacks

Path-based routing can be wrong. Review buckets should guide maintainers, not replace judgment.

Moving crates into stability-oriented directories has churn. The migration should be incremental.

Fast paths can hide risk if the rules are too broad. The initial fast path should be narrow and
exclude anything that looks like promotion or core behavior.

Incubating areas can become dumping grounds if they are not paired with a real promotion rule.

## Alternatives

### Keep the current process

This avoids process and repository churn, but it leaves reviewers to manually infer risk and
context for every PR.

### Add more reviewers

More reviewers help only when they have the right context. This RFC focuses on reducing the amount
of context each reviewer must reconstruct.

### Require more PR text from authors

Manual declarations are easy to ignore and hard to enforce. This RFC prefers metadata that bots can
infer or maintainers can apply with labels.

### Let AI approve PRs

AI approval would reduce latency but weaken accountability. This RFC keeps AI in advisory roles:
summarizing, classifying, and suggesting reviewer focus areas. Approval remains human.

## Unresolved Questions

- What are the initial core, shared, leaf, and incubating path rules?
- Which crates should move first, if any?
- What exactly counts as promotion from incubating to production?
- Which checks are required for core and promotion PRs?
- How narrow should the first fast path be?

## Future Possibilities

- A crate dependency map that identifies leaf crates and high-blast-radius core crates.
- Per-area review playbooks for encodings, arrays, file format, bindings, and integrations.
- A compatibility dashboard showing file corpus coverage and public API changes.
- Bot comments that suggest a stack split when a PR mixes mechanical, behavioral, and integration
  changes.
