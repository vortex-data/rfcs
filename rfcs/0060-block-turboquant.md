# Block-Decomposed TurboQuant: a Lossy Extension Type with PDX Layout

- Start Date: 2026-05-12
- Authors: @lwwmanning, @connortsui20
- RFC PR: [vortex-data/rfcs#60](https://github.com/vortex-data/rfcs/pull/60)

## Summary

We propose **TurboQuant** as a **lossy logical extension type** in Vortex for
high-dimensional float vectors, registered as `vortex.turboquant`. The long-term
plan progresses through three stages — all designed around the same semantic
foundation and architectural shape:

1. **Stage 1 — Single-block, biased (MSE-only):** what's implemented today in
   the standalone `vortex-turboquant` crate. Storage is a two-field struct
   `{norms, codes}` under the extension dtype; rotation and centroids are
   derived from metadata, not stored. Adopts EDEN's optimized scalar scale `S`
   in place of TurboQuant's fixed `S = 1` as a strict, drop-in improvement.
2. **Stage 2 — Block decomposition:** add `block_size` to the extension
   metadata; per-block norms move into a fixed-size list. Eliminates power-of-2
   padding for non-power-of-2 dimensions (768 → 3×256 blocks instead of 768 →
   1024 padded).
3. **Stage 3 — PDX physical layout:** register a separate `PDXArray` physical
   encoding for the codes child, enabling SIMD scan kernels (dimension-major
   layout within 64-vector chunks). The TurboQuant extension type is unchanged.

Two design principles are invariant across all three stages and drive every
decision below:

- **Lossy compression is logical, not physical.** TurboQuant is a logical
  extension type that surfaces the mutation in the dtype itself; it is not a
  transparent encoding of `Vector`.
- **Decompression is explicit, never implicit.** `TQDecode` is the sole path
  back to floats. There is no canonicalization to `Vector`, no implicit
  recompression by the default cascade compressor, and no "approximate" flag —
  the encoding is lossy by definition.

This RFC is anchored to the current state of the `vortex-data/vortex`
`develop` branch — specifically the extension-type implementation in the
standalone `vortex-turboquant` crate (PR #7829, merged 2026-05-07). The
algorithm sits in a family that includes **EDEN** [15] ([arXiv:2108.08842],
ICML 2022) and its predecessor **DRIVE** [16] ([arXiv:2105.08339],
NeurIPS 2021); both predate TurboQuant [1] ([arXiv:2504.19874]). A recent
note by the EDEN authors [14] ([arXiv:2604.18555], April 2026) shows that
TurboQuant is a special case of EDEN with a suboptimal fixed scale `S = 1`.
We adopt EDEN's contributions while keeping the codebase's TurboQuant
branding; see §4 "Naming."

[arXiv:2108.08842]: https://arxiv.org/abs/2108.08842
[arXiv:2105.08339]: https://arxiv.org/abs/2105.08339
[arXiv:2604.18555]: https://arxiv.org/abs/2604.18555
[arXiv:2504.19874]: https://arxiv.org/abs/2504.19874v1
[current-impl]: https://github.com/spiraldb/vortex/tree/ff120401a0f4796f2d1aa85d1f87e7195c1f3dbf/vortex-turboquant
[original-impl]: https://github.com/spiraldb/vortex/pull/7167

## Motivation

Two motivations drive this design, and they are usefully framed together rather
than separately.

**Embedding workloads need a lossy option.** Vector search, RAG, KV-cache
attention, and similar workloads operate on high-dimensional float embeddings
(typically 384–4096 dims) where ~4× storage compression is achievable with
recall losses below a percentage point. Today Vortex compresses these vectors
with lossless schemes (ALP, FSST, BitPacked) and leaves a large amount of
storage and bandwidth on the table. A first-class lossy quantization path
opens up new workloads — billions-scale ANN search, on-disk KV-cache, embedding
serving — where a lossless format is over-engineered.

**Vortex needs a clean answer to "where does lossy data live?"** The original
RFC modelled TurboQuant as a new physical encoding of the `Vector` extension
type. That model breaks down on three concrete questions:

- Are quantized vectors unit-normalized? After scalar quantization on a rotated
  unit vector, the inverse transform does not generally recover a unit vector.
  A "Vector encoding" would have to either lie about the dtype's invariant or
  silently violate it.
- Does `cosine_similarity(a, a)` return 1.0? Only after a full decode-and-renorm
  cycle — and even then, only approximately. A transparent encoding would have
  to make the user think this still works exactly.
- What does canonicalization mean? Canonicalization is for switching between
  _representations of the same value_ (BitPacked ↔ Primitive, FSST ↔ VarBin).
  Quantization is not a representation change; it is a value change. Treating
  the inverse direction as canonicalization would collapse a one-way mapping
  into a round-trip and bake lossiness into a primitive that other parts of
  Vortex assume is lossless.

The right model — and the one this RFC commits to — is that lossy quantization
introduces a **new logical type** that records the mutation in the dtype.
Reading a TurboQuant column yields TurboQuant codes; reconstructing floats
requires an explicit `TQDecode` call. The default compressor never recompresses
a TurboQuant column (it sees an extension-typed `Struct` and recurses without
trying to compress it). Users are the source of truth for whether a given
column should be lossy.

## Semantic model

This section is foundational: it states the two principles every stage of the
long-term plan upholds, and it is the basis on which the architecture in §5–§8
is derived.

### Principle 1: Lossy compression is logical, not physical

Quantization mutates the data. Once mutated, equality, unit-normalization, and
distance functions no longer agree with the original. A physical encoding that
hides this would have to lie about all three.

The TurboQuant tracking issue
([vortex-data/vortex#7830](https://github.com/vortex-data/vortex/issues/7830))
records the team's conclusion verbatim:

> Lossy data and compression MUST live at the logical layer, not at the
> physical layer. It is logical because losing data is a full modification of
> the data, not just a different way of storing it.

The architectural consequence: TurboQuant is a Vortex **extension type**
(`vortex.turboquant`), not an encoding. Its storage is a `Struct` carrying
norms and codes; that struct is opaque to the default compressor (which sees
"just a struct" and recurses without recompressing the lossy data). The user
controls whether a column is lossy by explicitly encoding it via `TQEncode`
before write.

### Principle 2: Decompression is explicit, never implicit

Canonicalization is Vortex's mechanism for representation changes between
forms of the same value. It is structurally one-way for type identity:
canonicalizing a `BitPacked<u32>` produces a `Primitive<u32>` carrying the same
values. Applying that mechanism to lossy data would make `canonicalize(tq) →
Vector` look like a representation change when it is actually an information
loss event with bounded error.

The current `vortex-turboquant` implementation removes this trap entirely:
there is no canonicalization to `Vector`. Decoding goes through the explicit
`TQDecode` scalar function, which the user (or query plan) invokes deliberately
and the result is named `Vector`, not "the original floats." `TQDecode(TQEncode(v))`
returns values close to `v` (within MSE bounds), not `v` itself, and that
distinction is now visible in the operator chain.

### Implications for the three-stage plan

Both principles hold across every stage of the long-term plan:

- **Stage 1** registers the extension type and the encode/decode scalar
  functions. It is the smallest viable expression of both principles.
- **Stage 2** adds block decomposition. This changes the storage shape (norms
  becomes a fixed-size list) but does not change the logical type — it is the
  same `vortex.turboquant` extension carrying a different physical struct.
- **Stage 3** adds the PDX physical layout. This changes the _physical
  encoding_ of the codes child but does not change the logical type or the
  storage shape's struct view at all — PDX is registered as a separate
  encoding of `FixedSizeList<u8>`, and a TurboQuant array can carry either
  layout for its codes interchangeably.

The extension type, the `TQEncode`/`TQDecode` boundary, and the
encoded-data-is-lossy contract are invariant across stages. Each stage is a
strictly-additive extension of the previous.

## Algorithm and prior art

### TurboQuant

TurboQuant [1] is a lossy vector quantization algorithm for high-dimensional
embeddings. It works by:

1. Randomly rotating a unit-norm vector so that each coordinate follows a known
   marginal distribution — specifically `(1 - x²)^((d-3)/2)` on [-1, 1], a
   concentrated Beta distribution (Lemma 1 in [1]).
2. Applying an MSE-optimal scalar quantizer (Max-Lloyd centroids) independently
   to each coordinate.
3. Optionally adding a 1-bit QJL (Quantized Johnson–Lindenstrauss) correction
   on the residual for unbiased inner-product estimation (Theorem 2 in [1]).

The paper prescribes a full random orthogonal rotation (QR decomposition of a
matrix with i.i.d. N(0,1) entries, yielding a Haar-uniform orthogonal matrix)
for the MSE stage — O(d²) storage and O(d²) per-vector. We replace this with a
three-round Structured Orthogonal Random Features (SORF) transform [5] for
O(d log d) compute and O(d) storage; see "Current Vortex implementation"
below.

### Theoretical MSE bound

The paper's MSE bound (Theorem 1 in [1]) is:

```
E[‖x - x̂‖² / ‖x‖²] ≤ (√3 · π / 2) / 4^b ≈ 2.72 / 4^b
```

**Crucially, Theorem 1 is proved for true random orthogonal matrices (QR of
Gaussian), not SORF.** Our SORF is an approximation. The bound holds exactly
only with a true random orthogonal rotation or with empirical SORF validation
(see Experimental plan). At d=1024, the observed 4-bit MSE exceeds the
theoretical bound by ~20% (0.0127 vs. 0.0106) — a small practical gap that
auto-vectorizes to SIMD on encode and decode.

For a vector split into k blocks (Stage 2), per-block MSE bounds compose
algebraically:

```
‖x - x̂‖² / ‖x‖² = Σ_k (‖xₖ‖² / ‖x‖²) × (‖xₖ - x̂ₖ‖² / ‖xₖ‖²)      (exact)
    E[...]         ≤ MSE_bound × Σ_k (‖xₖ‖² / ‖x‖²) = MSE_bound          (in expectation)
```

The conclusion: `E[‖x - x̂‖² / ‖x‖²] ≤ MSE_bound` assuming independent
per-block rotations. Theorem 1 applies because each block is normalized to
unit norm before rotation and quantization. Note that with smaller block
dimension B, within-block coordinate dependence after rotation may be stronger
even when marginals are correct — see the Experimental plan's cross-block
correlation tests.

### Relationship to EDEN and DRIVE

An important piece of prior art for this design family is **EDEN** [15]
(Vargaftik et al., ICML 2022; [arXiv:2108.08842]) and its predecessor
**DRIVE** [16] (NeurIPS 2021; [arXiv:2105.08339]) use the same building blocks
as TurboQuant — Randomized Hadamard Transform plus Lloyd–Max scalar
quantization on a rotated, Beta-distributed unit vector. EDEN predates
TurboQuant by two-plus years and generalizes it: DRIVE is a 1-bit quantizer
that EDEN extends to any `b > 0` bits per coordinate.

In April 2026, the EDEN authors posted a clarification note [14]
([arXiv:2604.18555], Ben-Basat, Ben-Itzhak, Mendelson, Mitzenmacher, Portnoy,
Vargaftik) titled "A Note on TurboQuant and the Earlier DRIVE/EDEN Line of
Work," which explicitly catalogues TurboQuant's relationship to EDEN. We
draw the comparisons in this section from that note's abstract and from EDEN
[15] directly.

The substantive differences are:

| Aspect                 | TurboQuant [1]                                | EDEN [15]                                    |
| ---------------------- | --------------------------------------------- | -------------------------------------------- |
| Rotation               | Random orthogonal (paper) / SORF (ours)       | Randomized Hadamard Transform (same family)  |
| Marginal distribution  | Beta `(1-x²)^((d-3)/2)`                       | Same (shifted Beta after rotation)           |
| Centroids              | Max-Lloyd on the Beta marginal                | Same                                         |
| Scalar quantizer scale | Fixed `S = 1`                                 | **Optimal `S` per `(dimension, bit_width)`** |
| Biased mode            | MSE-only                                      | Yes (biased EDEN, optimal `S`)               |
| Unbiased mode          | MSE + QJL stacking (b-1 MSE bits + 1 QJL bit) | **Native b-bit unbiased EDEN at any b > 0**  |
| Reported relative MSE  | Theorem 1 bound: `(√3·π/2)/4^b ≈ 2.72·4⁻ᵇ`    | Tighter bound from EDEN's optimal `S`        |
| Bias of inner product  | Biased (MSE) / unbiased (Prod)                | Either, both natively                        |

The note [14] argues TurboQuant is suboptimal in two specific ways:

1. **Fixed `S = 1` is asymptotically optimal but not finite-dim optimal.** The
   note states: _"The fixed choice `S = 1` used by TurboQuant is generally
   suboptimal, although the optimal `S` for biased EDEN converges to `1` as
   the dimension grows; accordingly TurboQuant_mse approaches EDEN's behavior
   for large d"_ [14]. At practical dimensions, EDEN's per-`(d, b)` optimal
   `S` strictly reduces MSE.
2. **EDEN's native unbiased mode dominates TurboQuant's MSE+QJL "product"
   stacking.** The note reports _"biased EDEN (with optimized S) is more
   accurate than TurboQuant_mse, and unbiased EDEN is markedly more accurate
   than TurboQuant_prod, often by more than a bit (e.g., 2-bit EDEN beats
   3-bit TurboQuant_prod)"_ [14].

**What this means for the RFC.**

- We adopt EDEN's **optimized scale `S`** as a Stage 1 refinement (see
  §6 "Stage 1"). This is a strict drop-in win: same storage, same metadata,
  better quantization accuracy at fixed bit budget. The implementer needs
  EDEN [15] (not the note [14]) for the optimization criterion — the note
  defers to "methods described in the EDEN works."
- We adopt EDEN's **native b-bit unbiased mode** as the preferred path for any
  future unbiased estimator, in place of TurboQuant's MSE+QJL stacking (see
  §15 "Future work" and Appendix C).

**What this RFC's block decomposition adds over both papers.** Both TurboQuant
and EDEN are single-rotation algorithms: one rotation over the full vector
dimension. The block decomposition in Stage 2 is independent of which scalar
quantizer is used — it is a Vortex-specific design that eliminates power-of-2
padding for non-power-of-2 dimensions (e.g., 768 → 3×256 blocks). Stage 3 (PDX
layout) is similarly independent of the scalar quantizer choice. So
adopting EDEN's `S` is orthogonal to the rest of the long-term plan.

### Naming

We use the **TurboQuant** name throughout this RFC and in the codebase
(`vortex-turboquant`, `vortex.turboquant` extension ID, `TQEncode`,
`TQDecode`) even though the algorithm is more correctly an instance of EDEN
with an extra block-decomposition layer. The pragmatic reasons:

- The codebase has converged on `vortex-turboquant` (PR #7829). Renaming the
  crate, extension ID, and scalar functions to EDEN-derived names mid-stream
  would churn downstream consumers and the migration story for no algorithmic
  gain.
- The embedding-quantization community recognizes "TurboQuant" via its
  ICLR 2026 acceptance, and external users encountering Vortex's lossy vector
  type will reach for that name first.

This is a credit-attribution decision rather than a technical one. The
algorithm we implement is EDEN with optimal `S` plus block decomposition;
this RFC engages explicitly with the EDEN priority (see §4 "Relationship to
EDEN and DRIVE" above and the references) so the academic record is
preserved. If the EDEN authors prefer an alternative framing in published
materials, we can revisit; we expect to consult them before any public-facing
launch.

### Comparison to Product Quantization

TurboQuant's block decomposition (Stage 2 of this RFC) is structurally similar
to Product Quantization (PQ) [9]: both partition a vector into sub-vectors and
quantize each independently. The key differences are:

|                        | TurboQuant                                                      | PQ                                                       |
| ---------------------- | --------------------------------------------------------------- | -------------------------------------------------------- |
| Quantization type      | Scalar (per-coordinate, after rotation)                         | Vector (per-sub-vector, learned codebook)                |
| Codebook               | Analytically derived from Beta distribution; **data-oblivious** | Learned via k-means on training data; **data-dependent** |
| Rotation               | Random orthogonal within each sub-vector                        | Typically none (OPQ [10] adds a learned rotation)        |
| Theoretical guarantees | Provable data-oblivious MSE bound (Theorem 1 [1])               | No comparable data-oblivious bound                       |
| Codebook training      | None (centroids derived from theory)                            | Requires training pass over data                         |
| Bits per sub-vector    | Scalar: b bits per coordinate                                   | Vector: typically 8 bits per sub-vector (256 codewords)  |

TurboQuant trades PQ's flexibility (data-dependent codebooks can exploit
structure) for data-obliviousness (no training, provable bounds, no offline
index-training phase). In return, PQ and OPQ retain a major advantage in
expressivity: they learn sub-vector codebooks from data rather than applying
an analytic scalar quantizer. In practice this means TurboQuant is attractive
when training-free operation, simple deployment, and theoretical guarantees
matter most, while PQ or OPQ may still win empirically when a learned vector
codebook can exploit dataset-specific structure.

### Comparison to HIGGS

HIGGS [12] (Malinovskii et al., 2024) is a data-free quantization method for
LLM weight matrices that shares TurboQuant's core idea — Hadamard rotation
followed by MSE-optimal grid quantization — but targets a different application
domain and makes different design trade-offs:

|                      | TurboQuant                                                               | HIGGS                                                                      |
| -------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| Application domain   | ANN embedding search (per-vector, online)                                | LLM weight quantization (per-layer, offline)                               |
| Rotation             | 3-round SORF (HD₃·HD₂·HD₁): high-quality random orthogonal approximation | Single RHT (H·D): one Hadamard × random diagonal signs                     |
| Target distribution  | Beta marginal (1-x²)^((d-3)/2) on unit sphere                            | Approximate Gaussian N(0,1)                                                |
| Quantization grid    | Max-Lloyd centroids (scalar, p=1), analytically derived for Beta         | CLVQ grids (Pagès & Printems 2003), supports vector quantization p∈{1,2,4} |
| Error metric         | Pure MSE (reconstruction error)                                          | MSE + Hessian-weighted per-layer coefficients αₗ (Linearity Theorem)       |
| Calibration data     | None                                                                     | None for quantization; small calibration set for αₗ estimation             |
| Non-uniform bitwidth | No (uniform across all vectors)                                          | Yes (DP solver for per-layer bit allocation)                               |
| Distance computation | Quantized-domain scan kernel (PDX layout, SIMD over 64 vectors)          | GPU matrix multiply (FLUTE kernel)                                         |
| Norm storage         | Explicit per-block norms for distance computation                        | Per-group scales folded into weight reconstruction                         |

Domain mismatch: comparing TurboQuant vs. HIGGS on LLM perplexity benchmarks
is misleading because HIGGS's Hessian-aware optimization naturally dominates
for that task. The relevant comparison is ANN recall@k on embedding datasets,
where TurboQuant's block decomposition, PDX scan layout, and per-vector
encode/decode are the critical features. HIGGS's vector quantization (p>1)
remains an interesting future direction — see §15 "Future work."

### Comparison to RotorQuant / IsoQuant

RotorQuant [13] replaces TurboQuant's full-dimension SORF with Clifford
algebra rotors in Cl(3,0), chunking vectors into 3-dimensional groups and
applying SO(3) sandwich products. IsoQuant extends this to SO(4) via
quaternions, and PlanarQuant uses SO(2) Givens rotations.

On real KV-cache tensors (Qwen2.5-3B), these small-block rotations showed
severe quality regressions: RotorQuant at 3-bit measured 3.843 MSE vs.
TurboQuant's 0.354 (10.8× worse), and IsoQuant at 4-bit incurred +36%
perplexity impact vs. TurboQuant's +11.7% [13]. Independent analysis
attributed this to the fundamental decorrelation limitation: block-diagonal
rotations in SO(2)/SO(3)/SO(4) provide no cross-group coordinate mixing,
while WHT/SORF mixes all coordinates simultaneously.

**Relevance to our design.** Stage 2's block decomposition is also
block-diagonal — each B-dim block has an independent SORF with no cross-block
mixing. The critical difference is block size: B=256 with 3-round SORF
provides 24 butterfly stages of within-block mixing (comparable to the
current B=1024's 30 stages), vs. RotorQuant's 3-4 coordinate groups with no
structured mixing at all. The RotorQuant/IsoQuant data validates the RFC's
minimum B ≥ 64 constraint and provides empirical evidence that the quality
cliff for block-diagonal rotations is steep at very small B. Whether B=256 is
large enough to avoid meaningful decorrelation loss is an empirical question
addressed in the Experimental plan.

### Current Vortex implementation

The Stage 1 implementation lives on `vortex-data/vortex` `develop` in the new
standalone [`vortex-turboquant` crate][current-impl] (PR #7829, merged
2026-05-07). The earlier monolithic-array prototype in `vortex-tensor/src/encodings/turboquant/`
and the decomposed-scalar-fn prototype (PR #7374) are predecessors that this
RFC supersedes; both remain on `develop` only until the example and
documentation migrate.

**Extension type.** Registered via `ExtVTable`:

- ID: `vortex.turboquant`
- Metadata (prost-serialized): `element_ptype`, `dimensions` (original,
  pre-padding), `bit_width`, `seed`, `num_rounds`. Stage 2 adds `block_size`;
  see §7.
- Storage dtype: `Struct { norms: Primitive<element_ptype>, codes: FixedSizeList<u8, padded_dim> }`
  with row-aligned validity.

**Scalar functions.** Two scalar functions form the explicit encode/decode
boundary:

- `TQEncode(Vector, config) → TurboQuant<extension>`. Stores per-vector L2 norm,
  L2-normalizes, applies SORF, quantizes each rotated coordinate to its nearest
  centroid. Not a `ScalarFnArrayVTable` — persisting an unexecuted encode
  would write the original Vector, defeating the lossy intent.
- `TQDecode(TurboQuant<extension>) → Vector`. Dequantizes via the centroids,
  applies inverse SORF, truncates to the original dimension, re-applies the
  stored norm. Lazy by default; users execute when needed.

**Derived state, not stored.** Two derivations make the storage shape minimal:

- **SORF rotation** is reconstructed from `seed` via Vortex's frozen local
  SplitMix64 sign stream. The rotation matrix is therefore deterministic in
  `(seed, num_rounds, padded_dim)`; no rotation signs are stored.
- **Centroids** are reconstructed from `(padded_dim, bit_width)` via numerical
  integration of the Beta marginal (trapezoid rule, 1000 points per interval,
  Max-Lloyd algorithm). They are cached process-locally in a `DashMap` keyed
  by that tuple. With EDEN's optimized scale (§6), the cache key is the same
  and the cache stores both centroids and `S` together.

**Crate boundary.** `vortex-turboquant` depends on `vortex-tensor` (for the
`Vector` extension type) but **nothing in core Vortex depends on
`vortex-turboquant`**. The crate is positioned to mimic a third-party
extension: users opt in via `vortex_turboquant::initialize(&session)` after
`vortex_tensor::initialize(&session)`. The default cascade compressor does not
know about TurboQuant at all — it sees the extension-typed `Struct` and
recurses, passing the encoded data through opaquely (norms get compressed by
whatever scheme matches; codes are typically left alone since they are
already near-entropy-limited at the chosen bit width).

**Reference implementation bug check.** The Eviox corrections study [7]
identified six material bugs in the paper's reference Python implementation.
The most critical is a mathematical error in the QJL scale factor: the
reference code used `√(π/(2d))` instead of `√(π/2)/d` (Definition 1 in [1]),
differing by a factor of √d (≈11× at d=128). Our implementation uses the
correct formula (`sqrt(FRAC_PI_2) / padded_dim` in Rust), so this bug does
not affect us. Other Eviox findings: (a) the reference code recomputes
codebooks at every instantiation (we cache in a `DashMap`); (b) the reference
uses float16 for codebook distance computation, causing misassignment at
small centroid spacings (we cast to f32 before quantization). See Appendix A
for more.

## Architecture overview

This section describes the pieces that every stage of the long-term plan
shares. Stages 1, 2, and 3 in §6–§8 below extend or refine specific aspects;
nothing in this section ever goes away.

```text
        ┌────────────────────────────────────────────────────────┐
        │                User-facing API surface                 │
        │                                                        │
        │   Vector<F, d>  ──TQEncode──▶  Extension<TurboQuant>   │
        │                 ◀──TQDecode──                          │
        └────────────────────────────────────────────────────────┘
                                  │
                                  │  ExtVTable
                                  ▼
        ┌────────────────────────────────────────────────────────┐
        │  Extension<TurboQuant>                                 │
        │     metadata (prost): {element_ptype, dimensions,      │
        │       bit_width, seed, num_rounds, [block_size]}       │
        │                                                        │
        │     storage:  Struct {                                 │
        │                 norms:  Primitive<F>     (Stage 1)     │
        │                       | FSL<F, num_blocks> (Stage 2 k>1)│
        │                 codes:  FSL<u8, padded_dim>            │
        │                       | FSL<u8, num_blocks*block_size> │
        │                       | PDXArray<u8, ...> (Stage 3)    │
        │               }                                        │
        └────────────────────────────────────────────────────────┘
                  ▲                                ▲
                  │                                │
                  │ derived (not stored)           │ derived (not stored)
                  │                                │
        ┌─────────┴────────┐              ┌────────┴──────────┐
        │ SorfMatrix       │              │ Centroids + S     │
        │ (seed, rounds,   │              │ (block_dim,       │
        │  padded_dim)     │              │  bit_width)       │
        │ via SplitMix64   │              │ cached in DashMap │
        └──────────────────┘              └───────────────────┘
```

### Extension type

```text
ExtId: vortex.turboquant
ExtVTable: TurboQuant
  Metadata = TurboQuantMetadata {
    element_ptype: PType,     // f16, f32, or f64
    dimensions:    u32,       // original, pre-padding (≥ 128)
    bit_width:     u8,        // 1..=8
    seed:          u64,       // SORF derivation
    num_rounds:    u8,        // SORF rounds (default 3)
    // Stage 2 addition:
    block_size:    Option<u32>,  // None == Stage 1 (= padded_dim, derived)
  }
```

Metadata is prost-serialized; adding fields with `Option<T>` or default values
is backward-compatible by design (see §13).

### Storage shape

Storage lives under the extension dtype as a `Struct` with row-aligned
validity across all fields. The struct grows monotonically across stages:

```text
Stage 1 (block_size implicitly = padded_dim, num_blocks = 1):
  Struct {
    norms: Primitive<element_ptype, row_validity>,
    codes: FixedSizeList<u8, padded_dim, row_validity>,
  }

Stage 2 with num_blocks > 1:
  Struct {
    norms: FixedSizeList<element_ptype, num_blocks, row_validity>,
    codes: FixedSizeList<u8, num_blocks * block_size, row_validity>,
  }
```

In Stage 2 with `num_blocks == 1`, the storage shape is identical to Stage 1
(a single-element norm list would be a Vortex-typing anomaly; we keep
`Primitive` in that case for wire-format compatibility). See §7 for the open
question on whether to unify to `FixedSizeList<element_ptype, 1>` uniformly.

### Scalar functions

The encode/decode contract is a pair of scalar functions registered on the
session:

```rust
// Eager encode: wraps a Vector column into a TurboQuant extension array.
// Not a ScalarFnArrayVTable — persisting unexecuted would write the original
// Vector, which defeats the lossy intent.
pub fn TQEncode::try_new_array(
    child: ArrayRef,                  // a Vector extension array
    config: &TurboQuantConfig,        // bit_width, seed, num_rounds, ...
) -> VortexResult<ScalarFnArray>;

// Lazy decode: ScalarFnArray that yields a Vector when executed.
// This is the sole path back to floats from a TurboQuant array.
pub fn TQDecode::try_new_array(
    child: ArrayRef,                  // a TurboQuant extension array
) -> VortexResult<ScalarFnArray>;
```

The contract:

- `TQEncode(v, cfg)` is **lossy by definition**: encoded values are not exact
  representations of `v`, only approximations within Theorem 1's MSE bound.
- `TQDecode(TQEncode(v, cfg)) ≈ v` within the MSE bound; not equal in general.
- There is no `canonicalize(tq_array)` path. Implementations that want a
  `Vector` array invoke `TQDecode` explicitly.

### Derived state, not stored

Two design choices keep the storage minimal:

- **SORF derived from `(seed, num_rounds, padded_dim)`.** The rotation matrix
  is reconstructed at encode and decode time from Vortex's frozen local
  SplitMix64 sign stream. No rotation parameters are stored in the array.
- **Centroids derived from `(block_dim, bit_width)`** (where `block_dim ==
padded_dim` in Stage 1 and `block_dim == block_size` in Stage 2 with k>1).
  Computed once via numerical integration of the Beta marginal and Max-Lloyd
  refinement; cached process-locally in a `DashMap`. With EDEN's optimized
  scale `S` (§6), the cache stores `(centroids, S)` together under the same
  key.

Both derivations are deterministic in their metadata inputs, so files written
by one process are readable by any other process with the same Vortex/EDEN
constants. The frozen-PRNG choice (SplitMix64) is part of Vortex's stable
contract; changing it would be a wire-format break.

### Crate boundary

`vortex-turboquant` is a standalone crate outside the main `vortex`
dependency tree. It depends on `vortex-tensor` (for the `Vector` extension
type that `TQEncode` consumes) and `vortex-array` (for the extension-type
machinery), but **nothing in the main `vortex` crate depends on
`vortex-turboquant`**. This crate is positioned as a model third-party
extension: users (or Vortex consumers like duckdb-vortex) opt in by calling

```rust
vortex_tensor::initialize(&session);
vortex_turboquant::initialize(&session);
```

The ordering matters today — the `Vector` parent extension type must be
registered before the TurboQuant child extension type can deserialize. Issue
#7830 notes this as a TODO to resolve via session-level dependency declaration.

### Default compressor behavior

The default cascade compressor (BtrBlocks) does not know about TurboQuant.
When it encounters an extension-typed `Struct`, it recurses into the struct
fields and applies its usual scheme matching:

- `norms` are typically compressed by ALP or Pco.
- `codes` are typically left alone at the chosen bit width (already
  near-entropy-limited; BitPacked is the obvious candidate but the default
  scheme often chooses identity).

The user is the source of truth for whether a column is lossy — they invoke
`TQEncode` explicitly, and the resulting extension type is preserved through
the rest of the write path. The compressor never recompresses lossy data and
never silently introduces lossiness.

## Stage 1: single-block, biased (MSE-only)

Stage 1 is the production-ready baseline. It implements TurboQuant as a
single-block extension type with biased (MSE-only) quantization, plus EDEN's
optimized scalar scale `S` as a strict refinement over TurboQuant's
fixed `S = 1`.

### Storage shape (recap)

```text
Extension<TurboQuant>(
  Struct {
    norms: Primitive<element_ptype>,
    codes: FixedSizeList<u8, padded_dim>,
  }
)

Metadata (in the extension dtype):
  element_ptype: PType (f16, f32, or f64)
  dimensions:    u32   (original, pre-padding)
  bit_width:     u8    (1..=8; default 8)
  seed:          u64   (frozen-PRNG seed for SORF)
  num_rounds:    u8    (default 3)
  block_size:    None  (Stage 1 implicit value = padded_dim)

padded_dim = next_power_of_two(dimensions)
```

### Encode path

`TQEncode(v: Vector, cfg)` for each row:

1. Compute `n = ‖v‖`. Store in `norms[i]`.
2. If `n > 0`: compute `û = v / n`. Pad `û` to `padded_dim` with zeros if
   `dimensions` is not a power of 2. Apply SORF (3-round Walsh–Hadamard with
   SplitMix64 sign diagonals derived from `seed`). Result is `r ∈ ℝ^padded_dim`.
3. Quantize each coordinate of `r`: `codes[i][j] = nearest_centroid(r[j] * S,
centroids)` where `S` is EDEN's optimized scale for `(padded_dim, bit_width)`.
4. If `n == 0`: store zero codes; mark validity according to input.

### Decode path

`TQDecode(tq)` for each row:

1. Lookup `centroids` and `S` from the cache by `(padded_dim, bit_width)`.
2. Dequantize: `r̂[j] = centroids[codes[i][j]] / S`.
3. Apply inverse SORF (same seed; the transform is self-inverse up to a sign).
4. Truncate to `dimensions` (drop padded zeros).
5. Re-apply norm: `v̂ = norms[i] * û`.

Decoded vectors are **not** guaranteed to have unit norm after roundtrip:
scalar quantization plus inverse SORF is not norm-preserving in general. See
§9 "Lossy semantics in practice."

### Stage 1 refinement: EDEN's optimized `S`

The current `vortex-turboquant` implementation uses TurboQuant's fixed `S = 1`.
Per EDEN [15], the optimal `S` is a function of `(padded_dim, bit_width)` and
converges to 1 only as dimension grows; the note [14] catalogues this gap as
the primary algorithmic suboptimality in TurboQuant_mse. At practical
dimensions, EDEN's `S` strictly reduces MSE at fixed bit budget. We adopt it
as a Stage 1 refinement:

- Compute `S` alongside the centroids at the same point in the algorithm
  (after Max-Lloyd converges) using EDEN's optimization criterion. The
  implementer should consult EDEN [15] for the precise criterion — the note
  [14] defers to "methods described in the EDEN works" and does not
  reproduce the algorithm itself. Reference implementation:
  https://github.com/amitport/EDEN-Distributed-Mean-Estimation (MIT;
  PyTorch and TensorFlow).
- Cache `(centroids, S)` together under the existing `(padded_dim,
bit_width)` key in the `DashMap`.
- Apply `S` at quantization time (encode: scale `r * S` before
  `nearest_centroid`; decode: scale `centroids[c] / S` after lookup).
- **No storage-shape change.** No metadata change for biased mode. The scale
  is reproducible from `(padded_dim, bit_width)`. If a future stage adds
  EDEN's unbiased mode, an `unbiased: bool` metadata flag is added and the
  cache key extends to `(padded_dim, bit_width, biased)`; see §15.

EDEN-`S` is a strictly-additive improvement. Files written before the EDEN-`S`
upgrade and files written after are wire-format-compatible, but they decode to
slightly different float values. We treat this as acceptable for an
experimental/preview feature; production stabilization requires either (a)
pinning EDEN's `S` table as part of Vortex's stable constants alongside the
SplitMix64 stream, or (b) versioning the centroid/scale algorithm in metadata
and selecting at decode time. We recommend (a) — see §13 "Migration."

### Defaults and configuration

- **Default bit_width: 8.** Near-lossless: normalized MSE ~4e-5; ~4×
  compression on f32. Safer than aggressive defaults for general use.
- **Default num_rounds: 3.** Sufficient SORF mixing at padded_dim ≥ 128.
- **MIN_DIMENSION = 128.** Hard-enforced in `validate_tq_metadata` (see
  "Minimum dimension" below).
- **MAX_BIT_WIDTH = 8.** Hardware-friendly (u8 codes); higher bit widths give
  diminishing returns vs. lossless schemes.

```rust
pub struct TurboQuantConfig {
    pub bit_width: u8,            // 1..=8
    pub seed: Option<u64>,        // default 42 (or session-configurable)
    pub num_rounds: u8,           // default 3
}
```

### Power-of-2 padding

SORF requires power-of-2 input dimension. Non-power-of-2 dimensions are
zero-padded internally (e.g., 768 → 1024). For non-power-of-2 dimensions
this gives:

- **33% storage overhead** for 768-dim vectors: 1024 codes stored vs. 768
  useful.
- **No scan-optimized layout**: row-major codes do not vectorize cleanly.

Both of these motivate Stage 2 (block decomposition) and Stage 3 (PDX) below.

### Minimum dimension

The scheme requires `dimensions ≥ 128`. Below this threshold, several factors
degrade quality and efficiency:

- **SORF mixing quality.** 3-round SORF at d=64 provides only 18 butterfly
  stages (vs. 21 at d=128, 30 at d=1024). The coordinate distribution
  deviates more from the analytical Beta, making Max-Lloyd centroids less
  optimal.
- **Practical MSE.** At smaller d, the SORF mixing quality and
  coordinate-independence approximations are weaker, potentially worsening
  practical quantization quality beyond what the dimension-free theoretical
  bound captures.
- **Overhead ratio.** Per-vector norm (32 bits) is a larger fraction of the
  compressed representation at small d. At d=32, b=5: codes=160 bits,
  norm=32 bits, total=192 — norm is ~17% of compressed size. At d=768: <1%.

The threshold of 128 is conservative; the experimental plan should determine
the true minimum (likely in the 64–128 range). Padding modest amounts (e.g.,
96 → 128) is probably acceptable; padding large fractions (e.g., 32 → 64) is
not.

### Compression ratio (Stage 1, f32 input)

For f32 input, b bits per code, N vectors:

| Component | Bits per vector |
| --------- | --------------- |
| Codes     | padded_dim × b  |
| Norm      | 32              |

| Component (shared) | Bits                         |
| ------------------ | ---------------------------- |
| Centroids          | (cached, not stored on disk) |
| SORF signs         | (derived, not stored)        |

Worked examples at b=8 (default, near-lossless):

| d    | padded_dim | Per-vec bits       | Ratio | Notes                |
| ---- | ---------- | ------------------ | ----- | -------------------- |
| 768  | 1024       | 1024×8 + 32 = 8224 | 3.0×  | Padded; 33% overhead |
| 1024 | 1024       | 1024×8 + 32 = 8224 | 4.0×  | No padding           |

For d=1024 (power-of-2), Stage 1 already achieves the full 4× ratio. For
d=768, Stage 1 leaves 33% on the table; Stage 2 (see §7) recovers it.

### Stage 1 known gaps

Tracked in issue #7830 and from the codex review of PR #7829:

- **Lazy `TQEncode` write path is not wired.** `vortex_turboquant::initialize()`
  registers `TQDecode` but not `TQEncode`'s lazy variant. File tests today
  write already-executed arrays, masking the gap. (P1 in PR #7829 review.)
- **`initialize()` not self-contained.** Requires
  `vortex_tensor::initialize()` first to register the `Vector` parent extension.
  Sessions that only initialize TurboQuant cannot deserialize files with
  `Vector` fields. (P2 in PR #7829 review.)
- **SORF dimension padding panics on oversized dims.** `tq_padded_dim()` uses
  unchecked `next_power_of_two()`. Should use the checked version and validate
  in `validate_sorf_options`. (P2 in PR #7829 review.)
- **EDEN-`S` not yet adopted** (this RFC's recommendation).
- **Pluggable scalar functions for TurboQuant-aware similarity not yet
  designed.** Currently `cosine_similarity` and `inner_product` must fall back
  to `TQDecode → compute on floats`; pushdown kernels that operate directly on
  codes are part of Stage 2 work (the per-block weighted-sum form generalizes
  across stages).

## Stage 2: block decomposition

Stage 2 eliminates power-of-2 padding for non-power-of-2 dimensions by
splitting the vector into independently-encoded blocks. The extension-type
semantic model and the encode/decode boundary are unchanged; only the
storage shape and the per-block normalization differ.

### Block size strategy

For each dimension `d`, choose `B` = the greatest power-of-2 ≥ 64 that
evenly divides `d`. If no such `B` exists (e.g., `d = 96`), fall back to
Stage 1 single-block padded encoding.

| Dimension d | Block size B | Blocks k | Notes                        |
| ----------- | ------------ | -------- | ---------------------------- |
| 512         | 512          | 1        | Single block (= Stage 1)     |
| 768         | 256          | 3        | Greatest dividing power-of-2 |
| 1024        | 1024         | 1        | Single block                 |
| 1536        | 512          | 3        |                              |
| 2048        | 2048         | 1        | Single block                 |
| 3072        | 1024         | 3        |                              |
| 4096        | 4096         | 1        | Single block                 |

Key observations:

- **Power-of-2 dimensions** (512, 1024, 2048, 4096) use `B = d` — a single
  block, identical to Stage 1 with `block_size = padded_dim`. No
  decomposition overhead.
- **Non-power-of-2 dimensions** (768, 1536, 3072) decompose into k=3 blocks
  at B=256 or B=512. No padding waste.
- **No qualifying B is rare** for common embedding dimensions. Dimensions
  where no power-of-2 ≥ 64 divides d (e.g., 96, 100) fall back to Stage 1
  padded single-block encoding.

### Metadata addition

```rust
pub struct TurboQuantMetadata {
    // ... unchanged Stage 1 fields ...
    pub block_size: Option<u32>,    // None == Stage 1 (= padded_dim)
}
```

`block_size` is added as a prost-optional field. Stage 1 readers see `None`
and treat the array as a single padded block; Stage 2 readers see `Some(B)`
and use the block decomposition path. The wire format is forward-compatible:
a Stage 1 file is exactly a Stage 2 file with `block_size = None`.

`num_blocks = ceil(dimensions / block_size)` is derived, not stored.

### Storage shape (k > 1)

```text
Extension<TurboQuant>(
  Struct {
    norms: FixedSizeList<element_ptype, num_blocks>,
    codes: FixedSizeList<u8, num_blocks * block_size>,
  }
)
```

For `k > 1`, the norms field becomes a fixed-size list of per-block norms.
The codes field stays a single `FixedSizeList<u8, ...>` with `list_size = k *
block_size`; the block boundary is implicit (block `b`'s codes are at offsets
`[b * B, (b+1) * B)`).

For `k == 1` (power-of-2 dimensions), the storage shape stays identical to
Stage 1 (single `Primitive<element_ptype>` norm). This preserves wire-format
compatibility for the common power-of-2 case — a Stage 1 writer and a Stage 2
writer produce bit-identical files at `d = 1024`.

**Open question (see §16):** Whether to unify to `FixedSizeList<element_ptype,
num_blocks>` for all k (including k=1). The current recommendation is to
preserve the Stage 1 single-`Primitive` form when k=1 for wire compatibility,
but the simplicity argument for unification is real.

### Per-block rotation

Each block has an independent SORF rotation. To keep the metadata small, the
seed for block `b` is derived from the array's single `seed` field by mixing
in the block index: `block_seed(b) = SplitMix64(seed ^ (b as u64))`. This
preserves determinism from the single stored seed and avoids storing per-block
seeds.

### Centroids: shared across blocks

All blocks at the same `(block_size, bit_width)` share a single centroid
codebook (and a single EDEN `S`). The process-local centroid cache is keyed
on `(block_size, bit_width)` rather than `(padded_dim, bit_width)`, so
multiple TurboQuant columns at different block sizes share cache lines naturally.

Why one codebook per `(block_size, bit_width)` is optimal: each block, after
its own SORF, has coordinates with the same Beta marginal distribution
`(1 - x²)^((B-3)/2)`. The Max-Lloyd grid is therefore identical for all
blocks. Per-block codebooks would just duplicate the same numbers.

### Encode and decode

The Stage 2 encode/decode pseudocode is in Appendix D.6 (encode) and is
structurally symmetric for decode (per-block dequantize via shared
centroids, per-block inverse SORF with `block_seed(seed, i)`, multiply by
`norms[row][i]`, concat). The key changes from Stage 1: split into k
blocks, store per-block norms, derive per-block SORF seeds from a single
stored seed, key the centroid cache on `block_size`.

### Quantized-domain operations

The pushdown forms generalize cleanly across stages. Let `unit_dot_k(a, b)`
denote the unit-vector dot product on block `k`'s codes (a sum over `B`
centroid products):

- **L2 distance:** `‖a-b‖² = Σ_k ‖a_k‖² + Σ_k ‖b_k‖² - 2·Σ_k ‖a_k‖·‖b_k‖·unit_dot_k`
- **Dot product:** `<a,b> ≈ Σ_k ‖a_k‖·‖b_k‖ · unit_dot_k`
- **Cosine similarity:** `cos(a,b) ≈ <a,b> / (‖a‖·‖b‖)` where
  `‖a‖ = √(Σ_k ‖a_k‖²)`
- **L2 norm:** `√(Σ_k ‖n_k‖²)` — O(k); a Stage 1 vs. Stage 2 regression
  from O(1) but modest at common k (≤ 3).

Both operands must have the same `(bit_width, block_size, seed, num_rounds)`
for the quantized-domain path to be valid. Stage 1 and Stage 2 arrays with
`block_size = padded_dim` and `num_blocks = 1` are equivalent for these
operations.

### Compression ratio (Stage 2)

For f32 input, b bits per code, k blocks:

| Component   | Bits per vector |
| ----------- | --------------- |
| Codes       | k × B × b       |
| Block norms | k × 32          |

Worked examples at b=8:

| d    | B    | k   | Per-vec bits             | Ratio | vs. Stage 1           |
| ---- | ---- | --- | ------------------------ | ----- | --------------------- |
| 768  | 256  | 3   | 3×256×8 + 3×32 = 6240    | 3.9×  | +30% over 3.0× padded |
| 768  | 1024 | 1   | (Stage 1 fallback: 8224) | 3.0×  | (no change)           |
| 1024 | 1024 | 1   | (Stage 1: 8224)          | 4.0×  | (no change)           |
| 1536 | 512  | 3   | 3×512×8 + 3×32 = 12384   | 4.0×  | +33% over 3.0× padded |

For non-power-of-2 dimensions, Stage 2 recovers the padding overhead almost
entirely. For power-of-2 dimensions, Stage 2 is identical to Stage 1.

### Empirical evidence from small-block rotations

The RotorQuant/IsoQuant experiments [13] provide direct evidence of the
decorrelation failure mode at very small block sizes: block-diagonal
rotations in SO(3) (3-dim groups) and SO(4) (4-dim groups) caused 10× MSE
regressions on real KV-cache vectors, attributed to the complete absence of
cross-group coordinate mixing.

Our Stage 2 design operates at a fundamentally different scale — B=256
blocks with 3-round SORF provide 24 butterfly mixing stages within each
block, vs. RotorQuant's 3-4 raw coordinates with no structured mixing. The
decorrelation loss should therefore be far less severe. Nevertheless, the
experimental plan (§12) includes explicit cross-block correlation
measurement on real embeddings to quantify any residual decorrelation gap.

### Zero-norm sub-vectors

When splitting into B-dim blocks, some blocks may have zero norm. The
encoding handles `‖v_k‖ = 0` explicitly: skip rotation and quantization,
store `norm = 0`, decode as all zeros for that block. This is unchanged
from Stage 1 behavior and preserves correctness when a vector has
zero-energy regions.

### Straggler blocks (future direction)

The current block-size rule requires B to evenly divide d. A natural
extension is **straggler blocks**: allow `k` blocks where `k-1` are full-size
B and the final block covers the remaining `d - (k-1)·B` dimensions, possibly
with a different encoding (raw float, padded TQ, or scalar quantization).

This is **deferred beyond Stage 2** because the basic block-size rule
already handles all common embedding dimensions (768, 1024, 1536, 3072,
etc.) without stragglers. Rare cases (d=96, d=800) fall back to Stage 1
padded encoding for now. See §15 "Future work" for the design sketch.

## Stage 3: PDX physical layout

Stage 3 introduces a separate physical encoding (`PDXArray`) for the codes
child, enabling SIMD scan kernels. The TurboQuant extension type is
unchanged; the codes child's logical dtype is unchanged; only its physical
layout changes.

**Terminology note.** Throughout this section, "chunk" refers to the PDX
group of 64 vectors over which the dimension-major transpose operates.
"Block" continues to refer to the TurboQuant Stage 2 block of `block_size`
coordinates. PDX's chunks and TurboQuant's blocks are orthogonal: a single
PDX chunk spans 64 rows and includes the full `k × block_size` codes per
row, while a single TurboQuant block spans `block_size` codes per row
across all rows.

### PDX background

PDX [4] is a data layout for vector similarity search. The SIGMOD '25 paper
describes a dimension-major layout within fixed-size chunks of 64 vectors,
enabling the compiler to auto-vectorize the inner distance loop over vectors
rather than dimensions. The paper reports an average 2× speedup for
auto-vectorized PDX distance kernels vs. explicitly SIMD-optimized row-major
baselines (SimSIMD, FAISS) across four architectures, with larger gains at
low dimensionality (5.5× at D ≤ 32) and ~1.5× at D > 32 [4, Table 4]. The
chunk size of 64 is empirically optimal across AVX-512, AVX2, and NEON
architectures [4, Table 5].

### PDX as a physical encoding of FSL

The key insight under the extension-type model: **PDX is a physical encoding
of `FixedSizeList<T>`, not a TurboQuant variant.** Register `PDXArray<T>` as
a separate encoding alongside the existing `FixedSizeListArray` encoding;
both have the same logical dtype (`FixedSizeList<T, list_size>`) and produce
the same scalar values, but they differ in their on-disk and in-memory
layout.

```text
FixedSizeListArray (row-major, current default):
  [row 0: dim 0, dim 1, ..., dim B-1]
  [row 1: dim 0, dim 1, ..., dim B-1]
  ...

PDXArray<T> (dimension-major within 64-row chunks):
  chunk 0 (rows 0..64):
    dim 0: [v0 v1 v2 ... v63]
    dim 1: [v0 v1 v2 ... v63]
    ...
    dim B-1: [v0 v1 v2 ... v63]
  chunk 1 (rows 64..128):
    dim 0: [v0 v1 v2 ... v63]
    ...
```

Both encodings encode the same logical FSL data; converting between them is a
transpose operation.

### TurboQuant interaction

A TurboQuant array's codes child can be either layout:

- **Default**: codes are a `FixedSizeListArray<u8, padded_dim>` (Stage 1) or
  `FixedSizeListArray<u8, k*B>` (Stage 2).
- **PDX-enabled**: codes are a `PDXArray<u8>` of the same logical FSL dtype.

The TurboQuant `ExtVTable::validate_dtype` cares only about the logical FSL
dtype, not the physical encoding. The extension type does not need an
"is_pdx" metadata flag.

### Scalar function dispatch

The scalar function implementations for `cosine_similarity`, `inner_product`,
and `l2_distance` on TurboQuant inputs inspect the codes child's encoding at
execution time:

```rust
match codes_child.encoding() {
    FSL_ENCODING => fallback_row_major_kernel(...),
    PDX_ENCODING => simd_pdx_kernel(...),
    _ => decode_then_compute(...),
}
```

For non-aligned slice/take or `scalar_at`, the PDXArray is converted to
`FixedSizeListArray` first via a transpose. The transpose cost is `O(rows ×
B)` per operation; consider 64-row-aligned fast paths for hot scan workloads.

### PDXArray design

```text
PDXArray<T> (general-purpose dimension-major layout for FixedSizeList):
  metadata: { list_size, chunk_size (default 64) }
  elements: PrimitiveArray<T>   // transposed: 64 values per dim, contiguous
  validity: ...                  // same as FSL validity
```

- `PDXArray::try_new(fsl) → PDXArray` — transposes a `FixedSizeListArray`
  into PDX layout.
- `PDXArray::to_fsl() → FixedSizeListArray` — un-transposes back to row-major
  FSL for decode, `scalar_at`, or non-aligned slice/take.
- `PDXArray::elements_for_dim(dim, chunk) → &[T]` — O(1) access to a
  contiguous slice of 64 values for one dimension within one chunk.
- The cascade compressor treats PDXArray as a valid encoding of FSL-typed
  data; user code declares preference for PDX layout via Vortex's scheme
  selection.

### Quantized distance kernel (dot product, b=4)

The kernel structure: for each `(tq_block, dim)` pair, fetch the query
code's row from a precomputed `(2^b)²` centroid distance table (1 KB at
b=4), then sum the 64-vector lane in a SIMD-friendly inner loop with no
inter-vector dependencies. After each TQ block, weight the 64 per-block
unit-norm dot products by `query_norms[tq_block] · data_norms[v][tq_block]`.

Full Rust pseudocode is in Appendix D.7. The salient property for design
review: the inner SIMD loop is purely data-parallel across 64 vectors and
contains no TQ-block bookkeeping; TQ block boundaries only affect where
norm weighting occurs, not the transpose.

### Where PDXArray lives

PDX is general-purpose; it can accelerate scans over any FSL-typed data
(raw float vectors, scalar-quantized vectors, future encodings). The
**recommended location is `vortex-array`** (or a sibling general-purpose
crate) rather than `vortex-turboquant`. This:

- Lets non-TurboQuant FSL columns benefit from PDX layout.
- Keeps `vortex-turboquant` focused on its lossy-extension-type role.
- Allows the cascade compressor to PDX-transpose any FSL child when a
  scan-heavy workload is detected (a future automation).

### Int8 layout variant

The open-source PDX implementation [pdx-impl] uses a different tiling for
int8 data: "4 dims × 16 vecs" to leverage VPDPBUSD/UDOT hardware
dot-product instructions (which process 4 unsigned×signed byte pairs per
operation). For TurboQuant codes at b ≤ 8, codes are uint8 centroid indices,
so VPDPBUSD doesn't apply directly — we need the distance-table-lookup path
above. Whether a "4 dims × 16 vecs" variant for hardware dot-product on
near-linear centroids at b=8 (Beta concentrates to Gaussian at high d) is
viable is an empirical question deferred to Stage 3 evaluation.

[pdx-impl]: https://github.com/cwida/PDX

## Lossy semantics in practice

TurboQuant is fundamentally different from Vortex's other arrays because its
values are lossy. This section catalogs the practical surprises users hit
when working with a TurboQuant column.

### Decoded vectors are not unit-norm after roundtrip

`TQDecode(TQEncode(v))` re-applies the stored norm to a quantized unit
direction. Scalar quantization plus inverse SORF is not norm-preserving in
general, so the decoded vector's L2 norm may differ from the original by a
small amount within the MSE bound. Code that depends on exact unit norms
must explicitly renormalize after `TQDecode`.

### `TQDecode(TQEncode(v))` ≠ `v` exactly

Equal only within Theorem 1's MSE bound. Code that expects exact roundtrip
identity must not use TurboQuant. The current example
(`vortex/examples/turboquant_vector_search.rs`) explicitly compares decoded
vs. original with a per-element tolerance and a max-abs-diff regression
ceiling.

### The default compressor doesn't recompress TurboQuant

It also doesn't know the data is lossy. Users explicitly opt into lossiness
by invoking `TQEncode`; once the column is TurboQuant-typed, the cascade
compressor sees a `Struct` and recurses without trying to introduce
additional compression beyond what already fits the codes/norms storage.

### Pushdown-aware scalar functions vs. fallback

`cosine_similarity`, `inner_product`, and `l2_distance` over a TurboQuant
column have two execution paths:

- **Quantized-domain path** (preferred): operates directly on codes and
  norms without decoding. Per-block weighted sum (Stage 2 form); SIMD on PDX
  codes (Stage 3 form).
- **Fallback**: `TQDecode → compute on floats`. Always correct but
  significantly slower.

The quantized-domain path requires both operands to have the same
`(bit_width, block_size, seed, num_rounds)`. The common case (column vs.
constant query vector) sidesteps this by re-encoding the query at query time
with the column's metadata.

### Null-vector handling

Null vectors are handled via row-aligned validity on both the norms and
codes fields. `TQEncode` zeros invalid rows (stores `norm = 0` and zero
codes). `TQDecode` returns nulls at the same row positions.

### Cross-column compatibility

Two TurboQuant columns with different `(dimensions, bit_width, block_size,
seed, num_rounds)` cannot share the quantized-domain operations — at least
one side must be decoded. In practice, columns written by the same pipeline
share the same config; cross-column comparisons across schemas are rare.

## Integration with Vortex

This section describes the end-to-end user flow with a TurboQuant column.
The canonical reference is
[`vortex/examples/turboquant_vector_search.rs`][example] in the Vortex
repo, which exercises both the write path and a filter-pushdown read.

[example]: https://github.com/spiraldb/vortex/blob/ff120401a0f4796f2d1aa85d1f87e7195c1f3dbf/vortex/examples/turboquant_vector_search.rs

**Current vs. target API.** The example today wires TurboQuant through the
older `vortex-tensor` `L2Denorm + SorfTransform` decomposition (see §13
"Predecessor implementations"); a Stage 1 stabilization task migrates it
to the `vortex-turboquant` crate's `TQEncode` / `TQDecode`. The snippets
below describe the target API. Where a real symbol from the current
example carries over to the target, the snippet uses the real name;
symbols marked `// target` are the names this RFC commits to once the
migration lands.

### Session setup

```rust
use vortex::session::VortexSession;

let session = VortexSession::default().with_tokio();
vortex_tensor::initialize(&session);       // register Vector (and dependencies)
vortex_turboquant::initialize(&session);   // register vortex.turboquant + TQEncode/TQDecode
let mut ctx = session.create_execution_ctx();
```

`vortex_tensor::initialize` must run before `vortex_turboquant::initialize`
today. The ordering requirement is tracked as the P2 finding from PR
#7829's review (see §14 "Current state and known gaps").

### Write path (target API, post-migration)

The user has two options. The recommended **opt-in compressor path** wires
TurboQuant into the cascade compressor — once enabled, the compressor
matches `Vector<float, d>` columns with `d ≥ MIN_DIMENSION` and applies
TurboQuant to them:

```rust
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex::file::WriteStrategyBuilder;

let compressor = BtrBlocksCompressorBuilder::default()
    .with_turboquant()              // target: opts in to TurboQuantScheme on Vector columns
    .build();

let strategy = WriteStrategyBuilder::default()
    .with_compressor(compressor)
    .build();

let mut buf = vortex::buffer::ByteBufferMut::empty();
session
    .write_options()
    .with_strategy(strategy)
    .write(&mut buf, struct_array.to_array_stream())
    .await?;
let bytes = buf.freeze();
```

The **explicit-encode path** uses `TQEncode` as a scalar function. Useful
when the user wants per-column control or wants the lossy mutation to be
visible in their pipeline as an explicit operator:

```rust
use vortex_turboquant::{TQEncode, TurboQuantConfig};

let cfg = TurboQuantConfig::try_new(/* bit_width */ 8, /* seed */ 42, /* num_rounds */ 3)?;
let emb_tq = TQEncode::try_new_array(emb_vector, &cfg)?;   // ScalarFnArray, lazy
let emb_tq_materialized = emb_tq.execute(&mut ctx)?;
```

Either path produces an `Extension<vortex.turboquant>` array that can be
nested in a `StructArray`, written to a Vortex file, and read back as a
TurboQuant column.

### Read path (plain scan)

```rust
use futures::TryStreamExt;

let chunks: Vec<vortex::array::ArrayRef> = session
    .open_options()
    .open_buffer(bytes.clone())?
    .scan()?
    .into_array_stream()?
    .try_collect()
    .await?;
```

Reading yields TurboQuant-typed columns. Decoding back to a `Vector` is an
explicit step:

```rust
use vortex_turboquant::TQDecode;

let emb_tq = chunked_table.get_item("emb")?;
let emb_vector = TQDecode::try_new_array(emb_tq)?;          // ScalarFnArray, lazy
let emb_executed = emb_vector.execute(&mut ctx)?;           // ExtensionArray<Vector>
```

### Filter pushdown (cosine similarity)

The current example pushes `cosine_similarity(emb, query) > threshold`
through `scan().with_filter(...)` and selects rows inside the scan rather
than after materialization:

```rust
use vortex::array::expr::{col, gt, lit};
use vortex::array::scalar_fn::EmptyOptions;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;

let cosine_expr = CosineSimilarity.new_expr(EmptyOptions, [col("emb"), lit(query_scalar)]);
let filter = gt(cosine_expr, lit(0.85_f32));

let chunks: Vec<ArrayRef> = session
    .open_options()
    .open_buffer(bytes.clone())?
    .scan()?
    .with_filter(filter)
    .into_array_stream()?
    .try_collect()
    .await?;
```

`CosineSimilarity` over a TurboQuant column inspects the codes child's
physical encoding (Stage 3: `PDXArray` vs. `FixedSizeListArray`) and the
extension metadata (Stage 2: `block_size`). For Stage 1/2 with row-major
codes, it dispatches to the per-block weighted-sum kernel without
decoding. For Stage 3 with PDX codes, it dispatches to the SIMD kernel.
The constant query literal expands to a `ConstantArray` whose row count
matches the current batch; the kernel encodes the query once per batch
with the column's metadata.

## Performance

### Encode/decode throughput (Stage 1)

SORF at B dimensions (heuristic; real cost is dominated by memory
bandwidth and constant factors): `R · B · log₂(B)` butterflies + `R · B`
sign applications per block. The per-vector normalization multiplies
(`B` per block) are omitted from these counts; they amortize against
memory bandwidth in practice. For Stage 1 with
padded_dim and 3 rounds:

| padded_dim | SORF FLOPs                  | k   | Total per-vector FLOPs |
| ---------- | --------------------------- | --- | ---------------------- |
| 256        | 3×256×8 + 3×256 = 6,912     | 1   | 6,912                  |
| 512        | 3×512×9 + 3×512 = 15,360    | 1   | 15,360                 |
| 1024       | 3×1024×10 + 3×1024 = 33,792 | 1   | 33,792                 |

### Encode/decode throughput (Stage 2)

Block decomposition at d=768 (k=3, B=256) is ~38% fewer SORF FLOPs than
the padded single-block Stage 1 approach, despite more blocks, because each
block is smaller:

| d   | Stage 1 (padded) | Stage 2 (k=3, B=256) | Reduction |
| --- | ---------------- | -------------------- | --------- |
| 768 | 33,792 FLOPs     | 3 × 6,912 = 20,736   | ~39%      |

### Scan throughput (Stage 3 vs. Stage 2)

PDX gives an expected 1.5–2× speedup on row-major code scans for
embedding-scale dimensions (B ≥ 64) [4, Table 4]. Concrete throughput
numbers will come from the experimental plan; we expect AVX-512 to be most
favorable.

### Compression ratio recap (across stages)

| d    | Stage 1 padded | Stage 2 (k=3, B=256) | Stage 3 (PDX) |
| ---- | -------------- | -------------------- | ------------- |
| 768  | 3.0×           | 3.9×                 | 3.9× (same)   |
| 1024 | 4.0×           | 4.0× (k=1, same)     | 4.0× (same)   |

PDX changes layout, not size — Stage 3's compression ratio is identical to
Stage 2's at all dimensions. The win is in scan throughput.

## Practical recommendations

For common model dimensions, the most promising configurations are:

| Dimension             | Recommendation             | Rationale                                                               |
| --------------------- | -------------------------- | ----------------------------------------------------------------------- |
| 512, 1024, 2048, 4096 | Stage 1 + EDEN-S + Stage 3 | B=d, no decomposition needed. Same as Stage 1 but with PDX scan layout. |
| 768, 1536, 3072       | Stage 2 (k=3) + Stage 3    | B=256 or 512. No padding waste. 3 blocks, shared centroids.             |
| 96, 100, 800 (rare)   | Stage 1 padded             | Internal zero-padding to next power-of-2.                               |
| < 128                 | Not recommended            | Scheme minimum; SORF mixing quality and overhead ratio degrade.         |

All recommendations default to b=8 (near-lossless, ~4× compression). Users
who want more aggressive compression can drop to b=4 or b=5 with measurable
recall impact; the experimental plan should provide per-dataset guidance.

## Experimental plan

### Minimum dimension threshold

Test TurboQuant quality at d ∈ {32, 64, 96, 128, 256} to validate the scheme
minimum of 128:

- Compare TurboQuant MSE distortion and ANN recall@k against scalar
  quantization matched on **total compressed bits per vector** (codes + norm
  - amortized shared metadata), not just bits-per-coordinate.
- Plot the crossover point: at what d does TurboQuant's recall@k drop below
  the rate-matched scalar baseline?
- Test SORF coordinate distribution quality at each d (histogram vs. Beta).
- Measure overhead ratio (norm bits / total compressed bits) at each d.

### MSE quality and scan performance vs. block size

- Compare actual normalized MSE at B ∈ {64, 128, 256, 512} vs. single-block
  at full power-of-2 dimension, at bit widths b ∈ {2, 3, 4, 5, 8}.
- Compare ANN recall@k and scan throughput at fixed d (e.g., d=3072) across
  B ∈ {256, 512, 1024}.
- Test SORF coordinate distribution at each B: histogram vs. analytical Beta.
- Test 3, 4, 5 SORF rounds at each B (the SORF round count is part of the
  extension metadata; we can sweep it without changing the wire format).
- Measure cross-block coordinate correlation on real embeddings (Contriever,
  OpenAI) before and after per-block SORF rotation to quantify residual
  decorrelation gap.

### EDEN-`S` validation

- Compare normalized MSE at b ∈ {2, 3, 4, 5, 8} with TurboQuant's `S = 1`
  vs. EDEN's optimal `S` at fixed dimensions d ∈ {128, 256, 512, 768, 1024,
  1536}. We expect strict improvement at all `(d, b)` pairs.
- Compare ANN recall@k at the same configurations. The improvement should
  be largest at small `(d, b)` where the fixed-`S` suboptimality is most
  pronounced.
- Verify the EDEN-`S` table is reproducible from `(d, b)` alone (i.e., we
  can pin it as a Vortex constant rather than storing per-column).

### Gaussian-optimal vs. Beta-optimal grids

HIGGS [12] demonstrates that Gaussian-optimal grids (computed via CLVQ for
N(0,1)) work well after a single Hadamard rotation. Since the Beta marginal
converges to Gaussian at high d, test whether Gaussian grids can replace
Beta Max-Lloyd centroids for ANN search:

- **Grid comparison**: At B ∈ {64, 128, 256, 512} and b ∈ {2, 3, 4, 5, 8},
  compare ANN recall@k and normalized MSE for (a) Beta Max-Lloyd centroids
  at B-dim with EDEN-`S`, (b) Gaussian-optimal scalar grids (Normal Float
  style), and (c) CLVQ-computed Gaussian grids.
- **Rotation depth**: If Gaussian grids match Beta Max-Lloyd at a given B,
  test whether 1-round RHT achieves comparable quality to 3-round SORF.

### Benchmarking datasets

The current test suite uses i.i.d. Gaussian vectors as a theory anchor and
sanity check: for isotropic data, a random orthogonal transform is
distributionally neutral, which cleanly validates theoretical bounds. This
is not a universal "worst case" for all production workloads.

| Dataset                       | Dim    | Size   | Why                                           |
| ----------------------------- | ------ | ------ | --------------------------------------------- |
| Contriever                    | 768    | ~1M    | Key non-power-of-2 target; real embeddings    |
| OpenAI text-embedding-3-large | 1536   | ~1M    | High-d production embeddings (RAG workloads)  |
| SIFT                          | 128    | 1M     | Low-d power-of-2 baseline                     |
| arXiv embeddings              | 768    | 2.25M  | Same dim as Contriever, larger scale          |
| DEEP                          | 96     | 10M    | Large scale; below scheme min — explicit-only |
| Synthetic Gaussian            | varies | varies | Theory anchor / sanity check                  |

**Metrics** (at b ∈ {2, 3, 4, 5, 8}):

- Recall@10, Recall@100 (ANN ranking quality)
- Normalized MSE distortion (reconstruction quality)
- Inner product mean signed relative error (bias measurement)
- Encode/decode throughput (vectors/sec)

## Current state and known gaps

Mirroring issue [vortex-data/vortex#7830](https://github.com/vortex-data/vortex/issues/7830)'s
checklist, with explicit per-stage status:

### Stage 1 (single-block, biased)

- ✅ Initial prototype: PR #7269 (merged 2026-04-02)
- ✅ Decomposition into composable layers (then collapsed back): PR #7374
- ✅ Standalone crate refactor: PR #7829 (merged 2026-05-07)
- ✅ Basic benchmarking and `turboquant_vector_search` example
- ⚠️ Lazy `TQEncode` write path not wired (P1 from PR #7829 review)
- ⚠️ `initialize()` self-containment vs. `vortex_tensor::initialize()` ordering
  (P2 from PR #7829 review; tracked as TODO in `lib.rs`)
- ⚠️ SORF dimension validation panics on oversized dims; should error (P2 from
  PR #7829 review)
- ⏳ **EDEN-`S` adoption** (this RFC's recommendation; near-term work)
- ⏳ Example migration from `vortex-tensor` `L2Denorm + SorfTransform` to
  `vortex-turboquant`
- ⏳ Public API stabilization

### Stage 2 (block decomposition)

- ⏳ Not started. Requires:
  - `block_size` field added to `TurboQuantMetadata`
  - Norm storage shape change for `k > 1` (`FixedSizeList<element_ptype, k>`)
  - Block-aware encode and decode paths
  - Centroid cache key change from `(padded_dim, bit_width)` to
    `(block_size, bit_width)`
  - Quantized-domain operation kernel update for per-block weighted sums

### Stage 3 (PDX physical layout)

- ⏳ Not started. Requires:
  - New `PDXArray<T>` encoding registered in `vortex-array` (recommended) or
    a sibling general-purpose crate
  - `PDXArray::to_fsl()` / `PDXArray::try_new(fsl)` transpose primitives
  - Scalar function dispatch on codes child encoding type
  - SIMD distance kernel for `b ∈ {3, 4, 5}` (distance table fits L1)
  - Slice/take fast paths for 64-row-aligned ranges

### Cross-cutting

- ⏳ Pluggable scalar functions for TurboQuant-aware similarity (registration
  - pushdown integration)
- ⏳ Documentation and public API stabilization
- ⏳ Eventual removal of `vortex-tensor/src/encodings/turboquant/` once the
  example migrates

## Future work beyond the three stages

The three stages above are the long-term plan this RFC commits to. The
following are genuine extensions or alternatives that are out of scope for
this RFC, recorded here so they aren't forgotten.

### Unbiased mode: EDEN's native b-bit unbiased quantizer

If unbiased inner-product estimation is needed for a specific workload, the
preferred path is **EDEN's native b-bit unbiased mode** [15] rather than
TurboQuant's MSE+QJL "Prod" stacking. The note [14] reports experimental
results showing direct b-bit unbiased EDEN dominates `(b-1)`-bit MSE +
1-bit QJL stacking, often by more than a bit (e.g., 2-bit EDEN beats 3-bit
TurboQuant_prod).

Storage shape: same as Stage 1/2 — just a different scalar quantizer in
the encode/decode kernels. Metadata: an additional `unbiased: bool` flag in
`TurboQuantMetadata`.

QJL discussion is preserved in Appendix C for historical context; community
findings on QJL vs. MSE-only for KV-cache attention remain valid, but
EDEN's unbiased mode is strictly superior to TurboQuant_prod and should be
the default if unbiased estimation is added.

### Multi-dimensional vector quantization (p > 1)

HIGGS [12] demonstrates that vector quantization with dimension p > 1
(quantizing groups of p coordinates jointly to an optimal multi-dimensional
grid) achieves better rate-distortion than scalar quantization (p = 1) at
the same bit budget.

For TurboQuant, this would mean replacing the per-coordinate Max-Lloyd
centroid lookup with a per-subvector codebook lookup. At p=2 with b=4 bits
per coordinate, the codebook has 256 entries and the distance table becomes
256×256 = 64K entries (256 KB) — fits in L1/L2 but much larger than the
current 1 KB at b=4 scalar. At p=4, the table is infeasible; alternative
distance strategies (asymmetric distance computation, partial codebook
scans) would be needed.

Evaluate after Stage 3 is validated. Compare ANN recall@k at matched bit
budgets: p=1 at b bits vs. p=2 at b bits. If p=2 shows meaningful recall
improvement (>2% recall@10), design the kernel changes as a separate
extension.

### GPU decode and fused distance computation

The B-dim block structure maps naturally to GPU tile sizes and tensor
cores. For a batch of N vectors sharing the same rotation matrix `R⁻¹`:

```
decoded_batch = diag(norms) × R⁻¹ × codebook_lookup_batch(codes)
                                     ↑ B×N matrix
                              ↑ B×B × B×N = GEMM
```

The codebook gather + inverse rotation + norm scaling can be fused into a
single kernel using an IO-aware streaming pattern. For distance computation
without full decode, a precomputed (2^b)²-entry distance table fits in
shared memory at low bit widths (1 KB at b=4, 4 KB at b=5). At the default
b=8, the table is 256² × 4 = 256 KB, which exceeds typical GPU shared memory
(48–228 KB); the distance-table approach is therefore practical only at
b ≤ 5 on GPU, or requires tiling/streaming for b=8.

### Straggler blocks

For dimensions with no qualifying B ≥ 64 dividing d (e.g., d=96, d=800),
allow `k` blocks where `k-1` are full-size B and the final block covers the
remaining `d - (k-1)·B` dimensions with a different encoding (raw float,
exact-rotation TQ, or scalar quantization). The block decomposition
structure already supports heterogeneous child encodings; this is a
metadata extension.

Deferred until empirical evidence shows the rare-dimension case is worth
the complexity. Today's fallback (Stage 1 padded single-block) handles all
common embedding dimensions adequately.

### ADSampling-style dimension pruning

ADSampling [4] applies a random orthogonal rotation as preprocessing,
enabling dimension-by-dimension hypothesis testing for early scan
termination. It is complementary to TurboQuant's block structure: when
scanning with block decomposition, the pruner could skip entire TQ blocks
(B dimensions at a time) if the partial distance already exceeds the
candidate threshold. Sharing rotations between TurboQuant and ADSampling
is speculative under per-block rotations (Stage 2); see Appendix C.

## Migration and compatibility

TurboQuant has not been included in a release yet, so the wire format can
still change freely. The Stage 1 target wire format (with EDEN-`S`
adopted) is intended to be ready for backward-compatibility guarantees,
without formally committing to stability until confirmed by Stage 2
implementation and benchmarking.

### Strategy: single extension ID, additive metadata

All stages use the same extension ID (`vortex.turboquant`). The metadata is
a prost message; new fields are added as `optional` so that older readers
treat them as their default value.

Stage 1 → Stage 2: add `block_size: Option<u32>`. Stage 1 writers leave it
as `None`; Stage 1 readers see `None` and treat the array as a single
padded block. Stage 2 writers populate it when k > 1; Stage 2 readers
honor it.

Stage 2 → Stage 3: no metadata change. The codes child's physical encoding
shifts from `FixedSizeListArray` to `PDXArray`; readers detect via
encoding ID.

### Wire-format constants

`MIN_DIMENSION = 128`, `MAX_BIT_WIDTH = 8`, default `num_rounds = 3`, the
SplitMix64 PRNG, and (after Stage 1 stabilization) EDEN's `S` table are
all part of Vortex's stable contract. Changing any of them is a
wire-format break.

### Norms are always internal children

The TurboQuant array is self-contained — it stores norms as a child slot,
not in a parent encoding:

- Stage 1: norms child is `Primitive<element_ptype>`, one norm per vector.
- Stage 2 with k=1 (power-of-2 dims): same as Stage 1, identical wire format.
- Stage 2 with k>1: norms child is `FixedSizeList<element_ptype, k>`, k
  norms per vector.

The decoder distinguishes k=1 from k>1 by reading `block_size` from
metadata (or its `None` Stage 1 value).

### Stage 3 is additive

PDX is not a TurboQuant metadata flag — it's a separate encoding
(`PDXArray`) for the codes child. Stage 1/2 readers see `FixedSizeListArray`
codes and proceed as today; Stage 3 readers see `PDXArray` codes and
dispatch to the SIMD kernel. PDXArray itself is registered as a
general-purpose encoding, independent of TurboQuant.

### Predecessor implementations on `develop`

Two predecessor implementations exist on `vortex-data/vortex` `develop` and
should be removed once the user-facing example and downstream consumers
migrate:

- `vortex-tensor/src/encodings/turboquant/` — the monolithic-array form
  with separate `codes`, `norms`, `centroids`, and `rotation_signs` child
  slots. Used today by `vortex/examples/turboquant_vector_search.rs`.
- `vortex-tensor/src/scalar_fns/{l2_denorm,sorf_transform}` — the
  decomposed `ScalarFnArray(L2Denorm, [ScalarFnArray(SorfTransform, [FSL(Dict)]), norms])`
  form (PR #7374). Superseded by the extension-type model in `vortex-turboquant`.

Removal is a Stage 1 cleanup task once the example migrates.

### Incremental shipping

| Stage      | Ships to users? | Reads prior stage files?   | Notes                                              |
| ---------- | --------------- | -------------------------- | -------------------------------------------------- |
| 1 (MSE)    | Yes             | N/A (first preview)        | Single block, EDEN-`S`, biased only                |
| 2 (blocks) | Yes             | Yes (k=1 is identical)     | `block_size` metadata added; k>1 needs S2+ readers |
| 3 (PDX)    | Yes             | Yes (FSL codes still work) | PDX codes need `PDXArray` registered               |

Each stage is independently shippable. Users can upgrade incrementally.
Files written by earlier stages are always readable by later decoders.

## Drawbacks

The semantic surprises in §9 are the right starting point — TurboQuant is
materially different from a lossless encoding, and that difference is
inescapable. This section consolidates the drawback-class costs scattered
across the design so a reviewer can see them in one place:

- **Decoded vectors are not unit-norm.** `TQDecode(TQEncode(v))` does not
  preserve `‖v‖ = 1` even when the input was a unit vector; quantization
  plus inverse SORF is not norm-preserving. Code paths that assume unit
  norms must renormalize after decode. See §9.
- **`TQDecode(TQEncode(v)) ≠ v` exactly.** Equal only within Theorem 1's
  MSE bound (with the SORF-vs-Haar approximation gap on top). Any code
  that depends on roundtrip identity is incompatible with TurboQuant. See
  §9 and Appendix A.
- **L2 norm readthrough degrades O(1) → O(k) at Stage 2.** With per-block
  norms, computing the full-vector `‖v‖` requires summing k squared norms.
  At common k values (1–3) this is small but a real regression from the
  Stage 1 stored-norm fast path.
- **Two stored implementations during migration.** Until
  `vortex/examples/turboquant_vector_search.rs` and downstream consumers
  move to `vortex-turboquant`, the legacy `vortex-tensor` paths
  (`L2Denorm + SorfTransform` and the older monolithic-array encoding)
  must continue to deserialize. Dual maintenance cost until the cleanup
  in §14 completes.
- **Initialization is order-sensitive.** `vortex_tensor::initialize` must
  precede `vortex_turboquant::initialize` for the `Vector` parent
  extension to be registered. Sessions that omit this ordering fail at
  deserialization. Tracked as the P2 finding from PR #7829's review.
- **Quantized-domain operations only fuse for matching configs.** Two
  TurboQuant columns with different `(dimensions, bit_width, block_size,
seed, num_rounds)` cannot pushdown — at least one side must decode.
  Cross-column TurboQuant joins are therefore typically more expensive
  than they would be on a lossless column.
- **EDEN-`S` adds a centroid-cache lookup dimension.** Adopting EDEN's
  optimized `S` introduces a per-`(d, b)` constant the implementer must
  pin and ship alongside the centroid table. If EDEN's optimization
  criterion ever updates (e.g., a corrected version of the EDEN paper
  publishes), the constants need a wire-format-stable migration.
- **PDX layout doubles the maintenance surface for the codes child.**
  Stage 3 adds a second physical encoding (`PDXArray`) alongside
  `FixedSizeListArray`. Every scalar function operating on TurboQuant
  codes must dispatch on the codes-child encoding. The cost is bounded
  but real.

These are the design's accepted costs. None of them are showstoppers; all
are direct consequences of the lossy-extension-type model that the RFC's
two principles (§3) commit to.

## References

_All lemma, theorem, and definition numbers for [1] refer to arXiv:2504.19874v1.
The ICLR 2026 camera-ready proceedings may use different numbering._

[1] Zandieh, A., Daliri, M., Hadian, M. and Mirrokni, V. "TurboQuant: Online
Vector Quantization with Near-optimal Distortion Rate." ICLR 2026.
arXiv:2504.19874, April 2025.

[2] Ailon, N. and Chazelle, B. "The Fast Johnson-Lindenstrauss Transform and
Approximate Nearest Neighbors." SIAM J. Comput. 39(1):302-322, 2009.

[3] Tropp, J.A. "Improved Analysis of the Subsampled Randomized Hadamard
Transform." Adv. Adaptive Data Analysis 3(1-2):115-126, 2011.

[4] Kuffo, L., Krippner, E. and Boncz, P. "PDX: A Data Layout for Vector
Similarity Search." SIGMOD '25. arXiv:2503.04422v1, March 2025.
Open-source implementation: https://github.com/cwida/PDX (MIT). Specific
file references throughout this RFC: `include/pdx/quantizers/scalar.hpp`
(SQ8), `include/pdx/pruners/adsampling.hpp` (ADSampling),
`include/pdx/layout.hpp` (int8 interleaving),
`include/pdx/distance_computers/avx512_computers.hpp` (VPDPBUSD kernels).

[5] Yu, F.X., Suresh, A.T., Choromanski, K., Holtmann-Rice, D. and Kumar, S.
"Orthogonal Random Features." NeurIPS 2016. arXiv:1610.09072.

[6] Yang, S. et al. "Flash-KMeans: Fast and Memory-Efficient Exact K-Means."
arXiv:2603.09229, March 2026.

[7] Pathare, T. et al. "TurboQuant: Implementation Corrections, Production
Hardening, and Deployment Infrastructure." Eviox Tech Report v1.2.0,
March 2026. https://eviox.tech/nexus/eviox_turboquant_corrections_study.pdf
_(Note: this URL may require Eviox account access; not publicly indexed.)_

[8] Community TurboQuant implementation reports (primarily KV-cache attention):

- https://github.com/tonbistudio/turboquant-pytorch — MSE-only (V3) vs MSE+QJL (V2); reports MSE-only wins for attention and generation quality. License: MIT.
- https://github.com/ggml-org/llama.cpp/discussions/20969 — TurboQuant discussion; quantized attention analysis and MSE vs Prod comparison.
- https://github.com/0xSero/turboquant — Triton kernels and vLLM integration; one of multiple groups reporting MSE-only behavior. License: **GPL-3.0** — cited for illustration of community findings only; not implementable under Vortex's MIT/Apache-2.0 license, and not a code dependency.
- https://github.com/scos-lab/turboquant — Reference reproduction; MSE vs Prod/QJL comparison. License: MIT.

Multiple groups report MSE-only beating MSE+QJL for attention metrics at
tested bit widths. ANN ranking conclusions remain preliminary pending
dedicated benchmarks. Note that EDEN [15] dominates TurboQuant_prod
regardless of these findings (per [14]: EDEN's biased mode beats
TurboQuant_mse, and EDEN's unbiased mode beats TurboQuant_prod), so any
unbiased work should adopt EDEN's quantizer rather than QJL.

[9] Jégou, H., Douze, M. and Schmid, C. "Product Quantization for Nearest
Neighbor Search." IEEE Trans. PAMI 33(1):117-128, 2011.

[10] Ge, T., He, K., Ke, Q. and Sun, J. "Optimized Product Quantization."
IEEE Trans. PAMI 36(4):744-755, 2014.

[11] Jääsaari, E., Hyvönen, V., Ceccarello, M., Roos, T. and Aumüller, M.
"VIBE: Vector Index Benchmark for Embeddings." arXiv:2505.17810, May 2025.

[12] Malinovskii, V., Panferov, A., Ilin, I., Guo, H., Richtárik, P. and
Alistarh, D. "Pushing the Limits of Large Language Model Quantization via
the Linearity Theorem." arXiv:2411.17525v1, November 2024.

[13] johndpope et al. "RotorQuant: Clifford algebra vector quantization."
PR #34, TheTom/turboquant_plus, March-April 2026.
https://github.com/TheTom/turboquant_plus/pull/34
Explores SO(2)/SO(3)/SO(4) block-diagonal rotations as alternatives to
full-dimension SORF. Rejected due to 10×+ MSE regressions on real KV-cache
tensors, attributed to insufficient cross-group decorrelation.

[14] Ben-Basat, R., Ben-Itzhak, Y., Mendelson, G., Mitzenmacher, M.,
Portnoy, A. and Vargaftik, S. "A Note on TurboQuant and the Earlier
DRIVE/EDEN Line of Work." arXiv:2604.18555, April 2026. Demonstrates that
TurboQuant_mse is a special case of EDEN [15] with fixed `S = 1`; reports
experiments where biased EDEN beats TurboQuant_mse and native b-bit
unbiased EDEN beats TurboQuant_prod, "often by more than a bit (e.g.,
2-bit EDEN beats 3-bit TurboQuant_prod)."

[15] Vargaftik, S., Ben-Basat, R., Portnoy, A., Mendelson, G., Ben-Itzhak,
Y. and Mitzenmacher, M. "EDEN: Communication-Efficient and Robust
Distributed Mean Estimation for Federated Learning." ICML 2022.
arXiv:2108.08842, August 2021 (v3, June 2022). Foundational paper for the
RHT + Lloyd-Max scalar quantization family TurboQuant belongs to;
introduces the optimal scalar scale `S(d, b)` that this RFC adopts in §6.

[16] Vargaftik, S., Ben-Basat, R., Portnoy, A., Mendelson, G., Ben-Itzhak,
Y. and Mitzenmacher, M. "DRIVE: One-bit Distributed Mean Estimation."
NeurIPS 2021. arXiv:2105.08339, May 2021.
https://proceedings.neurips.cc/paper/2021/hash/0397758f8990c1b41b81b43ac389e143-Abstract.html
Foundational 1-bit unbiased quantizer; EDEN [15] extends to any b > 0.

## Appendix A: Reference implementation bugs and Theorem 1 constant

### Reference implementation bugs

The Eviox corrections study [7] identified six material bugs in the paper's
reference Python implementation. The most critical is a mathematical error
in the QJL scale factor: the reference code used `√(π/(2d))` instead of
`√(π/2)/d` (Definition 1 in [1]), differing by a factor of √d (≈11× at
d=128). Our implementation uses the correct formula (`sqrt(FRAC_PI_2) /
padded_dim` in Rust), so this bug does **not** affect us.

Other notable Eviox findings: (a) the reference code recomputes codebooks at
every instantiation (we cache in a `DashMap`); (b) the reference uses
float16 for codebook distance computation, causing misassignment at small
centroid spacings (we cast to f32 before quantization). See [7] for the
full list.

### Theorem 1 constant

There is an ambiguity in the paper's notation for the MSE bound constant.
The formal proof gives `(√3 · π / 2) · 4^{-b}` where the constant √3·π/2 ≈
2.72. The Eviox report [7] (Item 7) deliberately adopts the alternative
parsing `√(3π)/2 ≈ 1.535`, claiming it is "consistent with the formal
proof." We treat `√3·π/2 ≈ 2.72` as the theorem constant because: (a) the
paper's prose describes the constant as "≈ 2.7," which matches 2.72 not
1.535; and (b) the paper's reported distortion values (b=2: 0.117, b=3:
0.03) exceed the 1.535-based bound (b=2: 0.096, b=3: 0.024), ruling out
`√(3π)/2` as a valid **upper** bound on the measured quantity. The
definitive resolution requires checking the exact LaTeX grouping in the
ICLR 2026 camera-ready proof. The paper's "explicit values" (0.36, 0.117,
0.03, 0.009) are the actual computed distortion of the optimal quantizer,
not the bound itself — they are well below the 2.72/4^b bound.

EDEN [15] provides tighter and more rigorous bounds via its optimized
`S`; once EDEN-`S` is adopted (§6 Stage 1 refinement), we should cite
EDEN's bound preferentially.

## Appendix B: Community findings on QJL

Multiple independent TurboQuant implementations have repeatedly reported a
practical finding for **KV-cache attention**: MSE-only often outperforms
MSE+QJL at the same bit budget. The likely mechanism is a variance-bias
tradeoff: QJL removes bias in raw inner-product estimation but adds
variance, and the softmax nonlinearity amplifies variance more than it
penalizes bias. In that setting, allocating all bits to MSE (more
centroids, lower quantization variance) can beat splitting the budget
between MSE + QJL. This behavior has been reported by multiple groups
across Python, C, and Rust implementations [8].

For ANN search, cosine ranking, and other non-softmax vector-search
workloads, the evidence is currently less settled. MSE-only is a
reasonable default, but the ANN question should be treated as empirical
until evaluated on ANN datasets with recall@k and ranking metrics (see
Experimental plan).

**EDEN's framing supersedes this discussion for unbiased estimation.**
The note [14] reports that direct b-bit unbiased EDEN [15] dominates
`(b-1)`-bit MSE + 1-bit QJL stacking, often by more than a bit (2-bit
EDEN beats 3-bit TurboQuant_prod). If a future workload needs unbiased
estimation, adopt EDEN's unbiased mode rather than QJL.

## Appendix C: Alternative rotation strategies

### Why not DCT?

DCT is O(B log B) and invertible, but it is a **fixed structured
transform**, not a random rotation — it does not produce the Beta marginal
distribution `(1-x²)^((B-3)/2)` (in block dimension B) that TurboQuant's
Max-Lloyd centroids are optimized for. ADSampling only needs approximate
coordinate independence (for hypothesis-testing pruning), so a fixed
orthogonal transform like DCT suffices there. TurboQuant needs a specific
known marginal distribution, so only random orthogonal rotations (QR or
SORF) are suitable.

### Shared rotation with ADSampling (speculative)

Both TurboQuant and ADSampling apply a random orthogonal rotation to make
coordinates independent. If we integrate ADSampling-style dimension pruning
(see Stage 3), the same rotation could in principle serve both purposes.
However, this is not automatic under the Stage 2 block-decomposed design:
ADSampling is formulated around a single full-dimensional random
projection whose coordinates can be sequentially sampled, whereas Stage 2
introduces per-block rotations and per-block norm weighting. Reusing one
rotation across both systems should be treated as a **future research
direction** that requires new analysis or direct empirical validation. If
it proves viable, it would avoid rotating the data twice.

### Fallback: dense rotation

If SORF proves insufficient at the chosen B, use a B × B random orthogonal
matrix (QR of Gaussian). Storage at B=256: 256 KB per block. For d=768
with k=3: 768 KB total. Amortizes for large columns (100K+ vectors). Each
block must have an **independent** rotation matrix. Not the default; only
worth pursuing if Stage 2 SORF quality benchmarks show a meaningful gap.

## Appendix D: Implementation Specification

This appendix is the operational reference for implementers. The main RFC
body is calibrated for an expert reviewer to read in 30–60 minutes; this
appendix is for the implementer who actually writes the Rust. Items here
either consolidate detail from the main body or fill gaps the main body
intentionally leaves open.

### D.1 Wire-format invariants

- **Endianness**: codes are `u8` (byte-addressable; endianness moot). The
  prost metadata uses prost's wire format which is little-endian for
  fixed-width fields; this is part of Vortex's general dtype-metadata
  contract, not specific to TurboQuant.
- **Alignment**: codes are stored as `FixedSizeList<u8, padded_dim>` (or
  `k × block_size`); no special alignment requirement above the FSL
  storage's own. PDX layout (§8) adds 64-row chunk-aligned access for
  fast paths; non-aligned slice/take falls back to row-major.
- **Validity**: row-aligned across both `norms` and `codes` fields. Invalid
  rows store `norm = 0` and placeholder zero codes (which are valid byte
  values referring to centroid 0, not "zero coordinate" — the validity
  bit is authoritative for whether the row is meaningful).

### D.2 Extension type metadata (prost)

The on-disk metadata is the prost-encoded form of:

```rust
// Reproduced from vortex-turboquant/src/vtable.rs at ff120401.
struct TurboQuantMetadataProto {
    element_ptype: PType,    // tag 1, enum
    dimensions:    u32,      // tag 2
    bit_width:     u32,      // tag 3 (fits in u8 at the type level)
    seed:          u64,      // tag 4
    num_rounds:    u32,      // tag 5 (fits in u8 at the type level)
    // Stage 2 addition (not in current source):
    block_size:    optional u32,  // tag 6
    // Future:
    unbiased:      optional bool, // tag 7 — reserve for EDEN unbiased mode
}
```

The current `TurboQuantMetadataProto` in `vortex-turboquant/src/vtable.rs`
defines tags 1–5. Stage 2 adds `block_size` at tag 6; future stages
add additional optional tags as needed. Prost optional-field semantics
mean older readers ignore unknown tags, so the wire format is forward-
compatible by construction.

**Constraint:** tags 1–5 are part of Vortex's stable contract once Stage 1
ships in a release. Renumbering or repurposing them would be a wire-format
break.

### D.3 Validation

The `ExtVTable::validate_dtype` implementation enforces:

- `dimensions >= MIN_DIMENSION` (`= 128`). On violation: error
  `"TurboQuant dimensions must be >= 128, got {N}"`.
- `1 ≤ bit_width ≤ MAX_BIT_WIDTH` (`= 8`). On violation: error
  `"TurboQuant bit_width must be 1-8, got {N}"`.
- `num_rounds > 0`. On violation: error
  `"TurboQuant num_rounds must be > 0, got 0"`.
- `element_ptype.is_float()` (one of F16, F32, F64). On violation: error
  `"TurboQuant element_ptype must be a float, got {ptype}"`.
- Storage dtype is `Struct { norms: Primitive<element_ptype>, codes:
FixedSizeList<u8, padded_dim> }` with matching row-validity propagation.

**Gap (PR #7829 P2 finding):** `tq_padded_dim()` currently uses unchecked
`next_power_of_two()` and panics on oversized dimensions. The Stage 1
stabilization task switches to the checked version and returns
`TurboQuant padded dimension overflow for {dimensions}` on the boundary.
Define `MAX_DIMENSION` as the largest `u32` whose `next_power_of_two()`
fits in `u32` (i.e., `2^31`).

### D.4 Encode (Stage 1) pseudocode

```text
fn tq_encode(v: Vector<F, d>, cfg: &TurboQuantConfig) -> TurboQuantArray:
    padded_dim = next_power_of_two(d)
    if padded_dim does not fit u32:
        return Err(OverflowError)

    n = ‖v‖₂   # in input dtype F (f16/f32/f64)
    if n > 0:
        u_padded[0..d]    = v / n
        u_padded[d..padded_dim] = 0.0
        r = SORF(u_padded, cfg.seed, cfg.num_rounds)   # in f32
        for j in 0..padded_dim:
            codes[j] = nearest_centroid(r[j] * S, centroids)
    else:
        codes[0..padded_dim] = 0   # placeholder; validity marks the row invalid

    return TurboQuantArray {
        metadata: TurboQuantMetadata {
            element_ptype = F,
            dimensions = d,
            bit_width = cfg.bit_width,
            seed = cfg.seed,
            num_rounds = cfg.num_rounds,
            block_size = None,   # Stage 1 implicit
        },
        norms: n,
        codes: codes,
    }
```

`centroids = get_centroids(padded_dim, bit_width)` and `S = get_eden_scale(padded_dim, bit_width)`
both come from the process-local cache keyed on `(padded_dim, bit_width)`.

### D.5 Decode (Stage 1) pseudocode

```text
fn tq_decode(tq: TurboQuantArray) -> Vector<F, d>:
    d           = tq.metadata.dimensions
    padded_dim  = next_power_of_two(d)
    centroids   = get_centroids(padded_dim, tq.metadata.bit_width)
    S           = get_eden_scale(padded_dim, tq.metadata.bit_width)

    if validity(row) == false:
        return null

    for j in 0..padded_dim:
        r_hat[j] = centroids[tq.codes[j]] / S

    u_hat_padded = SORF_inverse(r_hat, tq.metadata.seed, tq.metadata.num_rounds)
    u_hat        = u_hat_padded[0..d]   # truncate the zero-padding

    v_hat        = tq.norms * u_hat     # in dtype F
    return v_hat
```

### D.6 Encode (Stage 2) pseudocode

```text
fn tq_encode_stage2(v: Vector<F, d>, cfg: &TurboQuantConfig, B: u32) -> TurboQuantArray:
    k = d / B   # exact division required; if not, fall back to Stage 1 padded

    for i in 0..k:
        v_i  = v[i*B .. (i+1)*B]
        n_i  = ‖v_i‖₂
        if n_i > 0:
            u_i = v_i / n_i
            r_i = SORF(u_i, block_seed(seed, i), cfg.num_rounds)
            for j in 0..B:
                codes[i*B + j] = nearest_centroid(r_i[j] * S, centroids)
        else:
            codes[i*B .. (i+1)*B] = 0

    return TurboQuantArray {
        metadata: TurboQuantMetadata {
            element_ptype = F,
            dimensions = d,
            bit_width = cfg.bit_width,
            seed = cfg.seed,
            num_rounds = cfg.num_rounds,
            block_size = Some(B),
        },
        norms: [n_0, n_1, ..., n_{k-1}],
        codes: codes,
    }
```

`centroids = get_centroids(B, bit_width)` is keyed on the block dimension,
not the original dimension. `block_seed(seed, i)` is the per-block rotation
seed derivation, which the implementer must commit to before any Stage 2
file is shipped (see Open Questions §3).

### D.7 PDX distance kernel (dot product, b=4)

Reproduced verbatim from §8 for the implementer's convenience:

```rust
// Precomputed (2^b)² distance table; at b=4 this is 16×16 = 256 floats = 1 KB.
let dist_table = precompute_product_table(&centroids);

let mut distances  = [0.0f32; 64];
let mut unit_dots  = [0.0f32; 64];
let mut offset     = 0;

for tq_block in 0..k {
    for dim in 0..block_size {
        let qd  = query_codes[tq_block * block_size + dim];
        let row = &dist_table[qd as usize];
        for v in 0..64 {  // SIMD-friendly: no inter-vector deps
            unit_dots[v] += row[codes_pdx[offset] as usize];
            offset += 1;
        }
    }
    // Weight per-block unit-norm dot product by both vectors' block norms.
    for v in 0..64 {
        distances[v] += query_norms[tq_block]
                      * data_norms[v][tq_block]
                      * unit_dots[v];
        unit_dots[v] = 0.0;
    }
}
```

PDX chunk-tail handling at row counts not divisible by 64: the last chunk
covers the remaining rows row-major (i.e., the kernel above runs over the
non-tail chunks; the tail uses the FSL row-major fallback). 64-row-aligned
fast paths skip the transpose; non-aligned slice/take pays
`O(rows × block_size)` to materialize the FSL form.

### D.8 Error model

| Operation                         | Failure mode                                            | Error                                                                                            |
| --------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `TQEncode::try_new_array`         | child is not a `Vector` extension                       | `"expected a Vector extension array, got {dtype}"`                                               |
| `TurboQuantConfig::try_new`       | `bit_width` out of `[1, 8]`                             | `"TurboQuant bit_width must be 1-8, got {N}"`                                                    |
| `TurboQuantConfig::try_new`       | `num_rounds == 0`                                       | `"TurboQuant num_rounds must be > 0, got {N}"`                                                   |
| `validate_tq_metadata`            | `dimensions < 128`                                      | `"TurboQuant dimensions must be >= 128, got {N}"`                                                |
| `validate_tq_metadata`            | `element_ptype` not a float                             | `"TurboQuant element_ptype must be a float, got {ptype}"`                                        |
| `tq_padded_dim` (after fix)       | `next_power_of_two(dimensions)` overflows `u32`         | `"TurboQuant padded dimension overflow for {dimensions}"`                                        |
| Stage 2 `block_size` validation   | `B` not a power of 2 in `[64, padded_dim]` dividing `d` | `"TurboQuant block_size {B} must be a power of 2 in [64, padded_dim] that divides {dimensions}"` |
| `TQDecode::try_new_array`         | child is not a TurboQuant extension                     | `"expected a TurboQuant extension array, got {dtype}"`                                           |
| Quantized-domain ops, op operands | mismatched `(bit_width, block_size, seed, num_rounds)`  | (fall back to decode-then-compute; no error)                                                     |

Empty inputs (zero rows) return an empty TurboQuant array with `len() == 0`
and no codes/norms storage — no error.

### D.9 Worked example

Stage 1 round trip for a single vector at `d = 128, bit_width = 8`:

```text
Input:
  v = [v_0, v_1, ..., v_127]    (f32, ‖v‖ > 0)

Step 1 — store norm:
  n = ‖v‖₂                       (f32 scalar)

Step 2 — normalize:
  u = v / n                      (f32, ‖u‖ = 1)

Step 3 — SORF (3 rounds, seed = 42):
  r = SORF_3(u, seed=42)         (f32, padded_dim = 128, ‖r‖ = 1, marginally Beta-distributed)

Step 4 — scalar quantize with EDEN's S:
  for j in 0..128:
      codes[j] = nearest_centroid(r[j] * S, centroids_128_8)   (u8)

Storage shape (Extension<vortex.turboquant>):
  Struct {
    norms: Primitive<f32> = [n]
    codes: FixedSizeList<u8, 128> = [codes[0], codes[1], ..., codes[127]]
  }

Decode:
  for j in 0..128:
      r_hat[j] = centroids_128_8[codes[j]] / S
  u_hat = SORF_3_inverse(r_hat, seed=42)
  v_hat = n * u_hat              (f32, ‖v_hat‖ ≠ ‖v‖ in general; ‖v_hat - v‖² ≤ MSE_bound × ‖v‖²)
```

At `bit_width = 8`, MSE_bound = 2.72 / 4^8 ≈ 4.15e-5 (relative). On
random Gaussian inputs, observed normalized MSE ≈ 4e-5.

### D.10 Test plan

The current `vortex-turboquant` test suite (911 lines as of PR #7829)
covers:

- Roundtrip on random inputs at every supported bit width
- MSE bound assertions: `normalized_mse < 2.72 / 4^b` per vector
- Edge cases: empty arrays, single-row, all-zero, nullable vectors
- Serde roundtrip
- Compute pushdowns: cosine_similarity, dot product, L2 norm
- Centroid correctness against numerical integration
- SORF determinism (same seed → same rotation)

Stages 2 and 3 add:

- **Stage 2**: block-decomposition roundtrip at d ∈ {768, 1536, 3072};
  per-block norm storage shape assertion; centroid cache key change
  (`(block_size, bit_width)`); per-block weighted-sum kernel parity vs.
  decode-then-compute fallback.
- **Stage 3**: PDXArray transpose / un-transpose roundtrip; PDX kernel
  parity vs. row-major kernel; 64-aligned vs. non-aligned slice/take
  perf assertion.
- **Cross-cutting**: regression tests for each PR #7829 review finding —
  oversized-dim panic → error, `initialize` ordering documentation, lazy
  `TQEncode` write path.

### D.11 Performance budgets

Goals (from §11) become budgets here, with verification commands:

- **Encode throughput, Stage 1, AVX-512, d = 768, b = 8**: ≥ 1 M vectors/sec.
  Verify with `cargo run -p vortex-turboquant --release --bench encode_decode`.
- **Decode throughput, Stage 1, AVX-512, d = 768, b = 8**: ≥ 1 M vectors/sec.
- **Encode throughput, Stage 2, AVX-512, d = 768, k = 3, B = 256, b = 8**:
  ≥ 1.3 M vectors/sec (≥ 30% faster than Stage 1 padded, matching the FLOP
  ratio in §11).
- **Compression ratio, b = 8**: 3.0× (Stage 1 padded at d = 768), 3.9×
  (Stage 2 k = 3 at d = 768), 4.0× (any stage at d = 1024).
- **Normalized MSE, b = 8, d ≥ 128**: < 5e-5 on Gaussian inputs.
- **Stage 3 PDX scan throughput, AVX-512, b = 4, d = 768**: ≥ 1.5×
  Stage 2's row-major kernel throughput on 1 M-row scan.

These are starting budgets; the Experimental plan (§12) refines them with
real workloads.

### D.12 Registry / dispatch wiring

- **Extension ID**: `vortex.turboquant`, registered via
  `vortex_turboquant::initialize(&session)`. Implementation:
  `session.dtypes().register(TurboQuant)`.
- **Scalar function IDs**: `vortex.turboquant.encode` (TQEncode),
  `vortex.turboquant.decode` (TQDecode). Registered via
  `session.scalar_fns().register(...)`.
- **BtrBlocks scheme wiring (target)**: `BtrBlocksCompressorBuilder::with_turboquant()`
  installs a `TurboQuantScheme` that matches Vector columns with
  `dimensions ≥ MIN_DIMENSION` and non-nullable float elements. Not yet
  present in the new `vortex-turboquant` crate; Stage 1 stabilization
  adds it.
- **PDXArray encoding (Stage 3)**: register in `vortex-array` (recommended)
  or a sibling crate. Encoding ID TBD; suggested `vortex.pdx`.

### D.13 Crate boundaries and dependencies

- `vortex-turboquant` depends on `vortex-array`, `vortex-buffer`,
  `vortex-error`, `vortex-mask`, `vortex-session`, `vortex-tensor`,
  `vortex-utils` (with `dashmap` feature). No new external dependencies.
- Nothing in the main `vortex` crate depends on `vortex-turboquant`. The
  crate registers itself opt-in.
- Stage 2: no new crate dependencies.
- Stage 3: adds a dependency on whatever crate hosts `PDXArray`
  (recommend `vortex-array`).

### D.14 Migration sequence: removing the predecessor implementations

After `vortex-turboquant` ships:

1. (Now) Both `vortex-tensor/src/encodings/turboquant/` (monolithic) and
   `vortex-tensor/src/scalar_fns/{l2_denorm,sorf_transform}` (decomposed)
   exist alongside `vortex-turboquant`.
2. Migrate `vortex/examples/turboquant_vector_search.rs` to the new
   `TQEncode` / `TQDecode` path (Stage 1 stabilization task).
3. Migrate any downstream consumers of the legacy paths (Vortex test
   suite, benchmarks, duckdb-vortex if applicable).
4. Deprecate the legacy paths with `#[deprecated]` and a removal target
   version.
5. Remove the legacy paths in a follow-up PR.

Each step is independently mergeable.

## Open questions

These are recorded for resolution during Stage 1 / Stage 2 implementation
work, not blockers on the design:

1. **Where `PDXArray` lives.** `vortex-array` (recommended; general-purpose,
   reusable for other encodings) vs. `vortex-turboquant` (TQ-specific,
   generalize later). The deciding factor is whether `PDXArray` will have
   non-TurboQuant consumers; the first such consumer settles the question.
2. **EDEN-`S` table pinning.** Pin EDEN's `S` table as a Vortex constant
   (alongside the SplitMix64 stream), or version the centroid/scale
   algorithm in metadata? Recommend pinning for simplicity, but verify
   first that EDEN's `S` is fully deterministic from `(d, b)` alone — i.e.,
   no free parameters in EDEN's optimization criterion that would force
   versioning. Settle by reading EDEN [15] §X (specific section TBD once
   the implementer dives in).
3. **Per-block rotation derivation (Stage 2).** A single stored seed plus
   block-index mixing keeps metadata small while preserving determinism.
   One candidate: `block_seed(b) = SplitMix64(seed ^ (b as u64))`. The
   exact mixing function is a Stage 2 implementation detail that should be
   pinned before the first Stage 2 file is written; the wire format
   becomes load-bearing once any file is shipped.
4. **Wire-format identity at d=1024 across Stage 1 → Stage 2 writers.** When
   a Stage 2 writer emits a power-of-2-dimension TurboQuant array (so
   k = 1), should it write `block_size = None` (matching Stage 1 readers'
   default exactly) or `block_size = Some(padded_dim)`? Stage 2 readers
   accept both; Stage 1 readers only accept `None`. Writers must converge
   to one or the other to preserve cross-version write/read interop.
5. **Whether to lower `MIN_DIMENSION` after Stage 1 experimental
   validation.** If the experimental plan supports lowering to 64–96, the
   change is a wire-format break in the sense that files written at
   d < 128 by a new writer would be rejected by an old reader. Surface
   this in the migration plan when the experiment lands.

### Resolved during initial drafting

The following questions were considered during the design and have been
resolved here. Recording the resolutions so future readers don't
re-litigate.

- **Norm storage shape uniformity (Stage 2)** — resolved in favor of
  `Primitive<element_ptype>` when `num_blocks == 1` (matches Stage 1
  exactly, preserves bit-identical wire format at d = 1024) and
  `FixedSizeList<element_ptype, num_blocks>` when `num_blocks > 1`. The
  simplicity-of-uniform-FSL argument is real but breaks wire-format
  identity. See §7.
- **EDEN-`S` adoption timing** — resolved: adopt as part of Stage 1
  production stabilization. The change is strictly additive (no storage
  or metadata shape change) and a strict MSE win at fixed bit budget;
  there is no benchmarking reason to wait. See §6 "Stage 1 refinement."
- **Unbiased path: EDEN vs. QJL** — resolved: when an unbiased mode is
  eventually added, use EDEN's native b-bit unbiased quantizer [15] in
  place of TurboQuant's MSE+QJL "Prod" stacking. The note [14] reports
  EDEN dominates at every bit width they tested. See §15 and Appendix B.
- **`vortex-tensor/src/encodings/turboquant/` removal** — tracked as a
  Stage 1 cleanup task in §14 "Current state and known gaps." Removal
  follows once `vortex/examples/turboquant_vector_search.rs` migrates to
  the new `TQEncode` / `TQDecode` path.
