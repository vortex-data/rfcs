# Block-Decomposed TurboQuant with PDX Layout

**Authors:** Will Manning
**Status:** Proposal
**Date:** 2026-04-02

## Summary

We propose evolving the [TurboQuant vector quantization encoding][current-impl]
in three stages:

1. **MSE-only TurboQuant** (immediate): merge the current PR as an MSE-only
   encoding. This is a complete, self-contained building block.
2. **Block decomposition** (next): for dimensions where a valid B exists
   (greatest power-of-2 ≥ 64 dividing d), split into blocks of size B. For
   power-of-2 dimensions, B = d (single block). Dimensions with no qualifying
   B fall back to padded single-block. Per-block norms stored as internal
   children.
3. **PDX layout** (later): transpose codes into dimension-major order within
   groups of 64 vectors for SIMD scan performance.

QJL correction is deferred to a later stage and may ultimately be dropped.
Community findings from multiple independent TurboQuant implementations
often show that MSE-only outperforms MSE+QJL for KV-cache attention [8].
For ANN ranking and vector-search workloads, the evidence is currently less
complete, so QJL should remain an empirical question rather than a settled
conclusion.

[current-impl]: https://github.com/vortex-data/vortex/pull/7167

## Background

### TurboQuant

TurboQuant [1] is a lossy vector quantization algorithm for high-dimensional
embeddings. It works by:

1. Randomly rotating a unit-norm vector so that each coordinate follows a known
   marginal distribution — specifically `(1 - x²)^((d-3)/2)` on [-1, 1], a
   concentrated Beta distribution (Lemma 1 in [1]; verify numbering against the
   ICLR 2026 camera-ready if it differs from the arXiv version).
2. Applying an MSE-optimal scalar quantizer (Max-Lloyd centroids) independently
   to each coordinate.
3. Optionally adding a 1-bit QJL (Quantized Johnson-Lindenstrauss) correction
   on the residual for unbiased inner product estimation (Theorem 2 in [1];
   same camera-ready caveat).

The paper prescribes a full random orthogonal rotation (QR decomposition of a
matrix with i.i.d. N(0,1) entries, yielding a Haar-uniform orthogonal matrix)
for the MSE stage — O(d²) storage and O(d²) per-vector. For the QJL stage, the
paper uses a random Gaussian projection matrix S with i.i.d. N(0,1) entries (not
an orthogonal rotation); this distinction matters for the unbiasedness proof.

**Comparison to Product Quantization.** TurboQuant's block decomposition (Stage
2 of this RFC) is structurally similar to Product Quantization (PQ) [9]: both
partition a vector into sub-vectors and quantize each independently. The key
differences are:

|                        | TurboQuant                                                      | PQ                                                       |
| ---------------------- | --------------------------------------------------------------- | -------------------------------------------------------- |
| Quantization type      | Scalar (per-coordinate, after rotation)                         | Vector (per-sub-vector, learned codebook)                |
| Codebook               | Analytically derived from Beta distribution; **data-oblivious** | Learned via k-means on training data; **data-dependent** |
| Rotation               | Random orthogonal within each sub-vector                        | Typically none (OPQ [10] adds a learned rotation)        |
| Theoretical guarantees | Provable MSE bound (Theorem 1 [1])                              | Empirical quality only                                   |
| Codebook training      | None (centroids derived from theory)                            | Requires training pass over data                         |
| Bits per sub-vector    | Scalar: b bits per coordinate                                   | Vector: typically 8 bits per sub-vector (256 codewords)  |

TurboQuant trades PQ's flexibility (data-dependent codebooks can exploit
structure) for data-obliviousness (no training, provable bounds, no offline
index-training phase). Encode-time work (rotation + quantization) still applies.
In return, PQ and OPQ retain a major advantage in expressivity: they learn
sub-vector codebooks from data rather than applying an analytic scalar quantizer.
In practice this means TurboQuant is attractive when training-free operation,
simple deployment, and theoretical guarantees matter most, while PQ or OPQ may
still win empirically when a learned vector codebook can exploit dataset-specific
structure.

### Current Vortex implementation

Our [current implementation][current-impl] (Rust, in the `vortex-tensor` crate)
implements TurboQuant as a Vortex array encoding that compresses
`FixedSizeList<float>` arrays — the storage format of `Vector` and
`FixedShapeTensor` extension types. Key design choices and characteristics:

**Rotation.** Instead of the paper's O(d²) QR rotation, we use a 3-round
Structured Orthogonal Random Features (SORF) transform `HD₃·HD₂·HD₁` [5] for
both the MSE rotation and the QJL projection, giving O(d) storage (3d sign bits,
bitpacked) and O(d log d) per-vector. The rotation signs are stored as a
bitpacked child array rather than recomputed from a seed at decode time. The
3-round SORF was introduced for kernel approximation [5] and approximates a
random orthogonal matrix. It is distinct from the single-round SRHT (`R·H·D`)
analyzed by Tropp [3] and the FJLT (`P·H·D`) of Ailon-Chazelle [2], both of
which are dimensionality-reducing projections rather than rotation
approximations.

**Centroids.** Max-Lloyd centroids are computed via numerical integration
(trapezoid rule, 1000 points per interval) of the marginal Beta distribution at
the padded dimension, using the `HalfIntExponent` type for exact integer/half-
integer exponent arithmetic. Centroids are cached in a global `DashMap` keyed by
`(dimension, bit_width)` and stored as a shared `PrimitiveArray<f32>` child.

**Array structure.** The `TurboQuantArray` stores up to 7 child slots: codes
(`FixedSizeListArray<u8>`, one per vector, list_size = padded_dim), norms
(`PrimitiveArray<f32>`), centroids (shared), MSE rotation signs (shared,
bitpacked), and optionally 3 QJL children (signs, residual norms, QJL rotation
signs). Codes are stored as u8 centroid indices; the cascade compressor
(BitPacked encoding) handles packing to the actual bit width on disk.

**Compute pushdowns.** Slice and take propagate to per-row children (codes,
norms) while sharing rotation signs and centroids. Quantized cosine similarity
and dot product operate directly on codes and centroids without decompression.
L2 norm returns the stored norm directly (O(1) readthrough).

**Compression scheme.** `TurboQuantScheme` implements the `Scheme` trait for the
BtrBlocks cascading compressor. It matches `Vector` and `FixedShapeTensor`
extension arrays with non-nullable float elements and dimension ≥ 3 (to be
raised to ≥ 128 in Stage 1; see Minimum dimension below), using the
default config (5-bit QJL = 4-bit MSE + 1-bit QJL, seed 42).

**Input handling.** All float types (f16, f32, f64) are converted to f32 before
quantization. Per-vector L2 norms are computed and stored as f32. Non-power-of-2
dimensions are zero-padded to the next power of 2 for SORF compatibility. The
minimum dimension is 3 (d=2 causes a singularity in the Beta distribution
exponent).

### Reference implementation bugs

The Eviox corrections study [7] identified six material bugs in the paper's
reference Python implementation. The most critical is a mathematical error in
the QJL scale factor: the reference code used `√(π/(2d))` instead of
`√(π/2)/d` (Definition 1 in [1]), differing by a factor of √d (≈11× at d=128).
Our [current implementation][current-impl] uses the correct formula
(`sqrt(FRAC_PI_2) / padded_dim` in Rust), so this bug does **not** affect us.

Other notable Eviox findings: (a) the reference code recomputes codebooks at
every instantiation (we cache in a `DashMap`); (b) the reference uses float16
for codebook distance computation, causing misassignment at small centroid
spacings (we cast to f32 before quantization). See [7] for the full list.

### Theorem 1 constant

There is an ambiguity in the paper's notation for the MSE bound constant. The
formal proof gives `(√3 · π / 2) · 4^{-b}` where the constant √3·π/2 ≈ 2.72.
The Eviox report [7] interprets the notation as `√(3π)/2 ≈ 1.535`, but this is
incorrect: the measured distortion values from the paper (b=2: 0.117, b=3: 0.03)
exceed the putative `√(3π)/2` bound (b=2: 0.096, b=3: 0.024), confirming that
2.72 is the correct constant. The paper's "explicit values" (0.36, 0.117, 0.03,
0.009) are the actual computed distortion of the optimal quantizer, not the
bound itself — they are well below the 2.72/4^b bound.

### Community findings on QJL

Multiple independent TurboQuant implementations have repeatedly reported a
practical finding for **KV-cache attention**: MSE-only often outperforms MSE+QJL
at the same bit budget. The likely mechanism is a variance-bias tradeoff: QJL
removes bias in raw inner-product estimation but adds variance, and the softmax
nonlinearity amplifies variance more than it penalizes bias. In that setting,
allocating all bits to MSE (more centroids, lower quantization variance) can beat
splitting the budget between MSE + QJL. This behavior has been reported by
multiple groups across Python, C, and Rust implementations [8].

For ANN search, cosine ranking, and other non-softmax vector-search workloads,
the evidence is currently less settled. MSE-only is still a reasonable default
because it is simpler and better supported by the current implementation work,
but the ANN question should be treated as empirical until evaluated on ANN
datasets with recall@k and ranking metrics (see Experimental plan).

### Current limitations

The SORF requires power-of-2 input dimension. For non-power-of-2 dimensions
(e.g., 768-d embeddings), the input is zero-padded to the next power of 2
(1024). This causes:

- **33% storage overhead** for 768-d vectors: 1024 codes stored vs. 768 useful
  (equivalently, 25% of stored codes are wasted on zero-padded dimensions).
- **No scan-optimized layout**: row-major code storage prevents SIMD-over-vectors
  distance computation.

### PDX

PDX [4] is a data layout for vector similarity search. The paper (SIGMOD '25)
describes a dimension-major layout within fixed-size blocks of 64 vectors,
enabling the compiler to auto-vectorize the inner distance loop over vectors
rather than dimensions. In the paper, this yields average speedups of about 40%
over SIMD-optimized row-major kernels for the direct kernel comparison, while
dimension-pruning methods (ADSampling, BSA) recover much larger gains (2-7×)
when paired with the PDX layout [4]. The block size of 64 is empirically optimal
across AVX-512, AVX2, and NEON architectures [4].

**PDX implementation evolution.** The [open-source implementation][pdx-impl]
has evolved beyond the paper in several ways relevant to this RFC:

- **8-bit scalar quantization** (`IndexPDXIVFTreeSQ8`): Maps floats to 0-255 via
  linear min-max scaling. The int8 layout differs from float32: dimensions are
  packed in groups of 4 ("4 dims × 16 vecs") to leverage hardware dot-product
  instructions (VPDPBUSD on x86, UDOT/SDOT on ARM) that process 4 byte pairs
  per operation. This is a different tiling than the paper's "1 dim × 64 vecs."
- **ADSampling with random rotation**: The pruner applies a random orthogonal
  rotation (QR of Gaussian, or DCT when FFTW is available) to the entire
  collection as a preprocessing step. This makes coordinates approximately
  independent, enabling dimension-by-dimension hypothesis testing for early
  pruning. The rotation serves a similar purpose to TurboQuant's rotation —
  making the coordinate distribution known — but for pruning rather than
  quantization.
- **Dimension zones**: Consecutive dimensions are grouped into zones; at query
  time, zones are ranked by "distance-to-means" and the most discriminative
  zones are scanned first, enabling faster pruning.
- **Future: 1-bit vectors** are mentioned as planned.

**Implications for our design.** The PDX paper's float32 layout ("1 dim × 64
vecs") maps cleanly to our quantized-code scan kernel, where the inner loop
gathers from a centroid-product distance table over 64 vectors. However, if we
pursue direct int8 arithmetic (b_mse=8 with linear centroids, see GPU section),
the "4 dims × 16 vecs" int8 layout from the PDX implementation may be more
appropriate, as it enables hardware dot-product instructions.

Additionally, ADSampling's dimension-pruning approach is complementary to
TurboQuant's block structure: when scanning with block decomposition, the pruner
could skip entire TQ blocks (B dimensions at a time) if the partial distance
already exceeds the candidate threshold. This combines the storage efficiency of
quantization with the computational savings of early termination.

[pdx-impl]: https://github.com/cwida/PDX

## Proposal

### Block size strategy

For each dimension d, choose B = the greatest power-of-2 ≥ 64 that evenly
divides d. If no such B exists (e.g., d=96), fall back to the padded
single-block path from Stage 1. For common embedding dimensions, this rule
always produces a valid B and eliminates padding entirely:

| Dimension d | Block size B | Blocks k | Notes                        |
| ----------- | ------------ | -------- | ---------------------------- |
| 512         | 512          | 1        | Single block (= current TQ)  |
| 768         | 256          | 3        | Greatest dividing power-of-2 |
| 1024        | 1024         | 1        | Single block                 |
| 1536        | 512          | 3        |                              |
| 2048        | 2048         | 1        | Single block                 |
| 3072        | 1024         | 3        |                              |
| 4096        | 4096         | 1        | Single block                 |

**Key observations:**

- **Power-of-2 dimensions** (512, 1024, 2048, 4096) use B = d — a single block,
  identical to the current implementation except with PDX underneath (Stage 3).
  No block decomposition overhead, no per-block norms. These dimensions are
  already well-served by the current design.
- **Non-power-of-2 dimensions** (768, 1536, 3072) decompose into k=3 blocks at
  B=256 or B=512. No padding waste (vs. 33% for the padded single-block path).
  Each block has its own SORF rotation and shares a single centroid set.
- **No qualifying B is rare** for common embedding dimensions. Dimensions where
  no power-of-2 ≥ 64 divides d (e.g., 96, 100) fall back to Stage 1's padded
  single-block path. These are uncommon in modern model architectures.
- **The SORF approximation at B=256+ is expected to be adequate**: 3 rounds at
  B=256 provides 24 butterfly stages, and at B=512 provides 27 — both comparable
  to the current B=1024 (30 stages). This needs empirical validation; see
  Experimental plan.

### Minimum dimension

The compression scheme should only select TurboQuant for vectors with
dimension ≥ 128. Below this threshold, several factors degrade quality and
efficiency:

- **SORF mixing quality:** 3-round SORF at d=64 provides only 18 butterfly
  stages (vs. 21 at d=128, 30 at d=1024). The coordinate distribution deviates
  more from the analytical Beta, making Max-Lloyd centroids less optimal.
- **Practical MSE:** At smaller d, the Beta marginal is wider (variance ~1/d),
  leading to higher absolute MSE at the same bit width b. The gap between
  practical MSE and the theoretical upper bound is an empirical question at
  each d.
- **Overhead ratio:** Per-vector norm (32 bits) is a larger fraction of the
  compressed representation at small d. At d=32, b=5: norm is 20% of the
  compressed size. At d=768: <1%.
- **Diminishing returns for high bit widths:** With fewer coordinates, the
  fine-grained centroid structure of high-b quantization has less to exploit.

The threshold of 128 is conservative:

- d=128 (SIFT) is the smallest dimension in our recommended benchmark table.
- SORF at d=128 has 21 butterfly stages — tested and adequate in the current
  implementation.
- The block-size rule produces B=128 for d=128 (single block, no decomposition).

The array-level minimum remains d=3 (for the Beta distribution to be
well-defined), so users can still explicitly construct a TurboQuantArray at
smaller dimensions. The scheme minimum (128) controls automatic selection only.

The exact threshold should be validated experimentally — see Experimental plan.

### Stage 1: MSE-only TurboQuant (immediate — split from current PR)

Split the [current PR][current-impl] to extract and merge the MSE-only subset.
The QJL code can be preserved on a separate branch for Phase 4.

**Changes vs. current PR:**

| Aspect         | Current PR                                  | Stage 1                                               |
| -------------- | ------------------------------------------- | ----------------------------------------------------- |
| QJL support    | Full (encode, decode, QJL slots, QJL tests) | **Removed**                                           |
| Array slots    | 7 (4 MSE + 3 QJL)                           | **4** (codes, norms, centroids, rotation_signs)       |
| Scheme default | 5-bit QJL (4-bit MSE + 1-bit QJL)           | **5-bit MSE-only** (32 centroids)                     |
| Norms dtype    | Always f32                                  | **Same-or-wider**: f64 for f64 input, f32 for f32/f16 |
| Metadata       | `has_qjl: bool`                             | **Removed** (always MSE-only)                         |
| Scheme minimum | dimension ≥ 3                               | **dimension ≥ 128** (see Minimum dimension below)     |

**Unchanged from current PR:** SORF rotation, Max-Lloyd centroids,
zero-padding for non-power-of-2, slice/take/scalar_at pushdowns, quantized
cosine similarity and dot product, compression scheme integration.

**Added to metadata (for forward compat):** `block_size: u32` (always =
padded_dim), `num_blocks: u32` (always = 1). These fields are inert in Stage 1
but enable Stage 2 decoders to read Stage 1 files. (PDX is handled via the
codes child type, not a metadata flag — see Stage 3.)

This is a complete, useful encoding for all dimensions ≥ 3 (automatic scheme
selection applies only for d ≥ 128; smaller d remains available via explicit
array construction). Power-of-2 dimensions
have zero padding waste; non-power-of-2 dimensions have the padding overhead
described above.

### Stage 2: Block decomposition

For dimensions where the block-size rule produces a valid B (see table above),
split into blocks of size B. Each full block gets an independent B-dim SORF
rotation. Dimensions with no qualifying B (e.g., d=96) remain on the padded
single-block path from Stage 1.

**Changes vs. Stage 1:**

| Aspect                | Stage 1                              | Stage 2                                                                      |
| --------------------- | ------------------------------------ | ---------------------------------------------------------------------------- |
| Block count           | k = 1 (single block at padded_dim)   | **k = d/B** (multiple blocks, no padding)                                    |
| SORF dimension        | padded_dim (e.g., 1024 for d=768)    | **B** (e.g., 256 for d=768)                                                  |
| Rotation signs        | Single set, len = 3 × padded_dim     | **k sets**, len = k × 3 × B                                                  |
| Centroids             | Computed for padded_dim distribution | **Computed for B-dim distribution** (different codebook!)                    |
| Norms child           | `PrimitiveArray<F>`, 1 per vector    | **`PrimitiveArray<F>` (k=1) or `FixedSizeListArray<F>` (k>1)**, same dtype F |
| Codes list_size       | padded_dim                           | **k × B** (= d for no-straggler dims)                                        |
| Scheme compress()     | Pad → single SORF → quantize         | **Choose B → split → per-block normalize/rotate/quantize**                   |
| Quantized dot product | Single sum over padded_dim centroids | **Per-block weighted sum** (Σ_k norm_a_k · norm_b_k · unit_dot_k)            |
| L2 norm readthrough   | O(1) — return stored norm            | **O(k)** — compute √(Σ_k norm_k²)                                            |
| Zero-padding waste    | Up to 33% (768→1024)                 | **Zero** for common dims                                                     |

**Unchanged from Stage 1:** SORF construction (3-round HD), Max-Lloyd algorithm,
f32 internal quantization, slice/take semantics (per-row data sliced, shared
data cloned), bitpacked rotation sign storage, compression scheme trait.

**For power-of-2 dimensions**: B = d, k = 1. The encoding produces an identical
wire format to Stage 1 (single norm, single SORF, single codes block). A Stage 2
encoder writing k=1 data is fully backward-compatible with Stage 1 decoders.

**Key design properties:**

- **Self-contained.** The TurboQuant array handles block splitting, per-block
  normalization, rotation, and quantization internally. No parent cooperation
  is needed.
- **One shared centroid set** for all blocks at the same B-dim distribution.
- **Per-block SORF rotation signs.** Each block's SORF is independent (different
  seed). Signs are 3 × B bits per block.

#### Norm architecture

Per-block norms are stored as an **internal child** of the TurboQuant array:

- For k = 1 (power-of-2 dims): `PrimitiveArray<F>` with len = num_rows
  (identical to Stage 1's single-norm layout).
- For k > 1: `FixedSizeListArray<F>` with list_size = k, len = num_rows.

The norm dtype `F` matches or widens the input element type:

| Input dtype | Norm dtype | Rationale                                      |
| ----------- | ---------- | ---------------------------------------------- |
| f16         | f32        | f16 has insufficient range/precision for norms |
| f32         | f32        | Same type                                      |
| f64         | f64        | Preserve full precision                        |

Norms are stored as plain child arrays; the cascading compressor handles
secondary encoding (ALP, Pco, etc.).

Note: centroids and quantization always operate in f32 internally (the
[current implementation][current-impl] converts all input to f32 before
quantization). For f64 input, decode produces f32 unit-direction reconstructions
scaled by f64 norms — a mixed-precision multiply that preserves norm precision.

#### Zero-norm sub-vectors

When splitting a vector into B-dim blocks, some blocks may have zero norm. The
encoding handles ‖xₖ‖ = 0 explicitly: skip rotation and quantization, store
norm = 0, decode as all zeros.

#### Theoretical MSE bound

The paper's MSE bound (Theorem 1 in [1]; verify theorem numbering against the
ICLR 2026 camera-ready if it differs from the arXiv version) is:

```
E[‖x - x̂‖² / ‖x‖²] ≤ (√3 · π / 2) / 4^b ≈ 2.72 / 4^b
```

**Crucially, Theorem 1 is proved for true random orthogonal matrices (QR of
Gaussian), not SORF.** Our SORF is an approximation. The bound holds exactly
only with a true random orthogonal rotation or with empirical SORF validation
(see Experimental plan).

Assuming the per-block MSE bound holds, for a vector split into blocks the
first line is an **algebraic** identity (exact); the inequality on the second
line applies Theorem 1's **probabilistic** bound to each block and should be
read as holding in **expectation** over independent per-block rotations, not
almost surely:

```
‖x - x̂‖² / ‖x‖² = Σ_k (‖xₖ‖² / ‖x‖²) × (‖xₖ - x̂ₖ‖² / ‖xₖ‖²)      (exact)
    E[...]         ≤ MSE_bound × Σ_k (‖xₖ‖² / ‖x‖²) = MSE_bound          (in expectation)
```

The conclusion: `E[‖x - x̂‖² / ‖x‖²] ≤ MSE_bound` assuming independent
per-block rotations. (Theorem 1 applies because each block is normalized to
unit norm before rotation and quantization; the per-block encoding pipeline is:
split → normalize → rotate → quantize, matching the theorem's unit-sphere
assumption.) Note that TurboQuant's original analysis uses a single
global rotation in high-d where coordinates are nearly independent; with
smaller block dimension B, within-block coordinate dependence after rotation may
be stronger even when marginals are correct — this is an additional motivation
for the experimental plan's comparison of block sizes.

The actual MSE may depend on block dimension B: at larger B the coordinate
distribution is more concentrated (variance ~1/B), giving the Max-Lloyd
quantizer more to exploit. See Experimental plan.

**SORF approximation.** The 3-round SORF `HD₃·HD₂·HD₁` [5] provides log₂(B)
butterfly stages per round × 3 rounds = 3·log₂(B) total (18 at B=64, 24 at
B=256, 27 at B=512).
This is a rough heuristic for mixing quality — [5] does not analyze convergence
rate as a function of rounds × dimension. Empirical validation is needed.

**Fallback: dense rotation.** If SORF proves insufficient at the chosen B, use a
B × B random orthogonal matrix (QR of Gaussian). Storage at B=256: 256 KB per
block. For d=768 with k=3: 768 KB total. Amortizes for large columns (100K+
vectors). Each block must have an **independent** rotation matrix.

**Why not DCT?** The PDX implementation [pdx-impl] uses DCT (via FFTW) as a fast
rotation for ADSampling. DCT is O(B log B) and invertible, but it is a **fixed
structured transform**, not a random rotation — it does not produce the Beta
marginal distribution `(1-x²)^((B-3)/2)` (in block dimension B) that
TurboQuant's Max-Lloyd centroids are optimized for. ADSampling only needs
approximate coordinate independence
(for hypothesis-testing pruning), so DCT suffices there. TurboQuant needs a
specific known marginal distribution, so only random orthogonal rotations (QR or
SORF) are suitable.

**Shared rotation with ADSampling (speculative).** Both TurboQuant and
ADSampling apply a random orthogonal rotation to make coordinates independent.
If we integrate ADSampling-style dimension pruning (see Stage 3), the same
rotation could in principle serve both purposes. However, this is not automatic
under the Stage 2 block-decomposed design: ADSampling is formulated around a
single full-dimensional random projection whose coordinates can be sequentially
sampled, whereas Stage 2 introduces per-block rotations and per-block norm
weighting. Reusing one rotation across both systems should be treated as a
**future research direction** that requires new analysis or direct empirical
validation. If it proves viable, it would avoid rotating the data twice. The
query would also need to be rotated at query time with the same stored
transform.

#### Quantized-domain operations

All quantized operations read per-block norms from the internal child array:

- **L2 distance**: `‖a-b‖² = Σ_k ‖aₖ‖² + Σ_k ‖bₖ‖² - 2·Σ_k ‖aₖ‖·‖bₖ‖·
unit_dotₖ`. Primary ANN metric; reuses per-block dot product and norms.
- **Dot product**: `<a,b> ≈ Σ_k ‖aₖ‖·‖bₖ‖ · Σ_j centroids[code_aₖ[j]] ·
centroids[code_bₖ[j]]`.
- **Cosine similarity**: `cos(a,b) ≈ dot(a,b) / (‖a‖·‖b‖)` where
  `‖a‖ = √(Σ_k ‖aₖ‖²)`.
- **L2 norm**: `√(Σ_k ‖xₖ‖²)`. O(k) per vector — a regression from the
  current O(1) single-norm readthrough, but modest.

#### Encoding algorithm

```

Input: x ∈ ℝ^d, b_mse bits per coordinate, block_size B
k = d / B (exact division, no straggler for chosen B)
num_centroids = 2^b_mse

# Block split and normalize

for i in 0..k:
xᵢ = x[i*B .. (i+1)*B]
nᵢ = ‖xᵢ‖
if nᵢ > 0:
ûᵢ = xᵢ / nᵢ
else:
ûᵢ = zeros(B)

# MSE stage (per block, SORF rotation)

for i in 0..k:
if nᵢ > 0:
rᵢ = SORFᵢ(ûᵢ)
cᵢ[j] = nearest_centroid(rᵢ[j])
else:
cᵢ[j] = 0

Store (all as internal children):
codes (k × B per vector), norms (k per vector),
centroids (2^b_mse, shared), SORF signs (k × 3 × B, shared)

```

#### Decoding algorithm

```

for i in 0..k:
r̂ᵢ[j] = centroids[cᵢ[j]]
ûᵢ = SORF⁻¹ᵢ(r̂ᵢ)
x̂ᵢ = nᵢ × ûᵢ (nᵢ read from internal norms child)
x̃ = concat(x̂₀, ..., x̂ₖ₋₁)

```

### Stage 3: PDX dimension-major layout

Introduce a new `PDXArray` encoding type that wraps any `FixedSizeListArray`
with a dimension-major layout within groups of 64 vectors [4]. PDXArray is
**not TurboQuant-specific** — it is a general-purpose layout optimization for
any FixedSizeList of scalar elements (raw float vectors, scalar-quantized
vectors, TurboQuant codes, etc.).

**Changes vs. Stage 2:**

| Aspect           | Stage 2                                          | Stage 3                                                                         |
| ---------------- | ------------------------------------------------ | ------------------------------------------------------------------------------- |
| Codes child type | `FixedSizeListArray<u8>`                         | **`PDXArray<u8>`** (wraps FSL with transposed layout)                           |
| Codes detection  | N/A (codes always FSL)                           | **TQ checks child type**: FSL → row-major decode, PDXArray → un-transpose first |
| Distance kernel  | Per-vector loop with per-element centroid lookup | **SIMD-friendly 64-vector inner loop with distance-table lookup**               |
| Decode path      | Direct inverse SORF per vector                   | **PDXArray.to_fsl() first**, then inverse SORF                                  |

**Unchanged from Stage 2:** Block size B, centroid computation, norm storage,
SORF rotation, all encoding logic. The encode path produces row-major codes
(FSL), then the compressor wraps them in a PDXArray; the decode path converts
PDXArray back to FSL then decodes.

**PDXArray design:**

```

PDXArray<T> (general-purpose dimension-major layout for FixedSizeList)
├── metadata: { list_size, chunk_size (= 64) }
├── elements: PrimitiveArray<T> # transposed: 64 values per dim, contiguous
├── validity: ... # same as FSL validity

```

- `PDXArray::try_new(fsl)` — transposes a FixedSizeListArray into PDX layout
- `PDXArray::to_fsl()` — un-transposes back to row-major FSL (for decode,
  scalar_at, or non-aligned slice/take)
- `PDXArray::elements_for_dim(dim, chunk)` — O(1) access to a contiguous slice
  of 64 values for one dimension within one chunk
- Slice/take: un-transpose to FSL (simplest). Un-transpose cost is
  O(rows × list_size) per operation; consider 64-row-aligned fast paths for
  hot scan workloads. Preserving PDX layout is possible only for
  64-vector-aligned ranges.
- The cascade compressor treats PDXArray as a valid encoding of FSL-typed data.

**Benefits of PDXArray as a separate type:**

- PDX logic tested and maintained independently of TurboQuant
- Other encodings (raw float vectors, scalar quantization, future encodings)
  get PDX scan performance for free
- TurboQuant doesn't need an `is_pdx` metadata flag — it checks its codes
  child's type at runtime
- The distance kernel operates on PDXArray's dimension-contiguous slices

Within each 64-vector chunk, codes are stored dimension-major:

```

TQ block 0, dim 0: [v0 v1 v2 ... v63]
TQ block 0, dim 1: [v0 v1 v2 ... v63]
...
TQ block 0, dim (B - 1): [v0 v1 v2 ... v63]
TQ block 1, dim 0: [v0 v1 v2 ... v63]
...

```

The inner SIMD loop (64 vectors) has no inter-vector dependencies. TQ block
boundaries only affect where norm weighting occurs — they don't affect the
transpose.

**Quantized distance kernel (dot product):**

```rust
let dist_table = precompute_product_table(&centroids);
// At b_mse=4: 16×16 = 256 floats = 1KB, fits in L1

let mut distances = [0.0f32; 64];
let mut unit_dots = [0.0f32; 64];
let mut offset = 0;

for tq_block in 0..k {
    for dim in 0..B {
        let qd = query_codes[tq_block * B + dim];
        let row = &dist_table[qd as usize];
        for v in 0..64 {  // SIMD-friendly: no inter-vector deps
            unit_dots[v] += row[codes[offset] as usize];
            offset += 1;
        }
    }
    // Weight per-block unit-norm dot product by both vectors' block norms
    for v in 0..64 {
        distances[v] += query_norms[tq_block] * data_norms[v][tq_block]
                        * unit_dots[v];
        unit_dots[v] = 0.0;  // reset for next TQ block
    }
}
```

**Int8 layout variant.** The PDX implementation [pdx-impl] uses a different
tiling for int8 data: "4 dims × 16 vecs" to leverage VPDPBUSD/UDOT hardware
dot-product instructions (which process 4 unsigned×signed byte pairs per
operation). For TurboQuant codes at b_mse ≤ 8, codes are uint8 centroid indices,
so VPDPBUSD doesn't apply directly — we need the distance-table-lookup path
shown above. However, at b_mse=8 with high B, the Max-Lloyd centroids are
near-uniformly spaced (see GPU section), potentially enabling direct hardware
dot-product on the codes. Whether this requires a separate linear quantization
mode or works with the existing Max-Lloyd centroids is an empirical question. The
"4 dims × 16 vecs" layout would be a Stage 3 optimization to evaluate alongside
the "1 dim × 64 vecs" float-style layout.

**ADSampling integration.** The PDX dimension-pruning approach (ADSampling [4])
is complementary to TurboQuant's block structure. During a scan, the pruner
could evaluate partial distances after each TQ block (B dimensions) and skip
remaining blocks if the partial L2 distance already exceeds the candidate
threshold. This requires the per-block norm weighting to happen at block
boundaries (as shown in the kernel above), which our design already provides.

**Open design questions:**

- Should PDXArray live in `vortex-array` (general infrastructure) or
  `vortex-tensor` (vector-specific)?
- Should the cascade compressor automatically PDX-transpose FSL children when
  it detects a scan-heavy workload, or should PDX be opt-in?
- Should we support the "4 dims × 16 vecs" uint8 layout variant (for hardware
  dot-product) alongside the "1 dim × 64 vecs" float-style layout?

### QJL correction (deferred — experimental)

Based on community findings [8], QJL is deferred to after the MSE stages are
validated.

**Changes vs. MSE-only (if pursued):**

| Aspect                 | MSE-only                         | MSE + QJL                                                       |
| ---------------------- | -------------------------------- | --------------------------------------------------------------- |
| Bit budget             | All b bits → MSE (2^b centroids) | b-1 bits MSE + 1 bit QJL (2^(b-1) centroids)                    |
| Inner product estimate | Biased (MSE quantization noise)  | Unbiased (QJL correction; see TurboQuant_prod in [1])           |
| Additional children    | None                             | QJL signs, QJL residual norms, QJL projection params            |
| Encode cost            | SORF only                        | SORF + QJL projection (O(B²) for Gaussian, O(B log B) for SORF) |
| Decode cost            | Inverse SORF only                | Inverse SORF + QJL inverse projection                           |

If pursued, four strategies should be compared:

| Strategy             | Theoretical           | Speed            | Storage         |
| -------------------- | --------------------- | ---------------- | --------------- |
| Per-block Gaussian   | Correct (Lemma 4 [1]) | O(B²)/block      | k×B²×4 bytes    |
| Per-block SORF       | Approximate           | O(B log B)/block | k×3×B bits      |
| Full-dim padded SORF | Approximate           | O(d log d) total | 3×padded_d bits |
| MSE-only (no QJL)    | N/A                   | 0                | None            |

The paper's QJL uses Gaussian S (not SORF); Lemma 4 [1] (same camera-ready
numbering caveat as Theorem 1) is proved specifically
for Gaussian. SORF for QJL is an additional approximation (the
[current implementation][current-impl] uses SORF for QJL). Per-block QJL can
incur up to d/B times larger variance bound than full-dimension QJL (Lemma 4
[1]), depending on how query and residual energy are distributed across blocks.

Community reports indicate MSE-only often wins for KV-cache attention at all
tested bit widths [8]. Whether this extends to ANN ranking is an empirical
question (see Experimental plan); QJL may not be worth the complexity.

## Array layout

### Stage 1 (MSE-only single block)

```
TurboQuantArray
├── metadata: { dimension, b_mse, block_size (= padded_dim),
│               num_blocks (= 1) }
│
│  # Per-row children
├── codes: FixedSizeListArray<u8>           # list_size = padded_dim
│          (or PDXArray<u8> after Stage 3)
├── norms: PrimitiveArray<F>               # len = num_rows (F = f64 for f64, f32 otherwise)
│
│  # Shared children
├── centroids: PrimitiveArray<f32>          # len = 2^b_mse
├── mse_rotation_signs: PrimitiveArray<u8>  # len = 3 × padded_dim (bitpacked)
```

Same structure as the [current PR][current-impl] minus the 3 QJL slots, plus
the forward-compatible metadata fields and dtype-matching norms. The codes child
is `FixedSizeListArray` in Stages 1-2 and may be swapped to `PDXArray` in Stage
3 — TurboQuant checks the child type at runtime, not via a metadata flag.

### Stage 2 (block decomposition)

```
TurboQuantArray (self-contained, handles blocks internally)
├── metadata: { dimension, b_mse, block_size, num_blocks }
│
│  # Per-row children (sliced/taken on row operations)
├── codes: FixedSizeListArray<u8>           # list_size = k × B
│          (or PDXArray<u8> after Stage 3)
├── norms: PrimitiveArray<F>                # len = num_rows (k=1)
│      or  FixedSizeListArray<F>            # list_size = k (k>1)
│
│  # Shared children (cloned on row operations, not sliced)
├── centroids: PrimitiveArray<f32>          # len = 2^b_mse
├── mse_rotation_signs: PrimitiveArray<u8>  # len = k × 3 × B
```

## Compression ratio

For f32 input, b_mse bits MSE, k = d/B blocks, N vectors (for f64 input,
replace 32 with 64 in the norms row — ratios decrease accordingly):

| Component   | Bits per vector |
| ----------- | --------------- |
| MSE codes   | k × B × b_mse   |
| Block norms | k × 32          |

| Component  | Shared bits  |
| ---------- | ------------ |
| Centroids  | 2^b_mse × 32 |
| SORF signs | k × 3 × B    |

### Worked examples (f32, b_mse=5, N=1000)

| d             | B    | k   | Per-vec bits          | Ratio | Notes                    |
| ------------- | ---- | --- | --------------------- | ----- | ------------------------ |
| 768           | 256  | 3   | 3×256×5 + 3×32 = 3936 | 6.2×  | Block decomp; no padding |
| 1024          | 1024 | 1   | 1024×5 + 32 = 5152    | 6.4×  | Single block (= current) |
| 768 (current) | 1024 | 1   | 1024×5 + 32 = 5152    | 4.8×  | Padded; 33% overhead     |

Block decomposition improves the compression ratio for d=768 from ~4.8× to
~6.2× (about 29% higher ratio; equivalently, about 24% fewer compressed bits
per vector: 5152 → 3936). For d=1024 the encoding is identical to current.

**Shared overhead note:** centroids and SORF signs are amortized over N vectors;
for small N, per-column shared metadata is significant — report totals with and
without amortization when publishing ratios.

## Performance analysis

### Encode/decode throughput

SORF at B dimensions (heuristic — real cost is dominated by memory bandwidth
and constant factors): 3 × B × log₂(B) butterflies + 3 × B sign applications
per block (plus B normalization multiplies, omitted). For k blocks:

| B              | SORF FLOPs/block          | k (d=768) | Total MSE FLOPs |
| -------------- | ------------------------- | --------- | --------------- |
| 256            | 3×256×8 + 768 = 6,912     | 3         | 20,736          |
| 512            | 3×512×9 + 1536 = 15,360   | —         | —               |
| 1024 (current) | 3×1024×10 + 3072 = 33,792 | 1         | 33,792          |

Block decomposition at d=768 is ~40% fewer FLOPs than the current padded
approach, despite more blocks, because each block is smaller.

### Benchmarking plan

1. Encode/decode throughput: block TQ vs. current TQ at d=128, 768, 1024
2. Quantized cosine similarity: block vs. current
3. L2 norm readthrough: O(k) vs. O(1)
4. PDX scan throughput vs. row-major (Stage 3)

## Experimental plan

### Minimum dimension threshold

Test TurboQuant quality at d ∈ {32, 64, 96, 128, 256} to validate the scheme
minimum of 128:

- Compare TurboQuant MSE distortion and ANN recall@k against scalar
  quantization at matched bit rates (e.g., linear min-max quantization at the
  same bits-per-coordinate as TurboQuant's b_mse setting)
- Plot the crossover point: at what d does TurboQuant's recall@k drop below
  rate-matched scalar quantization?
- Test SORF coordinate distribution quality at each d (histogram vs. Beta)
- Measure overhead ratio (norm bits / total compressed bits) at each d

The scheme minimum should be set at the smallest d where TurboQuant reliably
beats rate-matched scalar quantization on recall@k across the benchmarking
datasets. The current proposal of 128 is conservative; experiments may justify
lowering to 64 or raising to 256.

### MSE quality vs. block size

- Compare actual normalized MSE at B ∈ {64, 128, 256, 512} vs. single-SORF at
  padded dimension, at bit widths b ∈ {2, 3, 4, 5, 8}
- Test SORF coordinate distribution at each B: histogram vs. analytical Beta
- Test 3, 4, 5 SORF rounds at each B
- Determine if the practical MSE constant is worse at smaller B

### QJL strategy comparison (if pursued)

- Per-block Gaussian QJL vs. per-block SORF QJL vs. full-dim padded SORF QJL
  vs. MSE-only
- Key metric: ANN recall@k on the datasets above (Contriever, OpenAI, SIFT)
- Per community findings for attention, MSE-only is expected to win [8]; ANN
  ranking is the key open question

### Benchmarking datasets

The current test suite uses i.i.d. Gaussian vectors as a theory anchor and
sanity check: for isotropic data, a random orthogonal transform is
distributionally neutral, which cleanly validates theoretical bounds. This is
not a universal "worst case" for all production workloads — heavy-tailed or
clustered embeddings can behave differently. Recent work
(VIBE [11]) argues that traditional benchmarks (SIFT, GloVe) are no longer
representative of modern ANN workloads.

**Recommended datasets:**

| Dataset                       | Dim    | Size   | Source           | Why                                                                                                                                       |
| ----------------------------- | ------ | ------ | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Contriever                    | 768    | ~1M    | PDX paper [4]    | Key non-power-of-2 target; real embeddings                                                                                                |
| OpenAI text-embedding-3-large | 1536   | ~1M    | Common in RAG    | High-d production embeddings                                                                                                              |
| SIFT                          | 128    | 1M     | Classic          | Low-d power-of-2 baseline, well-studied recall numbers                                                                                    |
| arXiv embeddings              | 768    | 2.25M  | PDX paper [4]    | Same dim as Contriever, larger scale                                                                                                      |
| DEEP                          | 96     | 10M    | Image embeddings | Large scale; d=96 < scheme min (128) and has no B ≥ 64 — requires explicit TurboQuantArray construction or benchmark-only scheme override |
| Synthetic Gaussian            | varies | varies | Internal         | Theory anchor / sanity check; not universal worst case                                                                                    |

**Metrics** (at b_mse ∈ {2, 3, 4, 5, 8}):

- Recall@10, Recall@100 (ANN ranking quality)
- Normalized MSE distortion (reconstruction quality)
- Inner product mean signed relative error (bias measurement)
- Encode/decode throughput (vectors/sec)

The Gaussian baseline validates that theoretical bounds hold. The real-embedding
datasets measure practical quality — which may be **better** than Gaussian
(structured data benefits more from rotation) or **worse** (if the data has
adversarial properties for the specific rotation).

### Dimensions with no qualifying B

Rare for common embedding dimensions (e.g., d=96). These fall back to the
Stage 1 padded single-block path (pad to next power-of-2, single SORF). No
block decomposition is attempted.

## Phasing

**Phase 1** — MSE-only single-block TurboQuant: Split the [current PR][current-impl]
to merge MSE-only (no QJL). This is a complete encoding for all dimensions
(with padding for non-power-of-2).

**Phase 2** — Block decomposition: Add block splitting for dimensions where a
valid B exists (greatest power-of-2 ≥ 64 dividing d). Per-block norms stored as
internal children. The `TurboQuantScheme::compress()` method must be updated to:
(a) choose B based on d, (b) split input into blocks, (c) normalize per-block,
(d) encode each block, and (e) store per-block norms as an internal child array.

**Phase 3** — PDXArray + scan kernels: Introduce `PDXArray` as a general-purpose
dimension-major layout for `FixedSizeListArray`. TurboQuant's codes child is
swapped from FSL to PDXArray by the compressor. Distance computation kernels
operate on PDXArray's dimension-contiguous slices.

**Phase 4** (experimental) — QJL: If the experimental plan shows QJL improves
recall@k beyond MSE-only, add per-block Gaussian or SORF QJL. Based on
KV-cache community reports [8], this may not be pursued.

## Practical recommendations

For common model dimensions, the most promising configurations are:

| Dimension              | Recommendation              | Rationale                                                                  |
| ---------------------- | --------------------------- | -------------------------------------------------------------------------- |
| 512, 1024, 2048, 4096  | Single-block MSE-only + PDX | B=d, no decomposition needed. Same as current TQ but with PDX scan layout. |
| 768, 1536, 3072        | 3-block MSE-only + PDX      | B=256 or 512. No padding waste. 3 blocks, shared centroids.                |
| No qualifying B (rare) | Padded single-block         | Fall back to Stage 1: pad to next power-of-2, single SORF.                 |

In all cases, MSE-only is the recommended starting point. QJL should only be
added if experiments demonstrate clear recall@k improvements for the target
workload.

## Future work: GPU decode and fused distance computation

The B-dim block structure maps naturally to GPU tile sizes and tensor cores.
For a batch of N vectors sharing the same rotation matrix R⁻¹:

```
decoded_batch = diag(norms) × R⁻¹ × codebook_lookup_batch(codes)
                                      ↑ B×N matrix
                               ↑ B×B × B×N = GEMM
```

The codebook gather + inverse rotation + norm scaling can be fused into a single
kernel using an IO-aware streaming pattern analogous to Flash-KMeans [6] — not
the same algorithm (Flash-KMeans is GPU k-means), but a similar systems goal:
reduce HBM traffic and avoid full materialization.
For distance computation without full decode, a precomputed (2^b_mse)²-entry
distance table fits in shared memory (1 KB at b_mse=4, 4 KB at b_mse=5); the
kernel streams code bytes from HBM with gather-reduce accumulation, using
4-8× less bandwidth than full float vectors.

At b_mse=8, codes are uint8 indices (0-255). Direct low-precision GEMM on
hardware accelerators (tensor cores on GPU, byte-dot-product instructions on
CPU) requires approximately linear
centroids — but at high B the Max-Lloyd centroids are already near-uniform
(the Beta distribution is highly concentrated, approaching Gaussian, for which
high-resolution optimal quantization is approximately uniform). Whether the
existing Max-Lloyd centroids are "linear enough" for hardware dot-product
instructions is an empirical question worth testing before introducing a
separate linear quantization mode.

## Integration with Vortex scan engine

TurboQuant's quantized-domain operations must integrate with Vortex's expression
evaluation and scan pushdown infrastructure. The current implementation provides
this via `ScalarFnVTable` implementations in `vortex-tensor`.

**Current integration path.** The `CosineSimilarity`, `DotProduct`, and `L2Norm`
scalar functions check whether their input storage arrays are TurboQuant-encoded
(via `TurboQuant::try_match()`). If both operands are TurboQuant and the
`ApproxOptions::Approximate` flag is set, the scalar function dispatches to the
quantized-domain kernel (e.g., `cosine_similarity_quantized_column`), bypassing
full decompression. Otherwise, it falls back to the exact path (decompress →
compute on floats).

**Stage 2 changes.** With block decomposition, the quantized kernels must be
updated to iterate over TQ blocks, weighting by per-block norms:

- `cosine_similarity_quantized_column`: currently computes a single unit-norm
  dot product per row pair. Must change to `Σ_k norm_a_k · norm_b_k ·
unit_dot_k / (‖a‖ · ‖b‖)` with `‖a‖ = √(Σ_k norm_a_k²)`.
- `dot_product_quantized_column`: same per-block weighting.
- `l2_norm`: currently returns the stored norm directly (O(1)). Must change to
  `√(Σ_k norm_k²)` — read the norms child (`PrimitiveArray` for k=1,
  `FixedSizeListArray` for k>1) and compute.
- Both operands must have the **same block size B**, compatible centroids (same
  `b_mse` and B-dim codebook), and **bit-identical MSE rotation parameters**
  (`mse_rotation_signs` and same SORF construction) for the quantized
  inner-product path to be valid. Two stored columns with different rotations
  must **fall back to exact** (decompress → float). The common **column vs.
  constant query** path avoids this: the query is re-encoded with the column's
  rotation and centroids at query time.

**Stage 3 changes.** The PDX distance kernel (shown in Stage 3 pseudocode) is a
new execution path that operates on `PDXArray`-typed codes. It should be exposed
as an alternative `ScalarFnVTable` implementation that activates when the codes
child is a `PDXArray` and the scan is over a contiguous 64-vector-aligned range.
For non-aligned ranges or single-vector access (`scalar_at`), the PDXArray is
converted to FSL first via `PDXArray::to_fsl()`.

**Expression tree integration.** The typical ANN scan expression is:

```
top_k(cosine_similarity(column, constant_query), k=10)
```

The `constant_query` is broadcast to match the column length. The
`CosineSimilarity` scalar function receives both the column (TurboQuant-encoded)
and the query (ConstantArray wrapping a single vector). For the quantized path,
the query is first encoded with the column's rotation and centroids to produce
query codes and query block norms, then the PDX kernel runs over the column's
codes without decompressing them.

## Migration and compatibility

TurboQuant has not shipped yet, so there are no existing files to migrate. We
can design the metadata for forward compatibility from day one.

**Strategy: single array ID, versioned metadata.** All stages use the same array
ID (`vortex.turboquant`). The metadata includes `block_size` and `num_blocks`
fields from Stage 1 onward. Stage 1 always writes `num_blocks=1`, but the field
exists so that Stage 2 decoders can read Stage 1 files without migration.

**Norms are always internal children.** The TurboQuant array is self-contained —
it stores norms as a child slot, not in a parent encoding. This means:

- Stage 1: norms child is `PrimitiveArray<F>`, one norm per vector (F = f64 for
  f64 input, f32 otherwise).
- Stage 2 with k=1 (power-of-2 dims): same as Stage 1, identical wire format.
- Stage 2 with k>1: norms child is `FixedSizeListArray<F>`, k norms per vector.

The decoder distinguishes k=1 from k>1 by reading `num_blocks` from metadata.
A k=1 decoder is backward-compatible with Stage 1 files. A k>1 decoder is a new
code path that only applies to files written by Stage 2+.

**Stage 3 (PDXArray) is additive.** PDX is not a TurboQuant metadata flag — it's
a separate array type (`PDXArray`) that wraps the codes child. Stage 1/2 files
have `FixedSizeListArray` codes; Stage 3 files have `PDXArray` codes. The
TurboQuant decoder checks the child type and un-transposes PDXArray on decode if
needed. `PDXArray` itself is registered as a new encoding, independent of
TurboQuant.

**Incremental shipping:**

| Stage        | Ships to users?  | Reads Stage 1 files?       | Notes                               |
| ------------ | ---------------- | -------------------------- | ----------------------------------- |
| 1 (MSE-only) | Yes, immediately | N/A (first version)        | New encoding, no backcompat concern |
| 2 (blocks)   | Yes              | Yes (k=1 is identical)     | k>1 files need Stage 2+ decoder     |
| 3 (PDX)      | Yes              | Yes (FSL codes still work) | PDX codes need PDXArray registered  |

Each stage is independently shippable. Users can upgrade incrementally. Files
written by earlier stages are always readable by later decoders.

## References

[1] Zandieh, A., Daliri, M., Hadian, M. and Mirrokni, V. "TurboQuant: Online
Vector Quantization with Near-optimal Distortion Rate." ICLR 2026.
arXiv:2504.19874, April 2025.

[2] Ailon, N. and Chazelle, B. "The Fast Johnson-Lindenstrauss Transform and
Approximate Nearest Neighbors." SIAM J. Comput. 39(1):302-322, 2009.

[3] Tropp, J.A. "Improved Analysis of the Subsampled Randomized Hadamard
Transform." Adv. Adaptive Data Analysis 3(1-2):115-126, 2011.

[4] Kuffo, L., Krippner, E. and Boncz, P. "PDX: A Data Layout for Vector
Similarity Search." SIGMOD '25. arXiv:2503.04422, March 2025.

[5] Yu, F.X., Suresh, A.T., Choromanski, K., Holtmann-Rice, D. and Kumar, S.
"Orthogonal Random Features." NeurIPS 2016. arXiv:1610.09072.

[6] Yang, S. et al. "Flash-KMeans: Fast and Memory-Efficient Exact K-Means."
arXiv:2603.09229, March 2026.

[7] Pathare, T. et al. "TurboQuant: Implementation Corrections, Production
Hardening, and Deployment Infrastructure." Eviox Tech Report v1.2.0,
March 2026. https://eviox.tech/nexus/eviox_turboquant_corrections_study.pdf

[8] Community TurboQuant implementation reports (primarily KV-cache attention):

- https://github.com/tonbistudio/turboquant-pytorch — MSE-only (V3) vs
  MSE+QJL (V2); reports MSE-only wins for attention and generation quality.
- https://github.com/ggml-org/llama.cpp/discussions/20969 — TurboQuant
  discussion; quantized attention analysis and MSE vs Prod comparison.
- https://github.com/0xSero/turboquant — Triton kernels; paper validation.
- https://github.com/scos-lab/turboquant — Reference reproduction; MSE vs
  Prod/QJL comparison.
  Multiple groups report MSE-only beating MSE+QJL for attention metrics at tested
  bit widths. ANN ranking conclusions remain preliminary pending dedicated
  benchmarks.

[9] Jégou, H., Douze, M. and Schmid, C. "Product Quantization for Nearest
Neighbor Search." IEEE Trans. PAMI 33(1):117-128, 2011.

[10] Ge, T., He, K., Ke, Q. and Sun, J. "Optimized Product Quantization."
IEEE Trans. PAMI 36(4):744-755, 2014.

[11] Jääsaari, E., Hyvönen, V., Ceccarello, M., Roos, T. and Aumüller, M.
"VIBE: Vector Index Benchmark for Embeddings." arXiv:2505.17810, May 2025.
