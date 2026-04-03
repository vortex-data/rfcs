# Block-Decomposed TurboQuant with PDX Layout

**Authors:** Will Manning
**Status:** Proposal
**Date:** 2026-04-02

## Summary

We propose evolving the [TurboQuant vector quantization encoding][current-impl]
in three stages:

1. **MSE-only TurboQuant** (immediate): merge the current PR as an MSE-only
   encoding. This is a complete, self-contained building block.
2. **Block decomposition** (next): for non-power-of-2 dimensions, split into
   blocks of size B = the largest power-of-2 ≥ 64 that divides d. For
   power-of-2 dimensions, B = d (single block, same as current). Per-block
   norms stored as internal children.
3. **PDX layout** (later): transpose codes into dimension-major order within
   groups of 64 vectors for SIMD scan performance.

QJL correction is deferred to a later stage and may ultimately be dropped.
Community findings from 6+ independent TurboQuant implementations consistently
show that MSE-only outperforms MSE+QJL for attention and ANN ranking in
practice [8].

[current-impl]: https://github.com/vortex-data/vortex/pull/7167

## Background

### TurboQuant

TurboQuant [1] is a lossy vector quantization algorithm for high-dimensional
embeddings. It works by:

1. Randomly rotating a unit-norm vector so that each coordinate follows a known
   marginal distribution — specifically `(1 - x²)^((d-3)/2)` on [-1, 1], a
   concentrated Beta distribution (Lemma 1 in [1]).
2. Applying an MSE-optimal scalar quantizer (Max-Lloyd centroids) independently
   to each coordinate.
3. Optionally adding a 1-bit QJL (Quantized Johnson-Lindenstrauss) correction
   on the residual for unbiased inner product estimation (Theorem 2 in [1]).

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
| Indexing time          | Zero (codebook precomputed from distribution)                   | Requires training pass over data                         |
| Bits per sub-vector    | Scalar: b bits per coordinate                                   | Vector: typically 8 bits per sub-vector (256 codewords)  |

TurboQuant trades PQ's flexibility (data-dependent codebooks can exploit
structure) for data-obliviousness (no training, provable bounds, zero indexing
time). For uniformly distributed embeddings, TurboQuant's analytically optimal
centroids should match or exceed PQ's learned codebooks. For highly structured
data, PQ may still win empirically.

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
extension arrays with non-nullable float elements and dimension ≥ 3, using the
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

Multiple independent TurboQuant implementations have converged on a
significant practical finding: **MSE-only consistently outperforms MSE+QJL for
attention and ANN ranking**. The mechanism is a variance-bias tradeoff:
TurboQuant's QJL correction eliminates bias but increases variance, and softmax
attention (and cosine/L2 ranking) amplifies variance more than bias. At the same
total bit budget, allocating all bits to MSE (more centroids, lower variance)
beats splitting between MSE + QJL (fewer centroids + 1-bit correction). This has
been confirmed by 6+ groups across Python, C, and Rust implementations [8].

This finding strongly supports making MSE-only the default strategy for our
columnar storage use case (ANN search, cosine similarity ranking).

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
rather than dimensions, achieving on average 2× speedups over SIMD-optimized
row-major kernels on modern CPUs. The block size of 64 is empirically optimal
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

For each dimension d, choose B = the largest power-of-2 ≥ 64 that evenly
divides d. This eliminates stragglers entirely for common embedding dimensions:

| Dimension d | Block size B | Blocks k | Notes                       |
| ----------- | ------------ | -------- | --------------------------- |
| 512         | 512          | 1        | Single block (= current TQ) |
| 768         | 256          | 3        | Largest dividing power-of-2 |
| 1024        | 1024         | 1        | Single block                |
| 1536        | 512          | 3        |                             |
| 2048        | 2048         | 1        | Single block                |
| 3072        | 1024         | 3        |                             |
| 4096        | 4096         | 1        | Single block                |

**Key observations:**

- **Power-of-2 dimensions** (512, 1024, 2048, 4096) use B = d — a single block,
  identical to the current implementation except with PDX underneath (Stage 3).
  No block decomposition overhead, no per-block norms. These dimensions are
  already well-served by the current design.
- **Non-power-of-2 dimensions** (768, 1536, 3072) decompose into k=3 blocks at
  B=256 or B=512. Zero padding waste. Each block has its own SORF rotation and
  shares a single centroid set.
- **Stragglers are eliminated** for all common embedding dimensions. Dimensions
  that are not multiples of 64 (e.g., 100, 200) would need straggler handling,
  but these are rare in practice for modern model architectures.
- **The SORF approximation at B=256+ is expected to be adequate**: 3 rounds at
  B=256 provides 24 butterfly stages, and at B=512 provides 27 — both comparable
  to the current B=1024 (30 stages). This needs empirical validation; see
  Experimental plan.

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

**Unchanged from current PR:** SORF rotation, Max-Lloyd centroids,
zero-padding for non-power-of-2, slice/take/scalar_at pushdowns, quantized
cosine similarity and dot product, compression scheme integration, minimum dim=3.

**Added to metadata (for forward compat):** `block_size: u32` (always =
padded_dim), `num_blocks: u32` (always = 1). These fields are inert in Stage 1
but enable Stage 2 decoders to read Stage 1 files. (PDX is handled via the
codes child type, not a metadata flag — see Stage 3.)

This is a complete, useful encoding for all dimensions. Power-of-2 dimensions
have zero padding waste; non-power-of-2 dimensions have the padding overhead
described above.

### Stage 2: Block decomposition

For non-power-of-2 dimensions, split into blocks of size B (as determined by the
table above). Each full block gets an independent B-dim SORF rotation.

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

The paper's MSE bound (Theorem 1 in [1]) is:

```
E[‖x - x̂‖² / ‖x‖²] ≤ (√3 · π / 2) / 4^b ≈ 2.72 / 4^b
```

**Crucially, Theorem 1 is proved for true random orthogonal matrices (QR of
Gaussian), not SORF.** Our SORF is an approximation. The bound holds exactly
only with a true random orthogonal rotation or with empirical SORF validation
(see Experimental plan).

Assuming the per-block MSE bound holds, for a vector split into blocks:

```
‖x - x̂‖² / ‖x‖² = Σ_k (‖xₖ‖² / ‖x‖²) × (‖xₖ - x̂ₖ‖² / ‖xₖ‖²)
                   ≤ MSE_bound × Σ_k (‖xₖ‖² / ‖x‖²) = MSE_bound
```

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
marginal distribution `(1-x²)^((d-3)/2)` that TurboQuant's Max-Lloyd centroids
are optimized for. ADSampling only needs approximate coordinate independence
(for hypothesis-testing pruning), so DCT suffices there. TurboQuant needs a
specific known marginal distribution, so only random orthogonal rotations (QR or
SORF) are suitable.

**Shared rotation with ADSampling.** Both TurboQuant and ADSampling apply a
random orthogonal rotation to make coordinates independent. If we integrate
ADSampling-style dimension pruning (see Stage 3), the same rotation could serve
both purposes: producing the Beta distribution for quantization AND enabling
hypothesis-testing for early pruning. This would avoid rotating the data twice.
Note that the query must also be rotated at query time with the same rotation
matrix (stored as a shared child); ADSampling already requires this.

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
k = d / B  (exact division, no straggler for chosen B)
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
    x̂ᵢ = nᵢ × ûᵢ                    (nᵢ read from internal norms child)
x̃ = concat(x̂₀, ..., x̂ₖ₋₁)
```

### Stage 3: PDX dimension-major layout

Introduce a new `PDXArray` encoding type that wraps any `FixedSizeListArray`
with a dimension-major layout within groups of 64 vectors [4]. PDXArray is
**not TurboQuant-specific** — it is a general-purpose layout optimization for
any FixedSizeList of scalar elements (raw float vectors, scalar-quantized
vectors, TurboQuant codes, etc.).

**Changes vs. Stage 2:**

| Aspect           | Stage 2                                          | Stage 3                                                           |
| ---------------- | ------------------------------------------------ | ----------------------------------------------------------------- |
| Codes child type | `FixedSizeListArray<u8>`                         | **`PDXArray<u8>`** (wraps FSL with transposed layout)             |
| TQ metadata      | `is_pdx` field                                   | **Removed** — TQ checks if codes child is PDXArray                |
| Distance kernel  | Per-vector loop with per-element centroid lookup | **SIMD-friendly 64-vector inner loop with distance-table lookup** |
| Decode path      | Direct inverse SORF per vector                   | **PDXArray.to_fsl() first**, then inverse SORF                    |

**Unchanged from Stage 2:** Block size B, centroid computation, norm storage,
SORF rotation, all encoding logic. The encode path produces row-major codes
(FSL), then the compressor wraps them in a PDXArray; the decode path converts
PDXArray back to FSL then decodes.

**PDXArray design:**

```
PDXArray<T> (general-purpose dimension-major layout for FixedSizeList)
├── metadata: { list_size, chunk_size (= 64) }
├── elements: PrimitiveArray<T>    # transposed: 64 values per dim, contiguous
├── validity: ...                  # same as FSL validity
```

- `PDXArray::try_new(fsl)` — transposes a FixedSizeListArray into PDX layout
- `PDXArray::to_fsl()` — un-transposes back to row-major FSL (for decode,
  scalar_at, or non-aligned slice/take)
- `PDXArray::elements_for_dim(dim, chunk)` — O(1) access to a contiguous slice
  of 64 values for one dimension within one chunk
- Slice/take: un-transpose to FSL (simplest). Preserving PDX layout is possible
  only for 64-vector-aligned ranges.
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
TQ block 0, dim 0:        [v0 v1 v2 ... v63]
TQ block 0, dim 1:        [v0 v1 v2 ... v63]
...
TQ block 0, dim (B - 1):  [v0 v1 v2 ... v63]
TQ block 1, dim 0:        [v0 v1 v2 ... v63]
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
| Inner product estimate | Biased (MSE quantization noise)  | Unbiased (QJL correction, Theorem 2 [1])                        |
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

The paper's QJL uses Gaussian S (not SORF); Lemma 4 [1] is proved specifically
for Gaussian. SORF for QJL is an additional approximation (the
[current implementation][current-impl] uses SORF for QJL). Per-block QJL has
d/B times more variance than full-dimension QJL (Lemma 4 [1]).

The community consensus is that MSE-only likely wins for ANN ranking at all
bit widths, so QJL may not be worth the complexity.

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

| d             | B    | k   | Per-vec bits          | Ratio | Notes                      |
| ------------- | ---- | --- | --------------------- | ----- | -------------------------- |
| 768           | 256  | 3   | 3×256×5 + 3×32 = 3936 | 6.2×  | Block decomp; zero padding |
| 1024          | 1024 | 1   | 1024×5 + 32 = 5152    | 6.4×  | Single block (= current)   |
| 768 (current) | 1024 | 1   | 1024×5 + 32 = 5152    | 4.8×  | Padded; 33% overhead       |

Block decomposition improves d=768 from 4.8× to 6.2× — a 30% storage
improvement. For d=1024 the encoding is identical to current.

## Performance analysis

### Encode/decode throughput

SORF at B dimensions: 3 × B × log₂(B) butterflies + 3 × B sign applications
per block (plus B normalization multiplies, omitted for simplicity). For k
blocks:

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
- Per community findings, MSE-only is expected to win [8]

### Benchmarking datasets

The current test suite uses i.i.d. Gaussian vectors, which is a pessimistic
baseline for TurboQuant: real embeddings have structure (clusters, anisotropy)
that rotation-based quantization can exploit, while Gaussian vectors are already
rotationally invariant (the rotation is a no-op in distribution). Recent work
(VIBE [11]) argues that traditional benchmarks (SIFT, GloVe) are no longer
representative of modern ANN workloads.

**Recommended datasets:**

| Dataset                       | Dim    | Size   | Source           | Why                                                    |
| ----------------------------- | ------ | ------ | ---------------- | ------------------------------------------------------ |
| Contriever                    | 768    | ~1M    | PDX paper [4]    | Key non-power-of-2 target; real embeddings             |
| OpenAI text-embedding-3-large | 1536   | ~1M    | Common in RAG    | High-d production embeddings                           |
| SIFT                          | 128    | 1M     | Classic          | Low-d power-of-2 baseline, well-studied recall numbers |
| arXiv embeddings              | 768    | 2.25M  | PDX paper [4]    | Same dim as Contriever, larger scale                   |
| DEEP                          | 96     | 10M    | Image embeddings | Large scale                                            |
| Synthetic Gaussian            | varies | varies | Internal         | Pessimistic baseline; validates theoretical bounds     |

**Metrics** (at b_mse ∈ {2, 3, 4, 5, 8}):

- Recall@10, Recall@100 (ANN ranking quality)
- Normalized MSE distortion (reconstruction quality)
- Inner product mean signed relative error (bias measurement)
- Encode/decode throughput (vectors/sec)

The Gaussian baseline validates that theoretical bounds hold. The real-embedding
datasets measure practical quality — which may be **better** than Gaussian
(structured data benefits more from rotation) or **worse** (if the data has
adversarial properties for the specific rotation).

### Straggler handling (if needed)

Rare for common dimensions. If encountered: zero-pad to B (simplest). Follow-up:
dense rotation at actual dimension.

## Phasing

**Phase 1** — MSE-only single-block TurboQuant: Split the [current PR][current-impl]
to merge MSE-only (no QJL). This is a complete encoding for all dimensions
(with padding for non-power-of-2).

**Phase 2** — Block decomposition: Add block splitting for non-power-of-2
dimensions. B = largest power-of-2 ≥ 64 dividing d. Per-block norms stored as
internal children. The `TurboQuantScheme::compress()` method must be updated to:
(a) choose B based on d, (b) split input into blocks, (c) normalize per-block,
(d) encode each block, and (e) store per-block norms as an internal child array.

**Phase 3** — PDXArray + scan kernels: Introduce `PDXArray` as a general-purpose
dimension-major layout for `FixedSizeListArray`. TurboQuant's codes child is
swapped from FSL to PDXArray by the compressor. Distance computation kernels
operate on PDXArray's dimension-contiguous slices.

**Phase 4** (experimental) — QJL: If the experimental plan shows QJL improves
recall@k beyond MSE-only, add per-block Gaussian or SORF QJL. Based on
community findings, this may not be pursued.

## Practical recommendations

For common model dimensions, the most promising configurations are:

| Dimension             | Recommendation              | Rationale                                                                  |
| --------------------- | --------------------------- | -------------------------------------------------------------------------- |
| 512, 1024, 2048, 4096 | Single-block MSE-only + PDX | B=d, no decomposition needed. Same as current TQ but with PDX scan layout. |
| 768, 1536, 3072       | 3-block MSE-only + PDX      | B=256 or 512. Zero padding waste. 3 blocks, shared centroids.              |
| Arbitrary d (rare)    | Padded single-block         | Fall back to current approach. Padding overhead bounded by B-1 dims.       |

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
kernel following the double-buffered streaming pattern from Flash-KMeans [6].
For distance computation without full decode, a precomputed (2^b_mse)²-entry
distance table fits in shared memory (1 KB at b_mse=4, 4 KB at b_mse=5); the
kernel streams code bytes from HBM with gather-reduce accumulation, using
4-8× less bandwidth than full float vectors.

At b_mse=8, codes are uint8 indices (0-255). Direct int8 tensor core GEMM
(using codes as the unsigned operand in VPDPBUSD) requires approximately linear
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
  `√(Σ_k norm_k²)` — read the norms FSL child and compute.
- Both operands must have the **same block size B** and compatible centroids for
  the quantized path to apply. If block sizes differ, fall back to exact.

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
March 2026.

[8] Community TurboQuant implementations and findings. Key sources:
tonbistudio/turboquant-pytorch (PyTorch, V3 MSE-only findings),
ggml-org/llama.cpp#20969 (C/C++, quantized attention analysis),
0xSero/turboquant (Triton kernels), vivekvar-dl/turboquant (pip package),
scos-lab/turboquant (reference reproduction). Consensus: MSE-only beats
MSE+QJL for attention and ANN ranking at all tested bit widths.

[9] Jégou, H., Douze, M. and Schmid, C. "Product Quantization for Nearest
Neighbor Search." IEEE Trans. PAMI 33(1):117-128, 2011.

[10] Ge, T., He, K., Ke, Q. and Sun, J. "Optimized Product Quantization."
IEEE Trans. PAMI 36(4):744-755, 2014.

[11] Kuffo, L. et al. "VIBE: Vector Index Benchmark for Embeddings."
arXiv:2505.17810, May 2025.
