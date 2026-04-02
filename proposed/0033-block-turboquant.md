# Block-Decomposed TurboQuant with PDX Layout

**Authors:** Will Manning
**Status:** Proposal
**Date:** 2026-04-02

## Summary

We propose modifying the [TurboQuant vector quantization encoding][current-impl]
to use uniform blocks of a tunable power-of-2 size B (to be experimentally
determined; expected range 64-256) with per-block SORF rotations for the MSE
stage. The QJL correction uses per-block Gaussian projection matrices as
prescribed by [1], not SORF — this eliminates the need for power-of-2 padding in
the QJL stage. The lowest-level TurboQuant array operates only on unit-norm
vectors, with norms externalized. In a second stage, a PDX-style
dimension-major code layout within groups of 64 vectors enables SIMD-friendly
vectorized distance computation. The TQ block size B (for quantization quality)
and the PDX vector group size (always 64, for SIMD utilization) are independent
parameters.

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

The paper prescribes a full random orthogonal rotation (QR of Gaussian) for the
MSE stage — O(d²) storage and O(d²) per-vector. For the QJL stage, the paper
uses a random Gaussian projection matrix S with i.i.d. N(0,1) entries (not an
orthogonal rotation); this distinction matters for the unbiasedness proof.

Our [current implementation][current-impl] substitutes a 3-round Structured
Orthogonal Random Features (SORF) transform `HD₃·HD₂·HD₁` [5] for both the MSE
rotation and the QJL projection, giving O(d) storage and O(d log d) per-vector.
The 3-round SORF construction was introduced for kernel approximation [5] and
approximates a random orthogonal matrix. Note that this is distinct from the
single-round SRHT (`R·H·D`) analyzed by Tropp [3] and the FJLT (`P·H·D`) of
Ailon-Chazelle [2], both of which are dimensionality-reducing projections. Using
SORF for QJL (where the paper prescribes Gaussian S) is an additional
approximation that may affect the unbiasedness guarantee of Theorem 2.

### Current limitations

The SORF requires power-of-2 input dimension. For non-power-of-2 dimensions
(e.g., 768-d embeddings from common transformer models), the input is
zero-padded to the next power of 2 (1024). This causes:

- **33% wasted storage** for 768-d vectors: 1024 quantized codes stored per
  vector when only 768 carry information.
- **QJL variance inflation**: The QJL correction also uses a padded SORF. Of
  the 1024 QJL sign bits, 256 encode information about the zero-padded region
  rather than the actual residual. The estimator remains unbiased in expectation,
  but the wasted measurement budget inflates variance. Empirically, this
  manifests as ~23% mean signed relative error in inner product estimation for
  768-d vs. ~11% for power-of-2 dimensions.
- **No scan-optimized layout**: Codes are stored in row-major order (all codes
  for one vector contiguous), which prevents SIMD-over-vectors distance
  computation.

### PDX

PDX [4] is a data layout for vector similarity search that stores dimensions in
a vertical (dimension-major) layout within fixed-size blocks of 64 vectors. This
enables the compiler to auto-vectorize the inner distance loop over vectors
rather than dimensions, achieving on average 2x speedups over SIMD-optimized
row-major kernels (up to 5.5x at low dimensions) on modern CPUs. The block size
of 64 is empirically optimal across AVX-512, AVX2, and NEON architectures [4].

## Proposal

### Stage 1: Tunable block decomposition

Partition d dimensions into `⌈d/B⌉` blocks of B dimensions each, where B is a
power-of-2 block size (configurable; expected range 64-256, to be determined
experimentally). The last block ("straggler") may have fewer than B dimensions
— see Straggler handling below. Each full block gets an independent B-dim SORF
rotation for the MSE stage.

Larger B gives better quantization quality (more concentrated coordinate
distribution, better SORF mixing) but more per-block norm overhead. Smaller B
gives more blocks (more norms) but each block has higher practical variance.
The optimal B should be determined experimentally — see Experimental plan.

**Key properties:**

- **B is a power of 2**, so every full block has zero SORF padding waste.
  Common embedding dimensions are divisible by 64 and 128; most are also
  divisible by 256. The straggler case is rare for well-chosen B.
- **One shared centroid set** for full blocks. All full blocks use the same
  B-dim marginal distribution, so a single Max-Lloyd codebook serves every full
  block.
- **Unit-norm assumption.** The TurboQuant array operates only on pre-normalized
  sub-vectors. Per-block norms are externalized (see Norm architecture below).
- **Per-block SORF rotation signs.** Each block's MSE SORF is independent
  (different seed). Signs are 3 × B bits per block. For d=768 with B=128:
  6 blocks × 3 × 128 = 2304 bits = 288 bytes.

#### Straggler handling

If `d % B ≠ 0`, the final block has `d mod B` dimensions. Initially, the
straggler is **zero-padded to B** and uses the same SORF and centroids as all
other blocks. This is the simplest approach and reuses all existing
infrastructure. The padding waste is bounded by B-1 dimensions — much smaller
than the current full-dimension padding (up to d-1 dimensions).

As a follow-up experiment, we will test using a **dense random orthogonal
matrix** (QR of Gaussian) at the straggler's actual dimension with its own
centroid set. This eliminates the straggler's padding waste entirely and may
allow increasing B (since the straggler cost is bounded). This requires separate
storage for the straggler rotation matrix and centroids.

#### Zero-norm sub-vectors

When splitting a vector into B-dim blocks, some blocks may have zero norm
(e.g., sparse embeddings, or dimensions that are structurally unused). The
encoding must handle ‖xₖ‖ = 0 explicitly: skip rotation and quantization for
that block, store norm = 0, and decode as all zeros. The codes for a zero-norm
block are arbitrary (never read during decode).

#### Theoretical MSE bound

The paper's MSE bound (Theorem 1 in [1]) is:

```
E[‖x - x̂‖² / ‖x‖²] ≤ (√3 · π / 2) / 4^b
```

This bound is dimension-independent — it depends only on the bit width `b`. For
a vector split into blocks with per-block normalization (assuming ‖xₖ‖ > 0):

```
‖x - x̂‖² / ‖x‖² = Σ_k (‖xₖ‖² / ‖x‖²) × (‖xₖ - x̂ₖ‖² / ‖xₖ‖²)
                   ≤ MSE_bound × Σ_k (‖xₖ‖² / ‖x‖²)
                   = MSE_bound
```

The Pythagorean identity `Σ_k ‖xₖ‖² = ‖x‖²` ensures the weights sum to 1.
Blocks with ‖xₖ‖ = 0 contribute zero error and are excluded from the sum.

**Practical MSE vs. theoretical bound.** The bound `(√3π/2)/4^b` is a worst-case
upper bound that is dimension-independent. However, the **actual** MSE depends
on the block dimension B. At higher dimensions, the coordinate distribution
after rotation is more concentrated (variance ~1/B), and the Max-Lloyd quantizer
exploits this concentration — the actual MSE can be well below the bound. At
smaller B, the distribution is wider and the quantizer has less concentration to
exploit.

Block decomposition preserves the theoretical bound but may give worse
**practical** MSE than a single full-dimension rotation, because each B-dim
block operates at a less favorable point on the distortion curve. This must be
quantified empirically (see Experimental plan).

**SORF approximation caveat.** Theorems 1 and 2 in [1] are proved for true
random orthogonal matrices (QR of Gaussian), not SORF. The 3-round SORF
construction `HD₃·HD₂·HD₁` [5] is a structured approximation. The
approximation quality depends on dimension: 3 rounds provides 3 × log₂(B)
mixing stages (18 at B=64, 21 at 128, 24 at 256, 30 at 1024). Empirical
validation is needed for each candidate B — see Experimental plan.

**Fallback: dense rotation.** If SORF proves insufficient at the chosen B, use a
B × B random orthogonal matrix (QR of Gaussian) instead. Storage at B=128:
128×128×4 = 64 KB per block. With k blocks for MSE: total = k × 64 KB. For
d=768, B=128 (k=6): 384 KB of MSE rotation matrices. This is significant for
small columns (1K vectors at 3 MB uncompressed ≈ 13% overhead) but amortizes to
negligible for large columns (100K+ vectors). Each block must have an
**independent** rotation matrix (different seeds) to avoid correlation between
quantization errors across blocks.

#### QJL correction

The QJL stage in the TurboQuant paper [1] uses a random Gaussian projection
matrix S ∈ ℝ^(d×d) with i.i.d. N(0,1) entries — **not** an orthogonal rotation.
The unbiasedness proof (Lemma 4 in [1]) and the scale factor `√(π/2)/d` are
derived specifically for Gaussian S. The [current implementation][current-impl]
uses SORF instead, which is an additional approximation.

The QJL correction has several possible strategies, differing in projection
type (Gaussian vs SORF), scope (per-block vs full-dimension), and whether QJL
is used at all. The choice significantly affects correctness guarantees,
variance, storage, and speed.

**Per-block Gaussian QJL** is the paper-correct approach: each block gets an
independent Gaussian projection Sₖ ∈ ℝ^(B×B) with i.i.d. N(0,1) entries. Since
Gaussian matrices work at any dimension, no padding is needed for the QJL stage.
Each block's QJL is provably unbiased by Lemma 4 in [1], and the sum over
blocks is also unbiased: `E[<y, correction>] = <y, r>`. However, the per-block
variance is **d/B times higher** than full-dimension QJL:

```
Per-block (B-dim):   Var[<y, correction>] ≤ (π / (2B)) × ‖y‖²
Full-dim (d-dim):    Var[<y, correction>] ≤ (π / (2d)) × ‖y‖²
```

At d=768, B=128: 6× more variance. Storage: B×B×4 bytes per block (384 KB for
k=6 at B=128). Encode/decode cost: O(B²) matmul per block.

**Per-block SORF QJL** substitutes a B-dim SORF (`HD₃·HD₂·HD₁` [5]) for the
Gaussian matrix. This is NOT theoretically justified — Lemma 4 requires Gaussian
or Haar-orthogonal S, and SORF is only an approximation to Haar measure.
However, the [current implementation][current-impl] already uses SORF for QJL at
d=1024 with acceptable results (~11% mean relative error for power-of-2 dims),
demonstrating practical viability. The tradeoff vs Gaussian is compelling:
O(B log B) speed (5× faster at B=128), O(B) storage (over 1000× less). Quality
at B=128 needs validation — with only 21 mixing stages, the approximation to
Haar measure is weaker than at d=1024 (30 stages).

**Full-dimension padded SORF QJL** applies a single SORF at the padded
dimension (e.g., 1024 for d=768) to the full residual vector `r = x - x̂`,
matching the [current implementation][current-impl]. The higher dimension gives
better SORF-to-Haar convergence (30 mixing stages at d=1024 vs 21 at B=128) and
full-dimension variance `(π/(2d))·‖y‖²`, but wastes `(padded_d - d)/padded_d`
of the sign bits on zero-padded coordinates (25% at 768→1024). This approach
requires computing the full residual from all blocks before applying QJL,
adding a full-dimension decode step to the encode path.

**MSE-only (no QJL)** skips the QJL correction entirely. Inner product
estimates are biased but this may be acceptable for ANN ranking where relative
ordering matters more than absolute accuracy. Eliminates all QJL storage and
computation.

**QJL strategy options** (to be experimentally compared):

| Strategy             | Theoretical       | Variance       | Padding waste | Storage         | Speed (per block) |
| -------------------- | ----------------- | -------------- | ------------- | --------------- | ----------------- |
| Per-block Gaussian   | Correct (Lemma 4) | (π/(2B))·‖y‖²  | None          | k×B²×4 bytes    | O(B²)             |
| Per-block SORF       | Approximate       | ~(π/(2B))·‖y‖² | None          | k×3×B bits      | O(B log B)        |
| Full-dim padded SORF | Approximate       | ~(π/(2d))·‖y‖² | (pad-d)/pad   | 3×padded_d bits | O(d log d)        |
| MSE-only             | N/A               | N/A            | N/A           | None            | 0                 |

#### Norm architecture

The TurboQuant array itself operates only on unit-norm B-dim sub-vectors. Norms
are externalized into a separate child array, following the pattern established
by the NormVector encoding (PR #7251).

The per-block norms are stored as a single `FixedSizeListArray<F>` with
`list_size = num_blocks`, where `F` matches or widens the input element type:

| Input dtype | Norm dtype | Rationale                                      |
| ----------- | ---------- | ---------------------------------------------- |
| f16         | f32        | f16 has insufficient range/precision for norms |
| f32         | f32        | Same type                                      |
| f64         | f64        | Preserve full precision                        |

Norms are stored as plain child arrays — TurboQuant does not apply any secondary
encoding to them. The cascading compressor treats norms like any other float
column and is free to re-encode them with ALP, Pco, FastLanes, or other float
compression schemes.

#### Quantized-domain operations with per-block norms

All quantized-domain operations require reading the block norms for both
operands. This is a cost increase vs. the [current implementation][current-impl]
which stores a single global norm per vector.

- **L2 norm**: Read block norms, compute `√(Σ_k ‖xₖ‖²)`. O(k) per vector — a
  regression from the current O(1) single-norm readthrough, but modest compared
  to full decompression.
- **Dot product**: `<a, b> ≈ Σ_k ‖aₖ‖·‖bₖ‖ · Σ_j centroids[code_aₖ[j]] ·
centroids[code_bₖ[j]]`. Per-block: compute unit-norm quantized dot product
  (sum of B centroid products), then weight by both vectors' block norms.
- **Cosine similarity**: `cos(a, b) ≈ (Σ_k ‖aₖ‖·‖bₖ‖·unit_dotₖ) /
(√(Σ_k ‖aₖ‖²) · √(Σ_k ‖bₖ‖²))`. Requires global norms reconstructed from
  block norms. The norms tensor should be read once per scan query and cached.

#### Encoding algorithm

```
Input: x ∈ ℝ^d, bit_width b, block_size B (power of 2)
k = ⌈d/B⌉

# Block split and normalize
for i in 0..k:
    xᵢ = x[i*B .. min((i+1)*B, d)], zero-pad to B if straggler
    nᵢ = ‖xᵢ‖
    if nᵢ > 0:
        ûᵢ = xᵢ / nᵢ
    else:
        ûᵢ = zeros(B)               (zero-norm block: skip quantization)

# MSE stage (per block, SORF rotation)
for i in 0..k:
    if nᵢ > 0:
        rᵢ = SORFᵢ(ûᵢ)             (3-round HD, independent per block)
        cᵢ[j] = nearest_centroid(rᵢ[j])  (shared codebook)
    else:
        cᵢ[j] = 0

Store: codes (k × B per vector), block_norms (k per vector),
       centroids (shared), SORF signs (k × 3 × B, shared across vectors)

# QJL stage (optional, one of four strategies)
for i in 0..k:
    if nᵢ > 0:
        x̂ᵢ = decode_mse_block(cᵢ, centroids, SORFᵢ)
        rᵢ = ûᵢ - x̂ᵢ                (per-block unit-norm residual)
        γᵢ = ‖rᵢ‖
        sᵢ = sign(Projᵢ × rᵢ)       (see QJL strategy options)
    else:
        γᵢ = 0, sᵢ = zeros

# Projᵢ is one of:
#   - Sᵢ ∈ ℝ^(B×B) Gaussian i.i.d. N(0,1)    (per-block Gaussian)
#   - SORF_qjlᵢ at B-dim                       (per-block SORF)
#   - SORF_qjl at padded_d-dim on full r        (full-dim padded SORF)

Store: qjl_signs, qjl_residual_norms, projection params (strategy-dependent)
```

#### Decoding algorithm

```
# MSE decode (produces unit-norm sub-vectors)
for i in 0..k:
    r̂ᵢ[j] = centroids[cᵢ[j]]
    ûᵢ = SORF⁻¹ᵢ(r̂ᵢ)

# QJL correction (if present)
for i in 0..k:
    if γᵢ > 0:
        correctionᵢ = (√(π/2) / dim_proj) × γᵢ × Projᵢᵀ × sᵢ
        ûᵢ += correctionᵢ
# dim_proj = B for per-block strategies, padded_d for full-dim strategy

# Denormalize and concatenate (using externalized norms)
for i in 0..k:
    x̂ᵢ = nᵢ × ûᵢ
x̃ = concat(x̂₀, x̂₁, ..., x̂ₖ₋₁)[0..d]
```

### Stage 2: PDX dimension-major layout

In a second stage, we transpose the code storage from row-major to
dimension-major within groups of 64 vectors, following the PDX layout [4]. The
64-vector group size is independent of the TQ block size B.

**Row-major (Stage 1):**

```
Vector 0: [tq0_d0 ... tq0_d(B-1) | tq1_d0 ... tq1_d(B-1) | ... ]
Vector 1: [tq0_d0 ... tq0_d(B-1) | tq1_d0 ... tq1_d(B-1) | ... ]
...
```

**PDX (Stage 2), within each 64-vector chunk:**

```
TQ block 0, dim 0:        [v0 v1 v2 ... v63]
TQ block 0, dim 1:        [v0 v1 v2 ... v63]
...
TQ block 0, dim (B - 1):  [v0 v1 v2 ... v63]
TQ block 1, dim 0:        [v0 v1 v2 ... v63]
...
```

All codes for the same dimension across 64 vectors are contiguous. The inner
loop (over 64 vectors) has no inter-vector data dependencies and
auto-vectorizes into SIMD. TQ block boundaries affect only where norm weighting
occurs in the distance kernel — they don't affect the PDX transpose itself.

This creates **B × 64 tiles**: B dimension coordinates × 64 vectors per chunk.

#### Relationship to chunking

The PDX 64-vector groups are effectively chunks. Shared metadata (centroids,
rotation signs/matrices) lives at the array level, not per-chunk.

**Open design questions:**

- Should slice/take produce row-major or PDX results? Row-major is simpler
  (un-transpose on the fly) but loses the scan benefit.
- Should the PDX flag be a property of the encoding or a separate layout layer?
- How does the PDX transpose interact with the cascading compressor?

#### Quantized distance computation

```rust
// Precompute: dist_table[i][j] = centroids[i] * centroids[j]
// At b=4: 16×16 = 256 floats = 1KB, fits in L1 cache

for tq_block in 0..num_tq_blocks {
    for dim in 0..B {
        let qd = query_codes[tq_block * B + dim];
        let row = &dist_table[qd as usize];
        for v in 0..64 {  // SIMD-friendly: no inter-vector deps
            distances[v] += row[codes[offset + v] as usize];
        }
    }
    // Weight by query_block_norm × data_block_norms[v] for this TQ block
}
```

**Full decompression from PDX layout.** When a scan is not the use case (e.g.,
`scalar_at` or `to_canonical`), 64-vector chunks are un-transposed back to
row-major before per-vector inverse SORF decoding.

## Array layout

```
TurboQuantArray (operates on unit-norm B-dim sub-vectors)
├── metadata: { dimension: u32, bit_width: u32, block_size: u32,
│               num_blocks: u32, has_qjl: bool, is_pdx: bool }
│
│  # Per-row children (sliced/taken on row operations)
├── codes: FixedSizeListArray<u8>              # len=num_rows, list_size=num_blocks×B
│
│  # Shared children (cloned on row operations, not sliced)
├── centroids: PrimitiveArray<f32>             # len=2^(bit_width-1) [MSE codebook]
├── mse_rotation_signs: PrimitiveArray<u8>     # len=num_blocks×3×B [per-block MSE SORF]
│
│  # Optional QJL children (per-block Gaussian)
├── [qjl_signs]: FixedSizeListArray<u8>        # len=num_rows, list_size=num_blocks×B
├── [qjl_matrices]: PrimitiveArray<f32>        # len=num_blocks×B×B [Gaussian i.i.d. N(0,1)]

Externalized (lives in parent encoding, not in TurboQuantArray):
├── block_norms: FixedSizeListArray<F>         # len=num_rows, list_size=num_blocks
└── [qjl_residual_norms]: FixedSizeListArray<F>    # len=num_rows, list_size=num_blocks
```

The `is_pdx` flag in metadata determines whether `codes` (and `qjl_signs`) are
in row-major or PDX-transposed layout.

## Compression ratio

For f32 input at dimension d with bit width b (QJL, so b-1 MSE bits + 1 QJL
bit), k = ⌈d/B⌉ blocks:

| Component          | Bits per vector |
| ------------------ | --------------- |
| MSE codes          | k × B × (b-1)   |
| QJL signs          | k × B × 1       |
| Block norms        | k × norm_bits   |
| QJL residual norms | k × norm_bits   |

| Component             | Shared bits  |
| --------------------- | ------------ |
| Centroids             | 2^(b-1) × 32 |
| MSE SORF signs        | k × 3 × B    |
| QJL Gaussian matrices | k × B² × 32  |

### Example: f32, d=768, b=5, B=128, N=1000 vectors, k=6

- Uncompressed: 768 × 32 × 1000 = 24,576,000 bits (3,000 KB)
- MSE codes: 6 × 128 × 4 × 1000 = 3,072,000 bits
- QJL signs: 6 × 128 × 1 × 1000 = 768,000 bits
- Block norms: 6 × 32 × 1000 = 192,000 bits
- QJL residual norms: 6 × 32 × 1000 = 192,000 bits
- Shared: 512 + 2,304 + 3,145,728 = 3,148,544 bits
- **Total compressed: 7,372,544 bits (899 KB)**
- **Ratio: 3.3×** (QJL Gaussian matrices dominate at small N)

At N=100K vectors the shared overhead amortizes to <0.04 bits/vector, giving
ratio ≈ 5.7×. At N=1M, ratio ≈ 5.8×.

### Comparison across configurations (f32, d=768, b=5)

| Config                                           | B   | Ratio (N=1K) | Ratio (N=100K) | Notes                         |
| ------------------------------------------------ | --- | ------------ | -------------- | ----------------------------- |
| Block MSE-only                                   | 128 | 6.5×         | 6.5×           | No QJL; biased inner products |
| Block + per-block Gaussian QJL                   | 128 | 3.3×         | 5.7×           | Unbiased; matrices amortize   |
| [Current][current-impl] (padded SORF + SORF QJL) | —   | 4.7×         | 4.7×           | 33% padding waste             |

For small columns, MSE-only or the current padded approach may be preferable.
For large columns (the common case for embedding tables), per-block Gaussian QJL
gives the best ratio.

## Performance analysis

### Encode throughput

With k blocks at B-dim, encoding requires per block:

| Operation                   | FLOPs (B=128)                         |
| --------------------------- | ------------------------------------- |
| MSE SORF (3-round)          | 3 × 128 × log₂(128) + 3 × 128 ≈ 3,072 |
| Centroid lookup             | 128 binary searches                   |
| QJL Gaussian matmul (S × r) | B² = 16,384                           |
| Norm computation            | 128 FMA + sqrt ≈ 129                  |

For d=768, k=6: MSE total ≈ 18K FLOPs, QJL matmul ≈ 98K FLOPs. The QJL
Gaussian matmul dominates encode cost — ~5× more expensive than the
[current][current-impl] SORF-based QJL. Acceptable for offline encoding.

### Decode throughput

| Operation                        | FLOPs per block (B=128) |
| -------------------------------- | ----------------------- |
| Codebook lookup                  | 128 table reads         |
| Inverse SORF                     | ≈ 3,072                 |
| QJL Gaussian matmul (Sᵀ × signs) | B² = 16,384             |
| Denormalize                      | 128 multiplies          |

For d=768: MSE decode ≈ 18K FLOPs, QJL decode ≈ 98K FLOPs. QJL decode is
significantly more expensive due to the dense matmul. For scan workloads that
only need inner products (not full reconstruction), the fused distance
computation path avoids full decode entirely.

### Scan throughput (PDX, Stage 2)

The PDX scan kernel is memory-bandwidth bound, not compute bound. The lookup
table fits in L1 (1 KB at b=4). With 64 vectors per SIMD pass, this achieves
near-peak memory bandwidth utilization. Norm weighting adds k scalar multiplies
per 64-vector group — negligible.

### Benchmarking plan

Stage 1:

1. Encode throughput: block TQ vs. current TQ at d=128, 768, 1024
2. Decode throughput: same dimensions
3. Quantized cosine similarity: block vs. current (includes norm overhead)
4. L2 norm readthrough: O(k) block norms vs. O(1) current

Stage 2:

5. PDX scan throughput vs. row-major scan at d=768, 1024
6. Full decompression from PDX layout (includes un-transpose overhead)

## Experimental plan

Before committing to specific parameters, the following experiments are needed:

### MSE quality vs. block size

- Compare actual normalized MSE at B ∈ {64, 128, 256} vs. full-dimension
  single-SORF at bit widths b ∈ {2, 3, 4, 5, 8}
- Test SORF coordinate distribution at each B: histogram vs. analytical Beta
- Test 3, 4, and 5 SORF rounds at each B (3 rounds provides 3 × log₂(B)
  mixing stages: 18 at B=64, 21 at 128, 24 at 256)

### QJL strategy comparison

Compare all four strategies at d=768 with B ∈ {64, 128, 256}:

- **Per-block Gaussian QJL** (paper-correct): measure inner product bias and
  variance. Baseline for correctness.
- **Per-block SORF QJL**: measure bias/variance vs. per-block Gaussian at same
  B. Quantify the quality cost of the SORF approximation at small block
  dimensions. Test at 3, 4, 5 SORF rounds.
- **Full-dimension padded SORF QJL** (current approach): measure for comparison.
  Higher dimension gives better SORF-to-Haar convergence (30 mixing stages at
  d=1024) which may compensate for the padding waste. This is the key
  comparison — does the better convergence of full-dim SORF outweigh the 25%
  wasted sign bits?
- **MSE-only** (no QJL): measure inner product bias to establish baseline.

Key metrics:

- Mean signed relative error on inner products (bias)
- Variance of inner product estimates
- ANN recall@k on standard benchmarks (e.g., SIFT, GloVe)

Per-block Gaussian QJL has d/B times more variance than full-dim QJL (6× at
d=768, B=128). Per-block SORF QJL has additional approximation error on top of
that. Full-dim padded SORF QJL has lower variance but wastes bits on padding.
The experiment should determine which effect dominates in practice.

### Straggler handling

- Compare padded straggler (zero-pad to B, use SORF) vs. dense rotation at
  actual straggler dimension with own centroids
- Measure whether dense straggler enables increasing B (e.g., B=256 with
  d=768 = 3×256, no straggler at all)

## Phasing

**Stage 1** (block decomposition): Modify the existing TurboQuant encoding to
use B-dim blocks with per-block SORF rotation, externalized norms, and
configurable QJL strategy (per-block Gaussian, per-block SORF, full-dim padded
SORF, or MSE-only). Row-major code layout.

**Stage 2** (PDX layout): Add dimension-major code transposition within
64-vector chunks. PDX-native distance computation kernels. Requires resolving
open design questions around chunk alignment and compressor interaction.

## Test migration

The [current TurboQuant test suite][current-impl] will need rewriting:

- **Slot counts**: New structure with per-block SORF signs, QJL Gaussian
  matrices, externalized norms.
- **Serde roundtrips**: New metadata (block_size, num_blocks).
- **MSE bound tests**: Add tests at B ∈ {64, 128, 256}.
- **QJL tests**: Compare all four strategies (per-block Gaussian, per-block
  SORF, full-dim padded SORF, MSE-only).
- **Quantized cosine similarity**: Update for per-block norm weighting.
- **Slice/take**: More children to manage, same semantics.

New tests:

- SORF quality at B=64, 128, 256: distribution vs. Beta at 3, 4, 5 rounds
- Practical MSE comparison: block vs. single-rotation at same bit width
- QJL comparison: per-block Gaussian vs. per-block SORF vs. full-dim padded SORF
- Zero-norm sub-vector handling
- Straggler handling (padded initially, dense rotation as follow-up)
- Encode/decode benchmarks

## Future work: GPU decode and fused distance computation

### Motivation

For ANN search workloads, the dominant operation is computing distances between
a query vector and millions of database vectors. On GPU, the goal is to perform
this computation directly on the compressed representation, avoiding the cost
of materializing full decompressed vectors in HBM.

### Decode as GEMM

The decode path for a single B-dim block is:

```
decoded_block = norm × R⁻¹ × codebook_lookup(codes)
```

For a batch of N vectors sharing the same block's rotation matrix R⁻¹:

```
decoded_batch = diag(norms) × R⁻¹ × codebook_lookup_batch(codes)
                                      ↑ B×N matrix
                               ↑ B×B × B×N = GEMM
```

The codebook lookup produces a B×N matrix, and the inverse rotation is a B×B
matrix multiply — a GEMM that maps directly to tensor cores.

**Partial decompression pipeline on GPU:**

1. **Decompress rotation matrix** (once per block, shared across all vectors):
   - If stored as bitpacked SORF signs: FastLanes unpack on CUDA cores, then
     expand to B×B matrix in shared memory
   - If stored as dense B×B matrix: direct load to shared memory

2. **Decompress norms** (per vector, per block):
   - If cascade-compressed with ALP or Pco: decompress on CUDA cores
   - If uncompressed: direct load

3. **Codebook gather + fused GEMM + scale**:
   - Stage codebook in shared memory (trivially small)
   - Stream code bytes from HBM, look up centroids, assemble B×N tile
   - R⁻¹ × tile on tensor cores, scale by norms
   - Follows the double-buffered streaming pattern from Flash-KMeans [6]

### Fused distance computation (no decode)

Stage the distance table in shared memory (1 KB at b=4), stream code bytes from
HBM, gather-reduce on-chip, write only final distances. The memory access
pattern follows Flash-KMeans [6]: double-buffered prefetch, on-chip
accumulation. We stream quantized code bytes — 4-8× less HBM bandwidth than
full float vectors.

### Int8 tensor core path (b=9)

At b=9, MSE codes are 8-bit indices. Direct int8 tensor core GEMM requires
approximately linear centroids (sacrificing Max-Lloyd optimality). Viable for
ANN ranking; prefer codebook gather for reconstruction.

### Interaction with Vortex file format

The B-dim block structure ensures rotation matrices fit in GPU shared memory
(B×B×4 bytes). Child arrays are individually compressed by the cascading
compressor; GPU decode requires either host-side decompression + GPU transfer,
or direct GPU decompression of FastLanes/ALP.

## References

[1] Zandieh, A., Daliri, M., Hadian, M. and Mirrokni, V. "TurboQuant: Online
Vector Quantization with Near-optimal Distortion Rate." arXiv:2504.19874,
April 2025.

[2] Ailon, N. and Chazelle, B. "The Fast Johnson-Lindenstrauss Transform and
Approximate Nearest Neighbors." SIAM Journal on Computing, 39(1):302-322, 2009.

[3] Tropp, J.A. "Improved Analysis of the Subsampled Randomized Hadamard
Transform." Advances in Adaptive Data Analysis, 3(1-2):115-126, 2011.

[4] Kuffo, L., Krippner, E. and Boncz, P. "PDX: A Data Layout for Vector
Similarity Search." Proceedings of SIGMOD '25. arXiv:2503.04422, March 2025.

[5] Yu, F.X., Suresh, A.T., Choromanski, K., Holtmann-Rice, D. and Kumar, S.
"Orthogonal Random Features." Advances in Neural Information Processing
Systems 29 (NeurIPS), 2016. arXiv:1610.09072.

[6] Yang, S. et al. "Flash-KMeans: Fast and Memory-Efficient Exact K-Means."
arXiv:2603.09229, March 2026.
