# Block-Decomposed TurboQuant with PDX Layout

**Authors:** Will Manning
**Status:** Proposal
**Date:** 2026-04-02

## Summary

We propose modifying the TurboQuant vector quantization encoding to use uniform
64-dimensional blocks with per-block rotations instead of a single full-dimension
SRHT. Both the MSE quantization and the QJL correction operate per-block,
completely eliminating zero-padding waste and the associated QJL variance
inflation for non-power-of-2 dimensions. The lowest-level TurboQuant array
operates only on unit-norm vectors, with norms externalized. In a second stage,
the block structure enables a PDX-style dimension-major code layout for
SIMD-friendly vectorized distance computation.

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

The paper prescribes a full random orthogonal rotation via QR decomposition of a
Gaussian matrix — O(d²) storage and O(d²) per-vector. Our current
implementation substitutes a Structured Random Hadamard Transform (SRHT) for
O(d) storage and O(d log d) per-vector, following the well-established use of
SRHT as a fast approximate random rotation [2, 3].

### Current limitations

The SRHT requires power-of-2 input dimension. For non-power-of-2 dimensions
(e.g., 768-d embeddings from common transformer models), the input is
zero-padded to the next power of 2 (1024). This causes:

- **33% wasted storage** for 768-d vectors: 1024 quantized codes stored per
  vector when only 768 carry information.
- **QJL variance inflation**: The QJL correction also uses a padded SRHT. Of
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

### Stage 1: 64-dim block decomposition

Partition d dimensions into `⌈d/64⌉` blocks of 64 dimensions each. The last
block ("straggler") may have fewer than 64 dimensions — see Straggler handling
below. Each full block gets an independent 64-dim SRHT rotation.

**Key properties:**

- **64 is a power of 2**, so every full block has zero padding waste. Common
  embedding dimensions (128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
  are all divisible by 64, so the straggler case is rare.
- **One shared centroid set** for full blocks. All full blocks use the same
  64-dim marginal distribution `(1 - x²)^(30.5)`, so a single Max-Lloyd
  codebook serves every full block.
- **Unit-norm assumption.** The TurboQuant array operates only on pre-normalized
  sub-vectors. Per-block norms are externalized (see Norm architecture below).
  This simplifies the encoding and enables efficient quantized-domain operations
  without norm lookups.
- **Per-block rotation signs.** Each block's SRHT is independent (different
  seed). Signs are 3 × 64 = 192 bits = 24 bytes per block. For 768-d: 12 × 24
  = 288 bytes total (less than the current 3 × 1024 = 384 bytes for a single
  padded SRHT).

#### Straggler handling

If `d % 64 ≠ 0`, the final block has `d mod 64` dimensions. The straggler block
uses a **dense random orthogonal matrix** (QR of Gaussian) instead of SRHT,
since SRHT requires power-of-2 dimension. At straggler dimensions ≤ 63, the
dense rotation is at most 63×63×4 = 15.9 KB — negligible. The straggler also
gets its own centroid set (computed for its actual dimension), since the
coordinate distribution differs from the 64-dim full blocks.

#### Zero-norm sub-vectors

When splitting a vector into 64-dim blocks, some blocks may have zero norm
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
upper bound that is dimension-independent. However, the **actual** MSE may
depend on dimension. At higher dimensions (d=768), the coordinate distribution
after rotation is more concentrated (variance ~1/768), and the Max-Lloyd
quantizer exploits this concentration — the actual MSE can be well below the
bound. At d=64, the distribution is wider (variance ~1/64), and the quantizer
has less concentration to exploit.

The block decomposition preserves the theoretical bound but may give worse
**practical** MSE than a single full-dimension rotation, because each 64-dim
block operates at a less favorable point on the distortion curve. This must be
quantified empirically: compare actual normalized MSE for d=64 blocks vs.
d=768 single-rotation at the same bit width. If the practical gap is
significant (e.g., >2x), a larger block size (128 or 256) would narrow it.

**SRHT approximation caveat.** Theorems 1 and 2 in [1] are proved for true
random orthogonal matrices (QR of Gaussian), not SRHT. Our use of SRHT is a
structured approximation that does not strictly satisfy the theorems'
assumptions. This is the same approximation the current single-SRHT
implementation makes — block decomposition does not introduce a new gap. However,
the approximation quality depends on dimension: at d=1024 the 3-round SRHT
closely approximates a random rotation, while at d=64 the approximation is
weaker (fewer mixing stages). This means the Max-Lloyd centroids (computed for
the analytical Beta distribution) may be slightly suboptimal for the actual SRHT
coordinate distribution at d=64, and the QJL scale factor `√(π/2)/B` may not
exactly cancel the SRHT's expected absolute coordinate value.

Empirical validation is needed to confirm that: (a) the MSE bound holds in
practice at d=64 with SRHT, (b) the QJL bias at d=64 is acceptable, and (c)
3 rounds of D·H is sufficient mixing at d=64. The number of SRHT rounds directly
affects mixing quality: at d=1024, 3 rounds provides 3 × log₂(1024) = 30
mixing stages, while at d=64 it provides only 3 × log₂(64) = 18. The coordinate
distribution should be tested at 3, 4, and 5 rounds to determine the minimum
sufficient for near-uniform mixing at d=64.

If the 3-round SRHT proves insufficient at d=64, fallback options include:

- **More SRHT rounds.** Increasing from 3 to 4 or 5 rounds of D·H improves
  mixing quality at modest O(d log d) cost.
- **Larger block size.** Using B=128 instead of B=64 — still a power of 2, still
  divides all common embedding dimensions, and the SRHT approximation is better
  at d=128 than d=64. PDX Stage 2 would need to handle the dimension-block vs.
  vector-block size mismatch (128-dim blocks with 64-vector groups).
- **Dense rotation at d=64.** Use a 64×64 random orthogonal matrix (QR of
  Gaussian) instead of SRHT for all blocks. At 64×64×4 = 16 KB this is
  negligible storage. This gives exact theoretical guarantees at the cost of
  O(d²) per-block rotation instead of O(d log d). For d=64 the difference is
  small: 4096 FLOPs (matmul) vs. ~384 FLOPs (SRHT). Note: if using the dense
  fallback, each block should still have an **independent** rotation matrix
  (generated from different seeds). Sharing a single rotation matrix across
  blocks introduces correlation between quantization errors, which doesn't
  affect the MSE bound but weakens the QJL independence assumption — the QJL
  rotation must be independent of the MSE rotation for Theorem 2's unbiasedness
  guarantee.

#### Per-block QJL correction

The QJL correction also operates per-block, using an independent 64-dim
rotation per block (distinct from the MSE rotation). For each block, the QJL
computes `sign(Sₖ × rₖ)` where `rₖ = xₖ - x̂ₖ` is the per-block residual and
`Sₖ` is the block's QJL rotation. The overall inner product estimator is:

```
<y, correction> = Σ_k <yₖ, correctionₖ>
```

Each block's correction is an independent unbiased estimate of `<yₖ, rₖ>`, so
the sum is unbiased: `E[<y, correction>] = <y, r>`. Theorem 2 of [1] applies
per-block because the QJL rotation `Sₖ` is independent of the MSE quantization
noise in block k.

Compared to a single full-dimension QJL with padding (768→1024), per-block QJL:

- **Wastes zero bits.** All 768 sign bits encode actual residual information,
  vs. 768 useful out of 1024 for the padded approach (25% waste eliminated).
- **Eliminates variance inflation.** No measurement budget is diluted by
  zero-padded coordinates.
- **Reuses the same rotation infrastructure** as the MSE stage (64-dim blocks),
  simplifying the implementation.
- **Trades cross-block mixing for efficiency.** A full-dimension rotation would
  mix residual information across blocks, potentially reducing variance. But
  this benefit is outweighed by the concrete information loss from padding waste
  in non-power-of-2 dimensions.

#### Norm architecture

The TurboQuant array itself operates only on unit-norm 64-dim sub-vectors. Norms
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

**Benefits of externalizing norms:**

- **L2 norm**: Read the block norms array directly, compute `√(Σ_k ‖xₖ‖²)`.
  This is O(k) per vector — modest cost compared to full decompression.
- **Cosine similarity**: Requires per-block norm weighting (see below), but the
  norms tensor can be read once and reused for all pairs.
- **Dot product**: `<a, b> = Σ_k ‖aₖ‖ · ‖bₖ‖ · <âₖ, b̂ₖ>` where `<âₖ, b̂ₖ>`
  is the quantized unit-norm dot product for block k. The norms tensor provides
  the weights.

#### Quantized-domain operations with per-block norms

With block decomposition and externalized norms, quantized cosine similarity
requires more work than the current single-block approach.

**Quantized dot product:**

```
<a, b> ≈ Σ_k ‖aₖ‖ · ‖bₖ‖ · Σ_j centroids[code_aₖ[j]] · centroids[code_bₖ[j]]
```

Per-block: compute the unit-norm quantized dot product (sum over 64 centroid
products), then weight by both vectors' block norms.

**Quantized cosine similarity:**

```
cos(a, b) ≈ <a, b> / (‖a‖ · ‖b‖)
           = Σ_k ‖aₖ‖ · ‖bₖ‖ · unit_dotₖ
             / (√(Σ_k ‖aₖ‖²) · √(Σ_k ‖bₖ‖²))
```

This requires reading all block norms for both vectors, computing per-block
weighted dot products, and normalizing by global norms. The norms tensor should
be read once per scan query and cached.

For the PDX scan kernel (Stage 2), the per-block norm weighting integrates
naturally: after accumulating the unit-norm dot product for each block of 64
dimensions, multiply by the query and data block norms before moving to the
next block.

#### Encoding algorithm

```
Input: x ∈ ℝ^d, bit_width b, block_size B=64
k = ⌈d/B⌉

# Block split and normalize
for i in 0..k:
    xᵢ = x[i*B .. min((i+1)*B, d)]
    nᵢ = ‖xᵢ‖
    if nᵢ > 0:
        ûᵢ = xᵢ / nᵢ                   (unit normalize, pad to B if straggler)
    else:
        ûᵢ = zeros(B)                   (zero-norm block: skip quantization)

# MSE stage (per block)
for i in 0..k:
    if nᵢ > 0:
        rᵢ = Rotateᵢ(ûᵢ)               (SRHT for full blocks, dense for straggler)
        cᵢ[j] = nearest_centroid(rᵢ[j]) (shared codebook for full blocks, own for straggler)
    else:
        cᵢ[j] = 0                       (arbitrary, never used in decode)

Store: codes (k × B per vector), block_norms (k per vector),
       centroids (shared), rotation params (per block, shared across vectors)

# QJL stage (per block, optional)
for i in 0..k:
    if nᵢ > 0:
        x̂ᵢ = decode_mse_block(cᵢ, centroids, Rotateᵢ)  (unit-norm reconstruction)
        rᵢ = ûᵢ - x̂ᵢ                   (per-block unit-norm residual)
        γᵢ = ‖rᵢ‖                       (per-block residual norm)
        sᵢ = sign(Rotate_qjlᵢ(rᵢ))     (independent rotation per block)
    else:
        γᵢ = 0, sᵢ = zeros

Store: qjl_signs (k × B per vector), qjl_residual_norms (k per vector),
       qjl_rotation params (per block, shared across vectors)
```

#### Decoding algorithm

```
# MSE decode (produces unit-norm sub-vectors)
for i in 0..k:
    r̂ᵢ[j] = centroids[cᵢ[j]]          (codebook lookup)
    ûᵢ = Rotate⁻¹ᵢ(r̂ᵢ)                (inverse rotation)

# QJL correction (if present)
for i in 0..k:
    if γᵢ > 0:
        correctionᵢ = (√(π/2) / Bᵢ) × γᵢ × Rotate⁻¹_qjlᵢ(sᵢ)
        ûᵢ += correctionᵢ

# Denormalize and concatenate (using externalized norms)
for i in 0..k:
    x̂ᵢ = nᵢ × ûᵢ
x̃ = concat(x̂₀, x̂₁, ..., x̂ₖ₋₁)[0..d]   (truncate straggler padding)
```

### Stage 2: PDX dimension-major layout

In a second stage, we transpose the code storage from row-major to
dimension-major within groups of 64 vectors, following the PDX layout [4].

**Row-major (Stage 1):**

```
Vector 0: [b0_d0 b0_d1 ... b0_d63 | b1_d0 ... b1_d63 | ... ]
Vector 1: [b0_d0 b0_d1 ... b0_d63 | b1_d0 ... b1_d63 | ... ]
...
```

All codes for a single vector are contiguous. Good for per-vector decode, bad
for vectorized scans.

**PDX (Stage 2), within each 64-vector chunk:**

```
Block 0, dim 0: [v0 v1 v2 ... v63]
Block 0, dim 1: [v0 v1 v2 ... v63]
...
Block 0, dim 63: [v0 v1 v2 ... v63]
Block 1, dim 0: [v0 v1 v2 ... v63]
...
```

All codes for the same dimension across 64 vectors are contiguous. The inner
loop (over vectors) has no inter-vector data dependencies and auto-vectorizes
into SIMD.

This creates a natural **64×64 tile** structure: 64 dimension-block coordinates
× 64 vectors per chunk. Each tile's codes are contiguous in memory.

#### Relationship to chunking

The PDX 64-vector groups are effectively chunks. The array is logically a
sequence of chunks, where each chunk contains 64 vectors (the last chunk may
be smaller). Within each chunk, codes and QJL signs are stored in
dimension-major order.

Shared metadata (centroids, rotation signs/matrices) lives at the array level,
not per-chunk. Each chunk is a view over the same codebook and rotations.

**Open design question:** The cleanest model may be to make each 64-vector
chunk a self-contained TurboQuant array of 64-dim unit-norm vectors
(dimension-major internally), with the chunk-level array managing the sequence
of chunks and the shared metadata. This interacts with Vortex's existing
chunked/sequence array patterns and needs further design work. Key questions:

- Should slice/take produce row-major or PDX results? Row-major is simpler
  (un-transpose on the fly) but loses the scan benefit. PDX is preservable
  only for aligned 64-vector slices.
- Should the PDX flag be a property of the encoding or a separate layout layer?
- How does the PDX transpose interact with the cascading compressor? The
  compressor sees the codes child array — should it see the transposed or
  un-transposed version?

#### Quantized distance computation

With PDX layout and a precomputed distance lookup table, the inner loop for
quantized dot product becomes:

```rust
// Precompute: dist_table[i][j] = centroids[i] * centroids[j]
// At b=4: 16×16 = 256 floats = 1KB, fits in L1 cache

for block in 0..num_blocks {
    let qa = query_codes[block * 64 + dim];
    for dim in 0..64 {
        let row = &dist_table[qa as usize];
        for v in 0..64 {  // SIMD-friendly: no inter-vector dependencies
            distances[v] += row[codes[offset + v] as usize];
        }
    }
    // Weight by query_block_norm × data_block_norms[v] for each v
}
```

This is a table-lookup-and-accumulate kernel — a single dependent load plus an
FP add per element, with the table row fitting in L1. PDX [4] reports an average
2x speedup from this layout on unquantized float vectors; with quantized codes
the lookup table further reduces arithmetic intensity.

**Full decompression from PDX layout.** When a scan is not the use case (e.g.,
`scalar_at` or `to_canonical`), the 64-vector chunks are un-transposed back to
row-major before per-vector inverse rotation decoding.

## Array layout

```
TurboQuantArray (operates on unit-norm 64-dim sub-vectors)
├── metadata: { dimension: u32, bit_width: u32, num_blocks: u32,
│               has_qjl: bool, is_pdx: bool, has_straggler: bool }
│
│  # Per-row children (sliced/taken on row operations)
├── codes: FixedSizeListArray<u8>              # len=num_rows, list_size=num_blocks×64
│
│  # Shared children (cloned on row operations, not sliced)
├── centroids: PrimitiveArray<f32>             # len=2^(bit_width-1) [MSE codebook]
├── mse_rotation_signs: PrimitiveArray<u8>     # len=(num_blocks-S)×3×64 [per-block MSE SRHT]
├── [straggler_mse_rotation]: PrimitiveArray<f32>  # straggler_dim × straggler_dim dense matrix
├── [straggler_centroids]: PrimitiveArray<f32> # centroids for straggler dimension
│
│  # Optional QJL children (also per-block)
├── [qjl_signs]: FixedSizeListArray<u8>        # len=num_rows, list_size=num_blocks×64
├── [qjl_rotation_signs]: PrimitiveArray<u8>   # len=(num_blocks-S)×3×64 [per-block QJL SRHT]
├── [straggler_qjl_rotation]: PrimitiveArray<f32>  # straggler dense QJL matrix
│
│  S = 1 if has_straggler else 0

Externalized (lives in parent encoding, not in TurboQuantArray):
├── block_norms: FixedSizeListArray<F>         # len=num_rows, list_size=num_blocks
└── [qjl_residual_norms]: FixedSizeListArray<F>    # len=num_rows, list_size=num_blocks
```

The `is_pdx` flag in metadata determines whether `codes` (and `qjl_signs`) are
in row-major or PDX-transposed layout.

## Compression ratio

For f32 input at dimension d with bit width b (QJL, so b-1 MSE bits + 1 QJL
bit), k = ⌈d/64⌉ blocks, B = 64:

| Component          | Bits per vector |
| ------------------ | --------------- |
| MSE codes          | k × B × (b-1)   |
| QJL signs          | k × B × 1       |
| Block norms        | k × norm_bits   |
| QJL residual norms | k × norm_bits   |

| Component          | Shared bits  |
| ------------------ | ------------ |
| Centroids          | 2^(b-1) × 32 |
| MSE rotation signs | k × 3 × B    |
| QJL rotation signs | k × 3 × B    |

### Example: f32, d=768, b=5 (4-bit MSE + 1-bit QJL), 1000 vectors, k=12

- Uncompressed: 768 × 32 × 1000 = 24,576,000 bits (3,000 KB)
- MSE codes: 12 × 64 × 4 × 1000 = 3,072,000 bits
- QJL signs: 12 × 64 × 1 × 1000 = 768,000 bits
- Block norms: 12 × 32 × 1000 = 384,000 bits
- QJL residual norms: 12 × 32 × 1000 = 384,000 bits
- Overhead: 16 × 32 + 2 × 12 × 192 = 5,120 bits
- **Total compressed: 4,613,120 bits (563 KB)**
- **Ratio: 5.3×**

### Example: f32, d=1024, b=5, 1000 vectors, k=16

- Uncompressed: 1024 × 32 × 1000 = 32,768,000 bits (4,000 KB)
- MSE codes: 16 × 64 × 4 × 1000 = 4,096,000 bits
- QJL signs: 16 × 64 × 1 × 1000 = 1,024,000 bits
- Block norms: 16 × 32 × 1000 = 512,000 bits
- QJL residual norms: 16 × 32 × 1000 = 512,000 bits
- Overhead: 16 × 32 + 2 × 16 × 192 = 6,656 bits
- **Total compressed: 6,150,656 bits (750 KB)**
- **Ratio: 5.3×**

### Comparison: current single-SRHT TurboQuant

| Dimension         | Current ratio | Block ratio | Delta       |
| ----------------- | ------------- | ----------- | ----------- |
| 768 (padded→1024) | 4.7×          | 5.3×        | +13% better |
| 1024 (no padding) | 6.3×          | 5.3×        | -16% worse  |

For power-of-2 dimensions, the block approach uses more storage due to per-block
norms and QJL residual norms (k × 2 × 32 bits/vector overhead). At d=1024 with
k=16 blocks, this is 1024 bits/vector = 128 bytes, which is significant.

This is the fundamental tradeoff: block decomposition optimizes for
non-power-of-2 dimensions (which dominate real embedding models) at the cost of
modest overhead for power-of-2 dimensions.

## Performance analysis

### Encode throughput

With k blocks at d-dimensional input, encoding requires:

| Operation              | Per-block          | Total (k blocks) |
| ---------------------- | ------------------ | ---------------- |
| SRHT (3-round, B=64)   | ~384 FLOPs         | k × 384          |
| Centroid lookup (B=64) | 64 binary searches | k × 64           |
| QJL SRHT               | ~384 FLOPs         | k × 384          |
| Norm computation       | 64 FMA + sqrt      | k × 65           |

For d=768, k=12: total ~20K FLOPs per vector (SRHT + QJL SRHT) vs. current
single-SRHT approach: 2 × ~10K = ~20K FLOPs. Encode throughput should be
comparable — the per-block overhead is offset by smaller per-block transforms.

### Decode throughput

Decoding is dominated by k inverse rotations + codebook lookups:

| Operation        | Per-block      | Total (k blocks) |
| ---------------- | -------------- | ---------------- |
| Codebook lookup  | 64 table reads | k × 64           |
| Inverse SRHT     | ~384 FLOPs     | k × 384          |
| QJL inverse SRHT | ~384 FLOPs     | k × 384          |
| Denormalize      | 64 multiplies  | k × 64           |

For d=768: ~10K FLOPs decode vs. current ~10K (single 1024-dim inverse SRHT).
Similar throughput.

### Scan throughput (PDX, Stage 2)

The PDX scan kernel is memory-bandwidth bound, not compute bound. The lookup
table fits in L1 (1 KB at b=4). The inner loop is a single dependent load +
FP add per code byte. With 64 vectors per SIMD pass, this achieves near-peak
memory bandwidth utilization.

### Benchmarking plan

Stage 1 should include benchmarks comparing:

1. Encode throughput: block TQ vs. current TQ at d=128, 768, 1024
2. Decode throughput: same dimensions
3. Quantized cosine similarity throughput: block vs. current
4. L2 norm readthrough latency: O(k) block norms vs. O(1) current

Stage 2 should benchmark: 5. PDX scan throughput vs. row-major scan at d=768, 1024 6. Full decompression from PDX layout (includes un-transpose overhead)

## Phasing

**Stage 1** (block decomposition): Modify the existing TurboQuant encoding to
use 64-dim blocks with per-block rotation, externalized norms, and per-block
QJL. Row-major code layout. Full encode/decode, slice/take, serde, scheme
integration. This is a self-contained improvement that eliminates padding waste
and QJL variance inflation for all dimensions.

**Stage 2** (PDX layout): Add the dimension-major code transposition within
64-vector chunks. PDX-native distance computation kernels. This requires
resolving the open design questions around chunk alignment, slice/take semantics,
and interaction with the compressor pipeline.

## Test migration

The current TurboQuant test suite validates specific behaviors that will change:

- **Slot counts** (MSE=4, QJL=7): Will change to reflect new slot structure
  with per-block rotation signs, straggler fields, and externalized norms.
- **Serde roundtrips**: Must be rewritten for new metadata (num_blocks,
  has_straggler) and new child structure.
- **MSE bound tests**: Can be adapted — same bound, different dimensions tested.
  Must add d=64 specifically.
- **QJL bias tests**: Must add d=64 bias tests. Existing d=768 tests should
  show improved bias (no padding).
- **Quantized cosine similarity**: Must be updated for per-block norm weighting.
- **Slice/take**: Same semantics (per-row data sliced, shared data cloned), but
  more children to manage.

New tests needed:

- SRHT quality at d=64: coordinate distribution vs. analytical Beta at 3, 4, 5 rounds
- Practical MSE comparison: d=64 blocks vs. d=768 single-rotation at same bit width
- Straggler block handling: dense rotation, separate centroids
- Zero-norm sub-vector handling
- Per-block norm consistency with input vectors
- Encode/decode benchmarks (see Performance analysis)

## Future work: GPU decode and fused distance computation

### Motivation

For ANN search workloads, the dominant operation is computing distances between
a query vector and millions of database vectors. On GPU, the goal is to perform
this computation directly on the compressed representation, avoiding the cost
of materializing full decompressed vectors in HBM. BlockTurboQuant's 64-dim
block structure maps naturally to GPU tile sizes and tensor core operations.

### Decode as GEMM

The decode path for a single 64-dim block is:

```
decoded_block = norm × R⁻¹ × codebook_lookup(codes)
```

For a batch of N vectors sharing the same block's rotation matrix R⁻¹:

```
decoded_batch = diag(norms) × R⁻¹ × codebook_lookup_batch(codes)
                                      ↑ 64×N matrix
                               ↑ 64×64 × 64×N = GEMM
```

The codebook lookup produces a 64×N matrix (one column per vector, each entry
is `centroids[code]`), and the inverse rotation is a 64×64 matrix multiply —
a GEMM that maps directly to tensor cores.

**Partial decompression pipeline on GPU:**

1. **Decompress rotation matrix** (once per block, shared across all vectors):
   - If stored as bitpacked SRHT signs: FastLanes SIMD unpack on CUDA cores,
     then expand to 64×64 matrix in shared memory
   - If stored as dense 64×64 matrix: direct load to shared memory (16 KB)

2. **Decompress norms** (per vector, per block):
   - If cascade-compressed with ALP or Pco: decompress via FastLanes on CUDA
     cores into a register tile
   - If uncompressed: direct load

3. **Codebook gather** (per vector, per block):
   - Stage the codebook in shared memory (16 entries × 4 bytes = 64 bytes at
     b=4 — trivially small)
   - Gather: stream code bytes from HBM, look up centroid values in shared
     memory, assemble 64×N tile

4. **Fused GEMM + scale**:
   - R⁻¹ × gathered tile (64×64 × 64×N) on tensor cores
   - Column-wise multiply by norms (element-wise scale)

Steps 3-4 can be fused into a single kernel, following the double-buffered
streaming pattern from Flash-KMeans [5]: prefetch the next batch of code bytes
from HBM while computing the current batch's GEMM on tensor cores. This avoids
materializing the full decompressed vectors in HBM — the decoded output is
either consumed immediately by a distance computation or written once to the
output buffer.

### Fused distance computation (no decode)

For distance computation without full decompression, the operation per block is:

```
dot_contribution_k = ‖query_k‖ × ‖data_k‖ × Σ_j dist_table[q_code[j]][d_code[j]]
```

On GPU, this becomes:

1. **Stage distance table in shared memory**: `dist_table[i][j] = centroids[i] ×
centroids[j]`, 16×16 = 1 KB at b=4. Fits trivially in shared memory.

2. **Stream code bytes from HBM**: For each 64-vector × 64-dim tile (matching
   the PDX layout), gather from the distance table and accumulate in registers.
   This is a gather-reduce pattern — no GEMM, just table lookups and FP adds.

3. **Norm weighting**: After accumulating the unit-norm dot product for all
   64 dimensions in a block, multiply by the query and data block norms.
   Norms for 64 vectors fit in a single register tile.

4. **Cross-block accumulation**: Sum the weighted dot products across all k
   blocks to get the final distance estimate.

The memory access pattern follows Flash-KMeans [5]: stream data tiles from HBM
with double-buffered prefetch, accumulate on-chip, write only the final result.
The key difference is that Flash-KMeans streams full float vectors while we
stream quantized code bytes — 4-8× less HBM bandwidth per vector.

### Int8 tensor core path (b=9)

At b=9, the MSE component uses 8-bit codes. These are indices into a 256-entry
codebook, not raw int8 values — so direct int8 tensor core GEMM does not apply
without transformation. However, if the codebook is approximately linear
(centroids roughly evenly spaced), the codes could be treated as approximate
int8 values with a linear rescaling, enabling direct int8 GEMM for the inner
product computation. This sacrifices some quantization optimality (linear
quantization vs. Max-Lloyd optimal) but enables tensor core throughput.

Whether this tradeoff is worthwhile depends on the application: for ANN ranking
(where relative ordering matters more than absolute accuracy), linear int8 may
be sufficient. For reconstruction (where MSE matters), Max-Lloyd centroids are
preferred and the gather-from-codebook path should be used.

### Interaction with Vortex file format

The GPU decode pipeline reads compressed data from Vortex files:

1. **File reader** loads compressed segments from storage (S3, local SSD)
2. **Host-side cascade decompression** (BitPacked → codes, ALP → norms) or
   direct GPU transfer of already-decompressed segments
3. **GPU kernel** performs fused decode or fused distance computation

The BlockTurboQuant encoding's child arrays (codes, norms, rotation signs) are
individually compressed by the cascading compressor. For GPU decode, we need
either:

- Host-side decompression of the cascade, then GPU transfer of the raw children
- Direct GPU decompression of FastLanes/ALP (if GPU decompression kernels exist)

The 64-dim block structure ensures that rotation matrices (64×64 dense or 192
bits SRHT signs) fit comfortably in GPU shared memory, enabling the fused
decode kernel without spilling to HBM.

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

[5] Yang, S. et al. "Flash-KMeans: Fast and Memory-Efficient Exact K-Means."
arXiv:2603.09229, March 2026.
