# OnPair: A GPU-Friendly String Encoding for Vortex

**Authors:** Will Manning
**Status:** Proposal
**Date:** 2026-05-12
**RFC PR:** [vortex-data/rfcs#59](https://github.com/vortex-data/rfcs/pull/59)

## Summary

This RFC proposes adopting **OnPair** (Gargiulo, Venturini, 2025) as a Vortex string encoding, with two extensions on top of the published design: (1) the code width is encoder-parametric (10/11/12/14/16 bits), with **12-bit as the default** so the flattened symbol table fits in NVIDIA Hopper-class shared memory; (2) a first-class CUDA decoder that combines a SMEM-resident dictionary, writer-side output-offset checkpoints (the design idea from GSST that decouples segment decoding from serial prior-state dependencies — commonly summarized as "block parallelism" but really *split parallelism via checkpoints*), and warp-level prefix-sum on per-code output lengths.

The encoding integrates with Vortex in two tiers — `OnPairArray` for self-contained per-array dictionaries, and `OnPairLayout` for cross-chunk dictionary sharing via Vortex's existing `LayoutRef`-Arc mechanism — exactly mirroring how the current FSST encoding and `DictLayout` cooperate today.

A note on naming. "OnPair" here means the published algorithm (single-pass pair-merge LPM training; fixed-width codes; merge-pair-list dictionary; no escape mechanism). At 16-bit codes this RFC's encoding is functionally identical to upstream OnPair. The 12/14-bit configurations narrow the code width; the 10/11-bit configurations narrow it further; the algorithm is unchanged at every width. Where width matters, the doc qualifies as `OnPair-12`, `OnPair-16`, etc.

## Motivation

Vortex currently uses FSST for variable-length string compression. Three converging signals from the recent literature suggest a meaningfully better point in the design space is reachable today:

1. **FSST8's escape mechanism is a real cost on diverse data.** FSST8 has 255 symbols + 1 escape code; bytes that don't fit any symbol cost two input bytes (escape + raw). For text-heavy columns this is fine, but for more diverse distributions (JSON, XML, identifiers in code, mixed-language text) the escape rate climbs and the branchy fast-vs-slow-path decode loop costs measurable throughput. The `cwida/fsst` reference implementation ships an FSST12 variant ([`fsst12.h`, `libfsst12.cpp`](https://github.com/cwida/fsst)) whose source comment states it "will outperform [FSST8] on datasets that are more chaotic, such as JSON and widely diverse URLs." With 4096 codes available, every byte 0–255 can be a symbol via training, which substantially reduces the escape rate — though FSST12 retains escape logic in the decoder for unseen bytes.

2. **GPU string decompression is now a real target.** The GSST work (Vonk, Hoozemans, Al-Ars, 2025; published in ACM SIGOPS Operating Systems Review, Vol. 59 No. 1) demonstrates that FSST-class encodings can be decoded on GPU at near-bandwidth rates — the paper reports **191 GB/s on A100** — given a shared-memory-resident symbol table and a kernel that handles variable-length output via prefix-sum + coalesced writes. Tim Anema's complementary ADMS 2025 work (["High Throughput GPU-Accelerated FSST String Compression"](https://github.com/timanema/fsst-gpu)) reports **74 GB/s on RTX 4090** for the encode side, using a thread-voting matching mechanism and a stream-compaction output pipeline. GPU decompression is no longer hypothetical for production formats; Vortex should target it directly.

3. **Fixed-width codes + larger dictionaries dominate variable-width-with-escapes.** OnPair ([Gargiulo, Venturini, 2025; arXiv:2508.02280](https://arxiv.org/abs/2508.02280)) introduces a single-pass pair-merge training algorithm — much cheaper than classical BPE — that produces up-to-65k-entry dictionaries with **no escape codes** (the first 256 codes are reserved as raw bytes; every byte has a representation). On the Book Titles dataset at 16-bit codes, Table 1 of the paper reports up to **~5.4 GB/s** CPU decode (on an Intel Core Ultra 7 265K performance core; the reference C++ implementation relies on compiler auto-vectorization of `memcpy`-with-overcopy rather than explicit SIMD intrinsics). The fixed-width codes are also intrinsically GPU-friendly: no warp divergence from variable-stride code consumption.

The field is converging on a clear direction: **fixed-width codes, no escapes, larger dictionaries trained adaptively from the data**. FSST12 covers the bitstream side. OnPair covers the training side. GSST covers the GPU decoder side. None of the three alone occupies the design point Vortex actually wants — a single encoding that is competitive on both CPU and GPU with first-class shared-dictionary support — but their union does. This RFC proposes that synthesis.

### Priorities

In strict priority order:

1. **Decompression throughput on CPU and GPU.** Primary metric. We are willing to sacrifice modest compression ratio for substantial decode-throughput improvements on both surfaces.
2. **Compression ratio.** Secondary; should match or exceed FSST on representative workloads.
3. **Compression throughput.** Tertiary; OnPair-class single-pass training (~100–230 MiB/s on a single core) is sufficient.

## Design

### Lineage

The encoding proposed here is a deliberate combination of three published encodings:

- **OnPair** ([Gargiulo, Venturini, 2025](https://arxiv.org/abs/2508.02280)) provides the backbone: the single-pass pair-merge training algorithm, the principle that symbols can be unbounded byte sequences built by chaining merges, and the escape-free fixed-width-code bitstream shape (codes 0–255 reserved as raw bytes; codes 256+ are learned merge-pair symbols). The reference implementation at [`gargiulofrancesco/onpair_cpp`](https://github.com/gargiulofrancesco/onpair_cpp) supports configurable code widths from 9 to 16 bits. This RFC's encoding *is* OnPair: same algorithm, same bitstream shape, with a 12-bit default and a GPU decoder layered on top.
- **FSST12** ([Boncz et al, PVLDB Vol 13, 2020](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf); 12-bit variant from the reference implementation at [`cwida/fsst`](https://github.com/cwida/fsst)) provides the well-tested 12-bit-codes-packed-two-per-three-bytes bitstream layout that we reuse at the 12-bit width. (FSST12 itself differs from this RFC by retaining FSST's training algorithm and escape mechanism; we keep only its bit-packing layout.)
- **GSST** (Vonk, Hoozemans, Al-Ars, 2025; ACM SIGOPS Operating Systems Review Vol. 59 No. 1) provides the GPU decoder design: cooperative shared-memory dictionary load, writer-side per-segment output-offset metadata so segments can be decoded independently in parallel (the actual key insight behind "parallel string decompression at 191 GB/s on A100"), warp-level prefix-sum on per-code output lengths, and coalesced bandwidth-optimal stores. We refer to the underlying technique throughout this RFC as "split parallelism via output-offset checkpoints"; the exact terminology in the paper may differ.

What's new in this RFC relative to those three is the *combination*: filling FSST12's bitstream with OnPair's training, defaulting to 12-bit code width so the flattened symbol table fits in NVIDIA Hopper-class shared memory, providing two CUDA decoder modes (dict-in-shared-memory and dict-in-global-memory) for different code widths, and integrating with Vortex's existing `DictLayout` sharing model.

### Two-tier Vortex integration

Vortex already has the right primitive for cross-chunk dictionary sharing: `DictLayout` has two children (`values`, `codes`) and sharing is expressed *structurally* by Arc-sharing the `values` `LayoutRef` across multiple `DictLayout`s wrapped in a `ChunkedLayout` — no IDs, no hashes, no side tables. The reader lazy-materializes the values array via `OnceLock` and caches it across calls; predicate pushdown evaluates expressions on the (small) dict array and applies via `take`. The OnPair encoding mirrors this exactly.

The proposed encoding has two surfaces:

**Tier 1 — `OnPairArray` (analog of `FSSTArray`).** Self-contained per-array encoding with the merge-pair dictionary embedded as buffers. The default mode, suitable for any workload that doesn't benefit from a corpus-wide dictionary. Looks like:

```
buffers:
  [0] merge_pairs        : (parent0, parent1)[n_pairs]   // dict; u16 × 2 per entry
  [1] code_offsets       : bitpacked u32[n_strings + 1]  // per-string boundaries
  [2] codes              : packed[n_codes]               // the compressed stream
  [3] gpu_checkpoints    : (input_offset, output_offset)[n_chk]  // OPTIONAL; for GPU split parallelism
metadata (Prost):
  code_width_bits:       u8       // 10/11/12/14/16
  n_pairs:               u32      // ≤ 2^code_width_bits - 256
  n_strings:             u32
  n_chk:                 u32      // 0 if gpu_checkpoints buffer is absent
  uncompressed_bytes:    u64
  code_offsets_ptype:    PType    // u16/u32 depending on n_codes
  max_symbol_length:     u8       // 8 or 16
```

The first 256 codes are reserved as raw bytes; codes `256..(256 + n_pairs)` are merge-pair symbols defined recursively by `(parent0, parent1)` indices into the same code space. The flattened symbol table is reconstructed at decode start, not stored on disk (a merge-pair list at ~4 bytes/entry is roughly 2× more compact than the OnPair reference implementation's flat-bytes-plus-cumulative-offsets representation at typical symbol lengths; the two formats are functionally equivalent and decode to the same byte sequences). The optional `gpu_checkpoints` buffer is described in the GPU decoder section and may be absent on arrays not destined for GPU decode.

**Tier 2 — `OnPairLayout` (analog of `DictLayout`).** A layout with two children:

```rust
pub struct OnPairLayout {
    /// The merge-pair dictionary; typically Arc-shared across many sibling
    /// OnPairLayouts in a ChunkedLayout, exactly like DictLayout's `values`.
    dict: LayoutRef,
    /// Codes + per-string offsets for this chunk.
    data: LayoutRef,
}
```

Sharing across chunks is identical in shape to `DictLayout`: the writer trains one dict on the corpus, emits one `dict` `LayoutRef`, and every chunk's `OnPairLayout` Arc-shares that same reference. The reader uses an `OnceLock`-cached flattened symbol table — for a maximally-sized 64k-entry dict this is ~1 MiB, easily resident in L2/L3 between calls, amortized over the millions of strings the dict applies to.

This Tier-2 mode is what makes corpus-wide dictionary training viable without inventing any new file-format machinery. It is also what allows the encoder to ingest an externally-trained dictionary (whatever the source) and apply it uniformly across all chunks.

### `code_width_bits`: the load-bearing knob

The natural OnPair code width is 16 bits (65,536 entries). At that capacity the flattened decode-time symbol table is ~1 MiB (16-byte symbol slots + 1-byte length per entry), which **does not fit in any current GPU's shared memory**. Forcing every deployment to use 16-bit codes would either trap us in a slower GPU decode mode (dict in global memory) or give up GPU acceleration entirely.

Instead, the format exposes code width as an encoder-time parameter. The bitstream shape is identical at every width — only the dictionary capacity and the code-stream packing change. Recommended values:

| `code_width_bits` | Dict capacity | Flattened dict footprint | Recommended for |
|---|---|---|---|
| **12 (default)** | **4,096** | **~68 KiB** | **Mixed CPU/GPU; SMEM-resident decode on Hopper-class GPUs.** |
| 16 | 65,536 | ~1 MiB | CPU-only deployments where ratio matters most |
| 14 | 16,384 | ~272 KiB | CPU-primary; L2-friendly on GPU |
| 11 | 2,048 | ~34 KiB | Ada-class GPUs (RTX 4090, L40) at higher occupancy |
| 10 | 1,024 | ~17 KiB | Extreme GPU occupancy |

The 12-bit default is deliberate. The flattened table footprint (~68 KiB) lands right inside the per-block shared-memory budget that lets H100 sustain ≥3 blocks per SM (concrete numbers in the GPU section below). At that width the bitstream is exactly FSST12's: two codes packed into three bytes, decodable by the same loop pattern the reference FSST implementation has shipped for years. CPU-only deployments that don't care about GPU at all can opt up to 14- or 16-bit codes and get a tighter dictionary; GPU-only deployments on narrower hardware can opt down to 11- or 10-bit codes for higher occupancy. The encoder picks the width once per array (or per corpus, in Tier 2); the decoder dispatches on it.

### Training algorithm: OnPair, at every code width

This is the design point most likely to be misunderstood: this encoding uses FSST12's *bitstream layout* at 12-bit codes but **not** FSST12's *training algorithm*. At 12-bit codes, FSST12's 5-round greedy iteration is dominated by OnPair's single-pass pair-merge algorithm. The bitstream side of FSST12 is the GPU-friendly part; the training side is replaceable, and OnPair's training fills the 4096-code budget more efficiently. Concretely:

- **Adaptive recursion depth.** FSST caps symbols at 8 bytes. Long repeating substrings (URL prefixes, JSON keys, identifier patterns in code) get fragmented across multiple symbols. OnPair builds long symbols by *chaining* pair-merges — a 30-byte repeated prefix can become a single code via a recursive merge chain, even though no individual merge step ever sees a >2-symbol concatenation. The recursion is implicit in the dictionary structure, not a length cap in the training loop.
- **Longest-match constraints differ.** The FSST paper relies on a "longest-prefix-first" ordering of stored symbols so that the greedy longest-match probe at training time terminates on the longest valid candidate. OnPair sidesteps the issue entirely because codes are fixed-width and the encoder's longest-prefix-matcher is keyed on the dictionary directly. The practical consequence is that OnPair can include candidate symbols that share prefixes with longer symbols, which expands the design space available to its training algorithm.
- **Small-dict efficiency.** With only 4096 codes available, every promotion has to earn its slot. OnPair's pair-merge ordering is plausibly more efficient at small budgets than FSST's gain-with-length-cap heuristic (which is tuned for FSST8's 255 codes), but the relative ratio of FSST12-training vs. OnPair-training at 4096 codes is unmeasured in the public literature — the validation campaign in Part V tests this directly.
- **Single pass.** OnPair training is a single sequential pass; FSST's training does five sampling rounds. The OnPair paper reports compression speeds of 99–229 MiB/s. Direct head-to-head numbers vs. FSST training at the same dict size aren't published; that's another measurement gap to close.

At 4096 codes, OnPair training naturally terminates much earlier than at 65,536 — most of the long-tail pair-merges OnPair-16 would have promoted don't make it in. The truncation drops the most-marginal merges first; the high-frequency, high-gain merges that dominate the ratio are kept. The predicted ratio (worth measuring) is that OnPair-12 lands between FSST12 and OnPair-16, much closer to the latter — because the diminishing returns past ~3–4K codes are real.

### CPU decoder

The decoder runs in two phases.

**Phase 1 — Symbol-table materialization.** Walk the merge-pair list once, building a flattened symbol table in scratch memory. Each new entry concatenates two earlier entries (which are themselves already flattened by induction), so a linear pass suffices:

```rust
let mut symbols: [u8; 16 * (256 + n_pairs)] = /* zero-init */;
let mut lengths: [u8; 256 + n_pairs] = /* zero-init */;
for b in 0..256 { symbols[16*b] = b as u8; lengths[b] = 1; }
for (i, (p0, p1)) in pairs.iter().enumerate() {
    let idx = 256 + i;
    let l0 = lengths[*p0 as usize] as usize;
    let l1 = lengths[*p1 as usize] as usize;
    // memcpy symbols[16*p0 .. 16*p0 + l0] then symbols[16*p1 .. 16*p1 + l1]
    // into symbols[16*idx ..]
    lengths[idx] = (l0 + l1) as u8;  // capped by max_symbol_length
}
```

The flattened table uses a fixed 16-byte stride per entry (zero-padded), which makes Phase 2 cache- and SIMD-friendly at the cost of ~6× memory inflation vs. a packed representation. For a 4096-entry dict this is ~68 KiB — fits in L2 on every relevant CPU. For 65,536 entries this is ~1 MiB — fits in L3.

**Phase 2 — Code-stream decode.** At 12-bit codes the hot loop is the FSST12 pattern:

```c
while (out_pos + 16 <= out_size && in_pos + 4 <= in_size) {
    uint32_t code = unaligned_load_u32(in + in_pos);
    uint32_t c0 = code & 0xFFF;
    uint32_t c1 = (code >> 12) & 0xFFF;
    in_pos += 3;
    unaligned_store_u128(out + out_pos, symbols[c0]);
    out_pos += lengths[c0];
    unaligned_store_u128(out + out_pos, symbols[c1]);
    out_pos += lengths[c1];
}
```

Two unaligned 16-byte stores per three input bytes, with the length array deciding how far the output pointer advances. The stores intentionally over-write (zeroes from the padded symbol slot) and are corrected by the next iteration's advance. This is the same trick FSST uses, generalized to 16-byte symbols.

**SIMD acceleration.** The fixed 16-byte-stride decode table opens up several SIMD acceleration paths. The OnPair reference implementation ([`onpair_cpp`](https://github.com/gargiulofrancesco/onpair_cpp)) uses scalar `memcpy` with deliberate over-copy (always-write-MAX_SYMBOL_LENGTH-bytes, then advance by the actual length), which compilers auto-vectorize and which reaches the ~5.4 GB/s number cited in Motivation. An explicit AVX-512 implementation could use parallel gathers for symbol fetch and byte-level masked-compress primitives (available in AVX-512 VBMI2) for variable-length output packing; this is a future implementation choice, not part of the format. The format guarantees the dictionary is decodable scalar at any code width.

**Per-string random access.** Per-string code offsets are stored in a bitpacked buffer. To decode string `i`: read `code_offsets[i]` and `code_offsets[i+1]`, decode codes from that slice. The flattened symbol table is built once and cached across queries (in Tier 2, the `OnceLock`-cached table from `OnPairLayout` provides this for free).

### GPU decoder (CUDA, Hopper-class and newer)

Primary GPU targets are Hopper (H100, H200), Blackwell (B100, B200), and Ada (RTX 4090, L40). Ampere (A100) is supported but not the design point; older GPUs are not in scope.

The flattened decode table is two arrays — a 16-byte-stride symbol slot per code, plus a 1-byte length per code — totalling 17 bytes per entry without padding. Per-SM shared-memory capacities (max user-shared carveout, per the NVIDIA tuning guides):

| GPU | Generation | SMEM/SM | Achievable blocks/SM for the flattened table |
|-----|------------|---------|----------------------------------------------|
| | | | 1K dict (~17 KiB) / 2K (~34 KiB) / 4K (~68 KiB) / 8K (~136 KiB) / 16K (~272 KiB) |
| **H100/H200** | Hopper (primary) | 228 KB | ≥8 / 6 / **3** / 1 / does not fit |
| **B100/B200** | Blackwell (primary) | 228 KB+ | ≥8 / 6 / **3** / 1 / does not fit |
| **RTX 4090 / L40** | Ada (primary) | 100 KB | 5 / 2 / 1 / does not fit / does not fit |
| A100 (comparison) | Ampere | 164 KB | ≥8 / 4 / 2 / 1 / does not fit |

(Bolded entries are the recommended `code_width_bits = 12` row.)

This is the structural reason GSST hits 191 GB/s with FSST8's ~2 KiB symbol table: a 2 KiB table is essentially free in SMEM, so the GPU isn't trading occupancy for table size. And it's the reason 12-bit is the GPU sweet spot for Hopper-class hardware: 4K × 17 bytes lands at 3 blocks/SM on H100, which is decent occupancy and leaves substantial latency-hiding headroom. The corresponding 16-bit (64K-entry, ~1 MiB) flattened table does not fit in any current GPU's SMEM and forces the slower dict-in-global-memory mode.

**Parallelism granularity: split-parallelism via writer-side checkpoints.** The thing that makes GSST work at near-bandwidth rates is not "one block per string" per se — it is that the writer pre-computes the output byte offset at every segment boundary, so any block or warp can decode any segment without a serial dependency on prior segments. Calling that "block parallelism" is a useful shorthand for the *resulting* compute pattern, but the design choice is really about three knobs: (a) how the code stream is segmented; (b) what compute unit owns one segment; (c) how segments are scheduled onto SMs. All three have first-class consequences for throughput on real Vortex workloads, where string lengths range from URLs and UUIDs at ~30 bytes to free-text columns at multiple KBs.

#### Segment-size analysis

The competing pressures are amortizing per-segment overhead vs. extracting enough parallelism to fill the GPU. The per-segment overhead is one global-memory read of the checkpoint table (~16 bytes; L2-cached after warmup) plus a few cycles of bookkeeping — call it ~30 ns minimum. The per-warp-iteration work, at 12-bit codes with 128 codes/iter (4 codes per thread, 256 bytes coalesced input), is roughly ~25–40 ns when input/output bandwidth and SMEM dict lookups are pipelined behind enough warp parallelism per SM.

That gives a warp-per-segment setup-amortization table:

| Segment size at 12-bit | Warp iters | Work | Setup | Setup % |
|---|---|---|---|---|
| 256 codes | 2 | ~60 ns | ~30 ns | **33% — bad** |
| 512 codes | 4 | ~120 ns | ~30 ns | **20% — marginal** |
| 1024 codes | 8 | ~240 ns | ~30 ns | **11% — OK** |
| 2048 codes | 16 | ~480 ns | ~30 ns | **6% — good** |
| 4096 codes | 32 | ~960 ns | ~30 ns | **3% — great** |

So warp-per-segment at 12-bit codes wants segments of **at least ~1K codes, ideally 2–4K**. At 16-bit codes the analysis is similar but slightly more generous on the small end (no bit-unpacking overhead, cleaner coalescing) — ~512 codes / segment is roughly the OK threshold.

The opposing pressure is parallelism. A typical Vortex chunk of ~500K codes at 4K codes/segment is only ~125 segments — well below H100's ~528 concurrent warps. We'd run out of work before saturating the GPU.

**This is exactly the regime where persistent threads win.** Launch ~512 warps once; each warp atomically claims segments from a global counter and decodes them in a loop. The per-segment overhead drops from ~30 ns (with kernel-grid setup costs) to ~10 ns (just the metadata read; no per-launch barrier), so segments can shrink to ~512–1K codes without setup dominating. Load balancing across heterogeneous segment costs is automatic. This is more complex than block-per-segment but is the right default for production performance.

#### Segmentation strategy

The writer needs to produce a checkpoint table. The four reasonable strategies are:

| Segmentation | Pros | Cons |
|---|---|---|
| Per-string boundaries only | Free (already in our format); natural for random access | Long strings (multi-KB) starve some warps; very-short strings cause warp underutilization |
| Per-K-strings groups (e.g., 32 strings per segment) | Uniform warp work for short-string columns | Coarser random access; two-level offset table |
| Fixed-input-stride checkpoints (every N codes) | Uniform GPU work regardless of string-length distribution | Extra metadata; two-level offset table |
| **Hybrid** (per-string boundaries always; inner checkpoints every ~1–2K codes within long strings) | Workload-adaptive; preserves per-string random access | Most complex writer |

The recommended default is **hybrid**, with the inner checkpoint stride defaulting to ~1K codes at 12-bit (~1.5 KiB input per inner segment, ~4 KiB typical output) and tunable per array. Per-string boundaries provide free random access for short-string columns; inner checkpoints provide uniform GPU work for long-string columns. Metadata overhead at this stride is ~16 bytes per ~1.5 KiB input ≈ 1%.

Short strings (≤1K codes typical) reach exactly one checkpoint — the per-string boundary — and act like the simple per-string-boundaries case. Long strings (>1K codes) get inner checkpoints automatically. The kernel treats both uniformly: it iterates segments from the checkpoint table without caring whether segment boundaries are string boundaries or inner-string checkpoints.

#### Mode A — dict-in-shared-memory (`code_width_bits` ≤ 12), persistent-thread kernel

The fast path. Sketched as persistent threads with warp-per-segment claiming; block-per-segment with larger segments is a simpler-but-slightly-slower variant we should also implement for the reference decoder.

```cuda
__global__ void decode_kernel_smem_persistent(
    const Checkpoint* checkpoints, uint32_t n_segments,
    const uint8_t* codes, uint8_t* out,
    const MergePair* pairs, uint32_t n_pairs)
{
    __shared__ uint4   sym[4096];   // 64 KB at 16 bytes × 4096, bank-interleaved
    __shared__ uint8_t len[4096];   //  4 KB
    __shared__ uint32_t seg_counter; // atomic claim counter, per block

    // Phase 1: cooperative symbol-table materialization.
    //   - First 256 entries: thread t initializes sym[t] = {t, 0, 0, 0, ...}, len[t] = 1.
    //   - Merge-pair walk: parent indices are strictly smaller than the entry being built,
    //     so a single warp walks the list linearly; one __syncthreads at the end.
    //   - Cost: ~60 µs amortized over the kernel lifetime; negligible.

    // Phase 2: warp-per-segment, persistent-thread work loop.
    //   - One warp lane per block claims the next segment via atomicAdd on a global counter;
    //     warp-broadcast to the other lanes.
    //   - Loop: while (segment_id < n_segments) { decode segment; claim next; }
    //   - Per-segment work:
    //     - Read checkpoint[segment_id] -> (input_start, input_len, output_start).
    //     - In a warp-iter loop over the segment:
    //         - 32 threads load 32×4 = 128 codes via 2 coalesced 128-byte reads,
    //           with bit-unpacking to extract the 12-bit codes per thread.
    //         - Each lane: lookup sym[code], len[code] from SMEM (~1–2 cycles each).
    //         - CUB warp-inclusive scan on lengths -> per-lane output offset within iter.
    //         - Warp-cooperative store: pack each lane's (1..16) live bytes into 128-byte
    //           coalesced global writes via __shfl_sync + masked stores. (Same shape as
    //           the variable-length-output handling described in the GSST paper; here it
    //           is simpler because there are no escape codes to detect or handle.)
}
```

Alternative kernels worth implementing (and measuring against the persistent variant):

- *Block-per-segment*, with each block processing one segment of ~8–32K codes via its 8 warps cooperating on output-offset prefix-sum. Simpler than persistent threads. Likely 10–20% slower at the typical segment sizes but easier to reason about and debug. Reference implementation should ship this.
- *Two-stage pipeline* (offset-computation kernel → scatter kernel). Almost certainly loses to single-kernel warp-cooperative on this workload due to extra global-memory traffic, but it's the obvious thing to compare against and helps quantify the value of split parallelism.

Expected throughput on H100: should match or exceed GSST's 191 GB/s. GSST spends cycles handling escape codes and the resulting warp divergence; we do not. With Hopper's higher per-SM SMEM and bandwidth, the prediction is ~250–300 GB/s on H100 for the 12-bit configuration with the persistent-thread kernel and hybrid checkpoints. This is a prediction tied to specific design choices, not a measurement; the validation campaign in the Unresolved Questions section gates it.

**Mode B — dict-in-global-memory (code_width_bits ∈ {14, 16}).** A separate kernel for when the encoder chose a larger dict for ratio reasons but GPU decode is still wanted. A pre-pass kernel walks the merge-pair list and emits the flattened symbol table to global memory once; subsequent decode kernels read it through L2 (50 MB on H100 per NVIDIA's Hopper tuning guide; the 1 MiB symbol table fits trivially with high hit rate). The lane-level lookup uses `__ldcg` or equivalent for a cache-global load. Expected throughput: ~150–200 GB/s on H100. Slower than Mode A but still 30–50× the CPU.

The encoder's choice of `code_width_bits` implicitly selects Mode A or Mode B at decode time. We can additionally support an encoder strategy that trains a 64K-entry dict, measures its on-disk size, and falls back to a 12-bit dict if the larger one doesn't pay off — but that's an optimization, not a format requirement.

**CUDA-specific risks worth flagging in implementation:**
- *Segmentation/checkpoint metadata.* The writer must emit a checkpoint table (input-stride → output-offset) for the GPU decoder to achieve split parallelism. This is small (one entry per segment, ~8 bytes; at the recommended ~1–2K-code inner stride, metadata overhead is well under 1%) but it is *required* for GPU mode to hit peak. Tier-1 arrays without checkpoints will fall back to slower paths on GPU.
- *Bank conflicts on dictionary access.* With 16-byte-wide entries (4 banks per entry), an unlucky access pattern can serialize. Standard fix is a one-word stride or interleaving the low bits of the code with the bank index; the CUB scan helpers handle this correctly.
- *Occupancy vs. SMEM tradeoff.* Mode A at 12-bit codes uses ~68 KB SMEM/block; on H100 this gives ~3 blocks/SM (decent, not maximal). Dropping to 10-bit codes (16 KB SMEM/block) doubles occupancy, but ratio loss is real and must be measured.
- *Per-string boundaries are NOT in the GPU hot path.* They live in a separate buffer that bulk-decode kernels skip entirely; only the random-access path and the writer's segmentation logic read them.

### Per-string random access and LIKE pushdown

**Random access** is direct from the format: read the per-string offset range and decode that slice. The flattened symbol table is built once and cached. For very-short-string workloads, decode latency is dominated by symbol-table materialization rather than the codes themselves — the `OnceLock` cache (Tier 2) or an in-decoder cache (Tier 1) is essential.

**LIKE pushdown** generalizes the technique Vortex's existing FSST encoding uses: build a DFA from the LIKE pattern over bytes, then for each dictionary entry precompute a "transition function" that maps an input DFA state to the state after consuming all the bytes the code expands to. Scanning the compressed code stream is then one DFA step per code.

This applies directly to OnPair: the materialized symbol table *is* the array of byte sequences keyed by code. Precomputation is `O(dict_size × n_states × avg_symbol_length)`. For a typical short LIKE pattern (10–20 chars, ~10–30 DFA states) and the 4K-entry default dict, that's ~500K operations — sub-millisecond. The transition table is `u8[dict_size][n_states]` ≈ 4K × 16 = 64 KiB per query, comfortably cacheable. At 16-bit codes the precompute is 16× larger but still bounded by ~1 MiB per query.

The DFA-over-codes precompute is meaningfully larger than FSST8's (which has only 255 codes), but for any scan of more than a few thousand rows the per-row scan savings dominate the precompute cost. The same is already true of FSST12 vs FSST8.

### Compression

OnPair's single-pass LPM training. Expected compression speed is ~100–230 MiB/s on a single CPU core, comparable to the OnPair paper's reported numbers. In Tier-2 (shared-dict) mode, training is a one-time corpus-wide cost; each chunk's compression is then a tokenization pass and should run 2–5× faster than per-chunk training.

### Why this is the right design

A reasonable reader will ask: why not just adopt one of the published encodings directly?

- **Why not FSST12 alone?** FSST12's bitstream is great. FSST12's training, at 4096 codes, leaves ratio on the table relative to OnPair-style pair-merging — for the reasons in the training section above. The proposal is what you get if you keep FSST12's bitstream but swap in OnPair's training.
- **Why not upstream OnPair-16 alone?** OnPair-16's 64k dictionary does not fit in any current GPU's shared memory. Either we run the GPU decoder in the slower global-memory mode (still available here as Mode B), or we accept code-width parametrization so deployments can pick the GPU-friendly point. The latter is strictly more flexible — and the 16-bit configuration of this RFC is essentially identical to upstream OnPair, so callers who don't need GPU acceleration get upstream's behavior unchanged.
- **Why not GSST alone?** GSST inherits FSST8's ratio; the escape mechanism is still in the bitstream. This proposal gets GSST's kernel structure with a better bitstream underneath.
- **Why both tiers?** Tier 1 covers the common case (per-array independence, FSST-like deployment). Tier 2 covers the corpus-wide-dictionary case, which gives non-trivial ratio improvements when the dataset has cross-chunk redundancy. Vortex already has this two-tier pattern for raw values (Array encoding + DictLayout); the OnPair encoding just extends it to compressed strings.

## Compatibility

This is a new encoding. It does not change existing arrays or layouts. The encoding registry gains two new entries — `OnPair` (Array) and `OnPairLayout` (Layout) — and readers without OnPair support will be unable to read arrays/files written with it. The existing FSST encoding remains the supported default for now; we recommend keeping it as the safe option until OnPair has accumulated production miles.

There are no migrations required. Files using FSST stay using FSST; files using OnPair are a deliberate per-writer choice.

The format-stability commitments for the OnPair encoding:

- The metadata schema is Prost-encoded and follows Vortex's existing convention for forward-compatible additions (optional fields with explicit defaults).
- `code_width_bits` is part of the format and must be honored exactly by any conforming decoder. Adding new allowed values in the future (e.g., 13-bit, 15-bit) is a forward-compatible addition only if older readers reject the unknown width cleanly rather than misinterpreting it.
- The merge-pair list encoding is fixed: u16 parent indices at all code widths ≤ 16; widths beyond 16 are not currently in scope.
- The flattened symbol-table layout (16-byte stride, zero-padded) is a decode-side implementation choice, not a format requirement. Future decoders may choose a different in-memory layout.

## Drawbacks

- **Decoder scratch memory.** Up to ~1 MiB of flattened symbol-table scratch per active decoder (at 16-bit codes), or ~68 KiB at the default 12-bit width. Per-thread footprint; not free but well-bounded. Tier-2 amortizes this across all chunks sharing a dictionary.
- **CUDA-only GPU.** This RFC scopes GPU decode to NVIDIA. Targeting AMD (ROCm/HIP) or vendor-neutral compute (SYCL, Vulkan) would be useful and is not in scope — see Future Possibilities.
- **Training is slower than no training.** Per-array OnPair-style training (~100–230 MiB/s) is meaningfully slower than encoding against a pre-existing dictionary. Tier 2 mitigates this for workloads with corpus-wide redundancy but adds a write-time training pass on the dataset.
- **Larger metadata than FSST8.** The merge-pair list is ~4 bytes/entry; at 4K entries that's ~16 KiB of dictionary, vs. FSST8's ~2 KiB. Negligible at array-scale but worth noting for very-small arrays. Tier 2 makes this a one-time corpus-level cost.
- **LIKE-pushdown precompute is larger.** Precomputing the DFA transition table is O(dict_size × n_states). At the 12-bit default this is roughly FSST12-equivalent; at 16-bit it's ~16× larger than FSST12 but still sub-millisecond on real hardware. For point-lookup queries (decode one row) the precompute is unjustified — fall back to decode-then-filter in that case.
- **More moving pieces than a single fixed-width encoding.** `code_width_bits` is a real configuration knob the writer must pick (or have picked for them by encoder heuristics). The validation campaign will need to recommend a default heuristic.

## Alternatives

- **Adopt FSST12 directly and stop there.** Smallest delta from the current encoding; would deliver a real ratio + decode-speed lift over FSST8, and the SMEM-fits property is the same as OnPair-12. But the training algorithm is suboptimal at 4096 codes — OnPair's pair-merge approach packs more compression into the same bitstream. Reduces to "OnPair-12 with worse training", strictly dominated on the priority-2 axis.
- **Adopt OnPair-16 directly and stop there.** Best CPU ratio of the alternatives, but the 1 MiB flattened dict forces GPU decode through global memory (Mode B), giving up ~30% of the throughput that's achievable at 12-bit codes. Acceptable for CPU-only deployments; the wrong choice as a default if GPU is even on the roadmap.
- **Adopt GSST directly and stop there.** Best GPU throughput of the published encodings, but inherits FSST8's ratio and escape-branch CPU cost. Solves only the GPU surface.
- **Use a fixed-12-bit format with no `code_width_bits` parameter.** Simpler. But Vortex deployments differ — some don't care about GPU; some are GPU-first on Ada hardware where 11-bit is the sweet spot. Single-width forces a global compromise. The code-width parameter is cheap to support (same bitstream, different table sizes) and the flexibility is real.
- **Use Vortex's existing `DictLayout` directly for string compression.** `DictLayout` works at the value level — each unique full string is one dictionary entry — which is excellent for low-cardinality columns (categorical strings) and bad for high-cardinality text. OnPair works at the byte/merge level, complementary to `DictLayout`. The two coexist; the writer picks the right one per column.

## Prior Art

- **FSST** — *FSST: Fast Random Access String Compression*, Boncz, Neumann, Leis. PVLDB Vol 13. https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf. Reference implementation: https://github.com/cwida/fsst (MIT), which ships both the FSST8 variant from the paper and a 12-bit FSST12 variant introduced in the source tree (see `fsst12.h` / `libfsst12.cpp`).
- **GSST** — *GSST: Parallel string decompression at 191 GB/s on GPU*, Vonk, Hoozemans, Al-Ars. ACM SIGOPS Operating Systems Review, Vol. 59 No. 1, pp. 55–61, 2025. https://dl.acm.org/doi/10.1145/3759441.3759450
- **GPU-side FSST encoding** — *High Throughput GPU-Accelerated FSST String Compression*, Anema, Hoozemans, Al-Ars, Hofstee. VLDB 2025 ADMS Workshop. Source: https://github.com/timanema/fsst-gpu (Apache-2.0).
- **OnPair** — *OnPair: Short Strings Compression for Fast Random Access*, Gargiulo, Venturini. arXiv:2508.02280, August 2025. https://arxiv.org/abs/2508.02280. C++ reference implementation: https://github.com/gargiulofrancesco/onpair_cpp (MIT). Rust reference implementation: https://github.com/gargiulofrancesco/onpair_rs (MIT).
- **Vortex's current FSST integration** — https://github.com/vortex-data/vortex/tree/develop/encodings/fsst (Apache-2.0) — particularly the LIKE-pushdown DFA, which generalizes to the OnPair encoding directly.
- **Vortex's `DictLayout`** — https://github.com/vortex-data/vortex/tree/develop/vortex-layout/src/layouts/dict (Apache-2.0) — the integration template for Tier 2's cross-chunk dictionary sharing pattern.

## Unresolved Questions

These will be settled through the validation campaign and during implementation review.

- **Encoder code-width selection policy.** Options: (a) caller specifies `code_width_bits` explicitly per array; (b) caller specifies a deployment-target label (`cpu_only`, `mixed`, `gpu_first`) and the encoder picks; (c) encoder picks dynamically based on data characteristics (entropy, distinct n-gram count, expected ratio at each width). The current proposal is (b) as the default with (a) as an escape hatch; (c) is appealing but requires the data-characteristics heuristic to be measured before committing.
- **`MaxSymbolLength`.** 16 bytes makes the SIMD decoder clean (one u128 store per code). If long-prefix workloads (URLs, deeply-nested JSON keys) measurably benefit from 24- or 32-byte symbols, we may want a variant. The cost is wider per-decode SIMD; the benefit is a tighter dict for very-redundant data. Worth a focused micro-benchmark.
- **Tier-2 `dict` layout type.** Should the shared `dict` `LayoutRef` be a `BinaryView` (one variable-length entry per code, suitable for direct predicate pushdown on the dict), a plain `Buffer` of u16 pairs (compact, opaque), or something else? `DictLayout` stores values as a typed array because predicate pushdown operates on typed values — we may want the same so the dict can be queried directly without re-materialization.
- **Bitpacked code-stream alignment at non-byte-aligned widths.** Only the 16-bit width has a clean per-code byte boundary; 12-bit packs two codes per three bytes (FSST12's pattern) and 10/11/14-bit codes are bitpacked at arbitrary positions. Does the decoder require segment-aligned access (every checkpoint starts at a byte boundary, or even a 16-byte boundary), or can it tolerate arbitrary start offsets within a packed stream? GPU coalescing strongly prefers byte-aligned starts; CPU is indifferent. The Vortex writer should pad each segment to a clean byte boundary; the exact alignment requirement and padding rules need to be specified.
- **GPU parallelism granularity and segmentation strategy.** The proposal recommends a persistent-thread kernel with warp-per-segment claiming, fed by a hybrid checkpoint table (per-string boundaries always; inner fixed-input-stride checkpoints every ~1–2K codes within long strings). The argued-out segment-size math is in the GPU section; the open question is whether the kernel-design and segment-size priors match the workload shapes Vortex's largest users actually have, especially when short and long strings are mixed in the same column. The validation sub-benchmark above is the gate. The format reserves a header bit for checkpoint-table presence; the *exact* checkpoint stride is encoder-time-configurable.
- **Validation benchmark plan.** The substantive open question. Detailed below.

### Validation campaign

Every design choice above is a prior, not a conclusion. Before any production commitment, the following measurements must be done on representative Vortex workloads (TPC-H/TPC-DS string columns, JSON columns, ClickBench string columns, the existing Public BI benchmark string columns):

1. **CPU decode throughput head-to-head.** FSST8 vs FSST12 vs OnPair across `code_width_bits ∈ {12, 14, 16}` (with the 16-bit row being upstream OnPair-as-published). Scalar + AVX-512. Establishes the CPU baseline.
2. **Training algorithm at 12-bit codes.** Hold the bitstream constant (12-bit, 4096 dict, FSST12 layout) and vary only the training: (a) FSST12's 5-round greedy iteration, (b) OnPair's single-pass pair-merge with early termination at 4096 codes, (c) OnPair-16 trained to full capacity then truncated to top-4096-by-coverage. The plan recommends (b); this benchmark tests that prior. **This is the load-bearing measurement for the whole RFC.**
3. **GPU decode throughput sweep.** Port GSST to the same test harness as a baseline; then benchmark the OnPair encoding at `code_width_bits ∈ {10, 11, 12, 14, 16}` on H100, Ada (RTX 4090), and A100 (for comparison). Confirms the Mode A vs Mode B crossover and the per-GPU code-width recommendation.

   **Sub-benchmark: parallelism granularity.** At the recommended `code_width_bits = 12`, measure four kernel designs — warp-per-segment, block-per-segment, persistent-thread + warp-per-segment claim, persistent-thread + block-per-segment claim — at three segment sizes (~512, ~2K, ~8K codes) and three workload shapes (short strings ~30 bytes mean, mixed bimodal distributions, long free-text bodies averaging >1 KB). Also vary the writer's segmentation strategy (per-string boundaries only / per-K-strings / fixed-input-stride / hybrid). The proposal recommends the **persistent-thread + warp-per-segment** kernel with **hybrid checkpoints** at ~1–2K-code inner stride; this benchmark tests that prior on real workload shapes. This is the load-bearing measurement for the GPU side of the RFC.
4. **Ratio vs throughput Pareto frontier on GPU.** For each `code_width_bits`, plot decompression throughput against compression ratio. Visualize where the design lives in tradeoff space; identify the right defaults per GPU class.
5. **Dictionary materialization cost.** Cold and warm decode-start time at all code widths. Drives the random-access strategy and the Tier-2 caching policy.
6. **Per-string decode latency.** OnPair vs FSST for short strings (names, URLs, UUIDs). Verify SIMD ramp-up doesn't make short-string decode worse than FSST's tight scalar loop.
7. **LIKE pushdown end-to-end.** DFA precompute time + per-code-stream scan throughput, vs. Vortex's existing FSST DFA, on representative LIKE workloads (e.g., ClickBench Q19/Q20).
8. **Tier-2 ratio uplift.** Train one dict on each whole dataset and Arc-share across chunks; measure ratio improvement vs. per-array (Tier-1) training.
9. **Compression speed.** Per-array LPM training cost, including cold-cache effects.
10. **Tier-2 dict materialization cost.** Confirm `OnceLock`-cached dict reuse pays for itself across realistic query patterns. Decide if `OnPairLayout` needs a different caching policy than `DictLayout`'s default.

## Future Possibilities

- **Non-CUDA GPU support.** ROCm/HIP for AMD, SYCL for Intel, Vulkan/Metal for cross-vendor compute. The kernel shape generalizes; only the SMEM/L2 budgets and the primitive names change. A natural follow-on.
- **Encoder heuristics for code-width selection.** Once the validation data exists, derive an encoder heuristic that picks `code_width_bits` automatically from data characteristics (n-gram entropy, distinct-prefix count, observed compression at each width on a sample). Removes one knob from the caller.
- **Cross-encoding pushdown.** The DFA-over-codes pushdown technique generalizes to any fixed-width-code dictionary encoding. Once the OnPair encoding lands, Vortex's compute layer can apply the same machinery to other encodings that materialize a code-to-bytes table at decode start.
- **Wider codes.** 20- or 24-bit codes for highly-redundant corpora where 16-bit isn't enough. Probably never wanted, but the format leaves room — the merge-pair list representation would need to widen to u32 indices and the decoder would need a new packing layout.
- **Adaptive recompression.** Vortex chunks could re-train their dictionaries at compaction time based on observed access patterns. Out of scope for this RFC; worth noting as a possibility.
