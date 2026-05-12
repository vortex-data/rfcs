# OnPair16: A GPU-Friendly FSST-Family String Encoding for Vortex

**Authors:** Will Manning
**Status:** Proposal
**Start Date:** 2026-05-12
**RFC PR:** [vortex-data/rfcs#59](https://github.com/vortex-data/rfcs/pull/59)

## Summary

This RFC adds a new array to Vortex's existing FSST encoding family ([`encodings/fsst`](https://github.com/vortex-data/vortex/tree/6f54d3d6dcc3d9b405658b4afcb4483616daa76f/encodings/fsst)) — a cousin to today's `FSSTArray`, structurally similar but built around **OnPair's pair-merge codebook construction** (Gargiulo & Venturini, [arXiv:2508.02280v1](https://arxiv.org/abs/2508.02280v1)) instead of FSST's 5-round evolutionary symbol selection, and decompressable on NVIDIA GPUs at GSST-class throughputs. The encoding inherits FSST's field-level random-access design archetype (small dictionary, per-string decode, compressed-domain predicate pushdown), keeps Vortex's existing `FSSTArray` integration shape, and lives in the same encoding crate.

The contributing prior work, in order of how much of the new encoding is theirs:

- **OnPair** (Gargiulo & Venturini, August 2025): the single-pass longest-prefix-matching pair-merge codebook construction — much cheaper than classical BPE and, we hypothesize, better at small dictionary sizes than FSST's 5-round evolutionary algorithm. The published compressed-domain predicate machinery (KMP / Aho–Corasick / prefix / equality / boolean composition over the token stream, all in [`onpair_cpp@ae590713/include/onpair/search/automata/`](https://github.com/gargiulofrancesco/onpair_cpp/tree/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/search/automata), MIT-licensed). The OnPair16 variant's bounded-symbol-length decode pattern (paper §3.4.2, Algorithm 3) that makes SIMD CPU decode straightforward. The on-disk dictionary representation (lexicographically-ordered flat bytes + Arrow-style offsets) that supports the prefix-range automata. OnPair is a recent, under-recognized substantive contribution to columnar string compression; this RFC adopts it wholesale and credits accordingly.
- **FSST** (Boncz, Neumann, Leis, [PVLDB Vol 13, 2020](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf)): the family archetype — field-level (per-string) random-access compression with a small static symbol table, the "no-block-decode-required" property, the late-decompression discipline that Vortex's compute layer already exploits. This RFC's encoding sits as a sibling to FSST in that family, *not* as a replacement; the two are complementary code-construction strategies over the same architectural idea.
- **FSST12** (post-paper variant in [`cwida/fsst@e638d4c/fsst12.h`](https://github.com/cwida/fsst/blob/e638d4cf8c26129d73c242a4127b42b975de5b63/fsst12.h)): the 12-bit-codes-packed-two-per-three-bytes bitstream layout we reuse at the 12-bit width. The training algorithm is FSST's, not used here; only the bit-packing is reused.
- **GSST** (Vonk, [TU Delft MSc thesis 2024](https://repository.tudelft.nl/) + Vonk, Hoozemans, Al-Ars, [ACM SIGOPS OS Review 2025](https://dl.acm.org/doi/10.1145/3759441.3759450)): the GPU decoder design via the split parallelism format (per-split uncompressed-size metadata in the block header → intra-SM parallelism without serial dependencies) and accompanying shared-memory + alignment + async-transfer memory-management optimizations.

The new pieces on top of those four contributions:

1. **Code width is encoder-parametric** (10/11/12/14/16 bits), with **12-bit as the default** so the dictionary fits in NVIDIA Hopper-class shared memory for GPU decode. (OnPair's paper §3.6 recommends 16 bits for CPU-only; this RFC's lower default is driven by GPU SMEM — see §[`code_width_bits`](#code_width_bits-the-load-bearing-knob).)
2. **A first-class CUDA decoder** that ports GSST's split parallelism format onto OnPair's escape-free bitstream — the GSST thesis's recommended one-block-per-SM kernel is the reference variant; persistent-thread warp-per-split claiming is a measured alternative.
3. **Two integration tiers** mirroring Vortex's existing `FSSTArray` / `DictLayout` split: `OnPair16Array` for self-contained per-array dictionaries (the default; analog of `FSSTArray`), `OnPair16Layout` for cross-chunk dictionary sharing via Arc-shared `LayoutRef` (analog of `DictLayout`).

The encoding is recommended to land in three stages (see [Staging](#staging)): **Stage 1** ships `OnPair16Array` at 16-bit codes, CPU-only, no GPU; **Stage 2** adds the parametric width, the `splits` buffer, and the Mode A GPU kernel; **Stage 3** adds `OnPair16Layout` (Tier 2 shared dict), Mode B (dict-in-global-memory) for 14/16-bit GPU decode, and the optional Coalesced Memory Access Format. Each stage is independently mergeable and validatable.

**A note on naming.** The Array type is `OnPair16Array` rather than something with "FSST" in the name, because the substantive innovation it carries is OnPair's. The encoding crate location (`encodings/fsst`) does the family grouping; the type name credits the algorithmic contribution. This matches the precedent in [`vortex-fastlanes`](https://github.com/vortex-data/vortex/tree/6f54d3d6dcc3d9b405658b4afcb4483616daa76f/encodings/fastlanes), where `BitPackedArray`, `DeltaArray`, `FoRArray`, etc. live in one family-named crate with algorithm-named types.

## Motivation

Vortex currently ships FSST for variable-length string compression — the encoding established by Boncz, Neumann, and Leis at VLDB 2020 ("[FSST: Fast Random Access String Compression](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf)"). FSST is the design archetype for "field-level, random-access, small-dictionary" string compression in columnar databases: per-string decode without block-level dependencies, ~2 GB/s decode (with AVX-512), late-decompression discipline so the compute layer can push predicates into the compressed representation. Five-plus years on, FSST is the strongest baseline in this corner of the design space, and the right starting point for a Vortex encoding.

Three subsequent contributions sharpen the design and motivate this RFC. They are best understood as improvements *within the FSST family*, not as replacements:

1. **OnPair's codebook construction is a strict improvement over FSST's at small dictionary sizes.** OnPair (the broader algorithm) and **OnPair16** (the bounded-symbol-length variant we adopt here) are introduced together in [Gargiulo & Venturini, arXiv:2508.02280v1](https://arxiv.org/abs/2508.02280v1), August 2025. The paper contributes a single-pass longest-prefix-matching pair-merge training algorithm that's an order of magnitude cheaper than classical BPE and avoids FSST's "dependency issue" (FSST paper §4.1) through a different mechanism — incremental merging rather than gain-estimation-after-the-fact. The resulting dictionary has up to 65k tokens, none of which are reserved as escape codes (codes 0–255 are the raw bytes, so every byte has a learned-or-literal representation). OnPair16 caps symbol length at 16 bytes, enabling a tight `memcpy(buf, src, 16); buf += len[t]` decode loop — a SIMD-friendly decode pattern that achieves **6.5–7.8 GB/s** CPU decode (paper Table 3, Intel Core Ultra 7 265K, AVX2 only — no AVX-512 on the test hardware). The OnPair paper also ships a published **compressed-domain predicate machinery** (Aho–Corasick, KMP, prefix, equality, boolean composition over the token stream — all in [`onpair_cpp@ae590713/include/onpair/search/automata/`](https://github.com/gargiulofrancesco/onpair_cpp/tree/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/search/automata)) that solves the same problem Vortex's FSST DFA solves, with a different and arguably more general technique. OnPair is recent and has received less attention than its contributions warrant; this RFC adopts OnPair16 in full and credits accordingly throughout. (Where the algorithm is the same between OnPair and OnPair16, this RFC uses "OnPair" generically; where the 16-byte symbol cap matters, "OnPair16" is used explicitly.)

2. **The FSST family already has a wider-dict member with the escape branch absent.** The `cwida/fsst` reference implementation ([`fsst12.h@e638d4c`](https://github.com/cwida/fsst/blob/e638d4cf8c26129d73c242a4127b42b975de5b63/fsst12.h#L46)) ships an FSST12 variant whose source comment states it "will outperform [FSST8] on datasets that are more chaotic, such as JSON and widely diverse URLs" (note the original's "dan/than" typo). FSST12 has 4096 codes instead of 255 — 16× more space for learned symbols — but retains FSST's escape mechanism and its 5-round evolutionary symbol-selection algorithm. It's not benchmarked publicly in head-to-head form. This RFC reuses FSST12's bit-packing layout at the 12-bit code width (two codes per three bytes) — well-tested, GPU-friendly — but swaps in OnPair's training algorithm because at the same 4K-code budget OnPair's incremental pair-merging plausibly outperforms FSST's gain-estimation-and-reorder heuristic. The training-algorithm question is the load-bearing empirical claim of this RFC; the validation campaign measures it directly (sub-benchmark 2).

3. **GPU string decompression is now a real target for FSST-family encodings.** GSST — *GPU Static Symbol Table* — (Vonk, [TU Delft MSc thesis, 2024](https://repository.tudelft.nl/); Vonk, Hoozemans, Al-Ars, [ACM SIGOPS Operating Systems Review Vol. 59 No. 1, 2025](https://dl.acm.org/doi/10.1145/3759441.3759450)) demonstrates that FSST-class encodings can be decoded on GPU at near-bandwidth rates — **191 GB/s on A100 at 2.74× compression ratio**. The headline design is three format optimizations (block parallelism, split parallelism, coalesced memory access; thesis §4.2) and three memory-management optimizations (shared-memory-resident symbol table, aligned memory accesses, asynchronous data transfer; §4.3). The *split parallelism format* specifically is what enables intra-SM parallelism without per-thread serial dependencies: the writer stores per-split uncompressed sizes in the block header so each GPU thread within an SM begins decoding at a known output offset. Tim Anema's complementary ADMS 2025 work *"High Throughput GPU-Accelerated FSST String Compression"* ([paper](https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/ADMS/ADMS25-01.pdf); [`timanema/fsst-gpu@a0b639c3`](https://github.com/timanema/fsst-gpu/tree/a0b639c33bb0d6d6272c04a2e1c83877d1941f2f), Apache-2.0) reports **74 GB/s** encode-side throughput on RTX 4090 using a thread-voting matching mechanism and a stream-compaction output pipeline. Both are FSST-family work; the GSST decoder approach ports cleanly onto OnPair's escape-free bitstream because the FSST family's "table-lookup-per-code, variable-length output" shape is the same.

### Quick comparison

| Axis | FSST8 (Boncz 2020) | FSST12 (`cwida/fsst@e638d4c`) | OnPair16 (paper) | This RFC |
|---|---|---|---|---|
| Code width | 8 bits + escape | 12 bits + escape | 9–16 bits (16-bit default) | 10/11/12/14/16 bits (12-bit default) |
| Dict capacity | 255 symbols | 4096 symbols | 512–65,536 tokens | 1024–65,536 tokens |
| Escape codes | Yes | Yes | No | No |
| Symbol length cap | 8 bytes | 8 bytes | 16 bytes | 16 bytes |
| Training | 5-round evolutionary + sampling (Section 4.1) | Same as FSST8 | Single-pass LPM pair-merge | OnPair16's algorithm unchanged |
| On-disk dict | u64 symbols + u8 lengths (~2 KiB) | Same shape (~32 KiB) | Flat bytes + u32 offsets, lex-ordered (~0.25–1 MiB) | Same as OnPair16 |
| CPU decode (paper) | "approaching 2 GB/s" (AVX-512) | unreported | 6.5–7.8 GB/s (AVX2) | inherits OnPair16's path |
| CPU compress (paper) | 1–3 GB/s (AVX-512), ~325–504 MiB/s (AVX2) | unreported | 137–229 MiB/s (AVX2, no AVX-512 yet) | inherits OnPair16's path |
| GPU decode | hostile (escapes + var-stride) | better but still has escapes | not implemented; structurally GPU-friendly | 191 GB/s expected at 12-bit (GSST-class) |
| Predicate pushdown | byte-DFA after decode | same | Compressed-domain automata (KMP, Aho–Corasick, prefix, equality, boolean composition) — onpair_cpp ships these | Inherits OnPair's automata |
| Random access | per-string | per-string | per-string | per-string |

The FSST family is converging on **fixed-width codes, no escapes, larger dictionaries trained adaptively from the data**, with GPU decode as a now-tractable target. FSST12 covers the bitstream-shape side. OnPair contributes the training algorithm + the OnPair16 SIMD-friendly decode pattern + the compressed-domain predicate machinery. GSST contributes the GPU decoder shape. None of these alone occupies the design point Vortex actually wants — a single FSST-family member competitive on both CPU and GPU with first-class shared-dictionary support and pre-existing query-pushdown machinery — but assembled together, they do.

### Priorities

In strict priority order:

1. **Decompression throughput on CPU and GPU.** Primary metric. We are willing to sacrifice modest compression ratio for substantial decode-throughput improvements on both surfaces.
2. **Compression ratio.** Secondary; should match or exceed FSST on representative workloads.
3. **Compression throughput.** Tertiary; OnPair16's reference implementation reaches ~150–230 MiB/s on a single AVX2 core (no AVX-512 acceleration yet), which is ~½× FSST's AVX2-only compression speed and ~⅙× FSST's AVX-512 compression speed. SIMD-accelerating OnPair16's hot path (longest-prefix matching against a hash map; bytewise copies) is straightforward; an AVX-512-vectorized OnPair16 should close most of the gap to FSST AVX-512 (~GB/s). The reference-impl numbers are the floor.

## Design

### Lineage

This RFC's encoding sits in Vortex's FSST encoding family alongside today's `FSSTArray` — a cousin in the same crate, applying the algorithmic improvements that have accumulated in the field over the five years since FSST's publication.

The family ancestry, and the specific contributions inherited:

- **FSST** ([Boncz, Neumann, Leis, PVLDB 2020](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf)) — the family archetype. Field-level (per-string) random-access decompression. Small static symbol table that lives in L1 cache. The "strings stay strings" property (FSST paper §3.4) — compressed bytes pass through any operator that handles byte strings. Late-decompression discipline so the compute layer pushes predicates into the compressed representation. ~2 GB/s decode (AVX-512) on the original paper's hardware. Vortex's current `FSSTArray` is FSST-the-encoding embedded as buffers + a Vortex-implemented DFA pushdown that generalizes the "future work" idea of FSST §3.2. This RFC takes the entire FSST design archetype as the starting point.

- **OnPair** ([Gargiulo & Venturini, arXiv:2508.02280v1](https://arxiv.org/abs/2508.02280v1)) — the codebook-construction improvement. The single-pass longest-prefix-matching pair-merge algorithm (paper §3.2) replaces FSST's 5-round evolutionary symbol selection (FSST paper §4) — a deliberate restructuring of how symbols come into the dictionary, motivated by but conceptually distinct from BPE. The paper also contributes (a) **OnPair16**, a bounded-symbol-length variant (paper §3.4.2, Algorithm 3) whose 16-byte cap enables a tight `memcpy(buf, src, 16); buf += len[t]` decode hot loop — the source of the 6.5–7.8 GB/s CPU decode result in paper Table 3; (b) the **lexicographically-ordered flat-bytes-with-Arrow-offsets on-disk dictionary format** (paper §3.5, [`include/onpair/core/dictionary.h`](https://github.com/gargiulofrancesco/onpair_cpp/blob/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/core/dictionary.h)) — the lex-order is load-bearing for the prefix automaton's O(log n) range computation; (c) the published **compressed-domain predicate machinery** ([`include/onpair/search/automata/`](https://github.com/gargiulofrancesco/onpair_cpp/tree/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/search/automata) — KMP, Aho–Corasick, prefix, equality, boolean composition over the token stream); and (d) a parameter sweep of 9–16-bit code widths in the reference implementation. This RFC takes (a)–(c) wholesale and uses (d) plus a tighter recommended subset of (d) for the parametric code-width design.

  OnPair is a recent, substantive contribution that has received less attention than its results warrant. This RFC's design depends on it heavily and credits accordingly throughout. *If you are reading this RFC and have not yet read the OnPair paper, the paper is the more important document — read it first.*

- **FSST12** (post-paper variant in [`cwida/fsst@e638d4c/fsst12.h`](https://github.com/cwida/fsst/blob/e638d4cf8c26129d73c242a4127b42b975de5b63/fsst12.h), MIT) — a small but useful technical reference. Boncz et al. extended the FSST source tree with a 12-bit variant after the paper landed; we adopt its 12-bit bit-packing layout (two codes packed into three bytes) at the 12-bit code width, both because the layout is well-tested in production-adjacent code and because at 12-bit codes there's no reason to invent a different packing. FSST12 itself retains the FSST training algorithm and escape mechanism; we keep only the bit-packing layout.

- **GSST** (Vonk, 2024 thesis Chapter 4; Vonk, Hoozemans, Al-Ars 2025 paper) — the GPU decoder design for the FSST family. Block parallelism format, split parallelism format, coalesced memory access format (thesis §4.2). Shared-memory-resident symbol table, aligned memory access, asynchronous host↔device transfers (§4.3). The GSST design is articulated specifically for FSST8's bitstream; it ports cleanly onto OnPair's escape-free bitstream because the "table-lookup-per-code, variable-length output" decode shape is the same FSST-family shape, simpler in our case because there are no escape codes to detect or handle.

**Caveat on the OnPair reference impl's stability.** The OnPair C++ implementation is pre-1.0; its README explicitly states *"The on-disk format and the public API may change without notice. There are no tagged releases yet — pin to a specific commit hash if you embed the library."* This RFC pins to commit `ae590713`. Any non-trivial upstream format change is a coordinated upstream-and-Vortex update.

What's new in this RFC, beyond the contributions above: the combination — bringing OnPair's training + OnPair16's bounded decode pattern + OnPair's compressed-domain automata into Vortex, packing the codes at 12 bits using FSST12's bit-packing layout (the 12-bit default is a Vortex-specific choice; the OnPair paper §3.6 recommends 16 bits, but GPU SMEM constraints flip the recommendation here), and porting GSST's GPU decoder design onto OnPair's bitstream. The new pieces are: the GPU kernel implementation, the `OnPair16Array` and `OnPair16Layout` Vortex integration as a cousin to today's `FSSTArray` in [`encodings/fsst`](https://github.com/vortex-data/vortex/tree/6f54d3d6dcc3d9b405658b4afcb4483616daa76f/encodings/fsst), and the validation campaign that measures the cross-product.

### Current Vortex state

The existing Vortex encoding for strings is FSST, integrated as `FSSTArray` in [`encodings/fsst`](https://github.com/vortex-data/vortex/tree/6f54d3d6dcc3d9b405658b4afcb4483616daa76f/encodings/fsst) (commit `6f54d3d6`, Apache-2.0). It is a self-contained per-array encoding with the symbol table embedded as buffers; the FSST8 symbol table is ~2 KiB; the encoding implements per-string random access and a DFA-based LIKE pushdown (Vortex's compute kernel, generalizing the "string matching as future work" idea from FSST paper §3.2 — Boncz et al. did not implement this in the original paper). The encoding registers in Vortex's encoding registry alongside `DictLayout` ([`vortex-layout/src/layouts/dict`](https://github.com/vortex-data/vortex/tree/6f54d3d6dcc3d9b405658b4afcb4483616daa76f/vortex-layout/src/layouts/dict)), the existing value-level shared-dictionary layout.

This RFC adds two cousin types to the same FSST encoding crate: **`OnPair16Array`** (an analog of `FSSTArray`, Stage 1) and **`OnPair16Layout`** (an analog of `DictLayout`, Stage 3). They live alongside `FSSTArray` in `encodings/fsst` — the family-shared crate — for the same reason `BitPackedArray`, `DeltaArray`, and `FoRArray` cohabit `encodings/fastlanes`: they share the family's architectural archetype (in FSST's case: small static symbol table, field-level random access, late-decompression-with-pushdown) while differing in the substantive algorithmic choice (codebook construction; here, OnPair's pair-merge instead of FSST's evolutionary algorithm). `FSSTArray` remains the default until `OnPair16Array` has accumulated production miles; both ship from the same crate going forward.

### Two-tier Vortex integration

Vortex already has the right primitive for cross-chunk dictionary sharing: `DictLayout` has two children (`values`, `codes`) and sharing is expressed *structurally* by Arc-sharing the `values` `LayoutRef` across multiple `DictLayout`s wrapped in a `ChunkedLayout` — no IDs, no hashes, no side tables. The reader lazy-materializes the values array via `OnceLock` and caches it across calls; predicate pushdown evaluates on the (small) dict array and applies via `take`. The OnPair16 encoding mirrors this exactly.

The proposed encoding has two surfaces:

**Tier 1 — `OnPair16Array` (analog of `FSSTArray`).** Self-contained per-array encoding with the dictionary embedded as buffers. The default mode, suitable for any workload that doesn't benefit from a corpus-wide dictionary. The on-disk format mirrors OnPair's reference implementation directly:

```
buffers:
  [0] dict_bytes         : u8 [dict_bytes_size + DECOMPRESS_BUFFER_PADDING]
                                                       // flat concatenation of token bytes,
                                                       // lexicographically ordered; padded by
                                                       // MAX_SYMBOL_LEN bytes to make
                                                       // unconditional 16-byte over-copy safe
  [1] dict_offsets       : u32[n_tokens + 1]           // Arrow-style; offsets[i]..offsets[i+1]
                                                       // is the byte range of token i in dict_bytes
  [2] code_offsets       : u32[n_strings + 1]          // Arrow-style; per-string boundaries
                                                       // in the codes buffer (in *codes*, not bytes)
  [3] codes              : packed [n_codes]            // bit-packed codes; LSB-first per OnPair
  [4] splits             : u32[n_splits]               // OPTIONAL; GSST-style: per-split
                                                       // *uncompressed sizes*. Present iff the
                                                       // writer prepared the array for GPU decode.
metadata (Prost):
  code_width_bits        : u8     // 10/11/12/14/16
  n_tokens               : u32    // ≤ 2^code_width_bits; 256 (single-byte tokens) ≤ n_tokens
  n_strings              : u32
  n_codes                : u32
  n_splits               : u32    // 0 iff splits buffer absent
  codes_per_split        : u32    // GSST's "constant number of codes" per split; valid iff n_splits > 0
  uncompressed_bytes     : u64
  dict_bytes_size        : u32    // = dict_offsets[n_tokens]; offsets buffer cannot exceed 0.25 MiB
  flags                  : u8     // bit 0: coalesced-memory-access reordering applied to codes
```

The first 256 tokens are reserved for the 256 single-byte values (this is OnPair's invariant, paper §3.2). Tokens `256..n_tokens` are learned merge-pair symbols, each represented by its full byte sequence stored contiguously in `dict_bytes`. **The dictionary is stored in lexicographic order of byte sequences** (paper §3.5; [`dictionary.h`](https://github.com/gargiulofrancesco/onpair_cpp/blob/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/core/dictionary.h) docs) — this is load-bearing for the compressed-domain prefix automaton, which uses sorted order to compute prefix-range token-id intervals in O(log n_tokens).

The `dict_bytes` buffer is padded by `MAX_SYMBOL_LEN = 16` zero bytes past the logical end, so a decoder over-copying 16 bytes from any valid offset stays within the allocated buffer (paper §3.5; `pad_for_decoder()` in `dictionary.h`).

For a 4K-token dictionary on Book Titles–like data, the dict footprint is ~76 KiB on disk: ~60 KiB token bytes (256 × 1-byte single-byte tokens + ~3840 × ~16-byte merge tokens, per paper §3.6) + 16 KiB offsets (4097 × u32). For 16-bit / 65K tokens, the OnPair paper Table 4 reports ~0.7 MiB total per dict.

The optional `splits` buffer is described in the [GPU decoder](#gpu-decoder-cuda-hopper-class-and-newer) section.

**Tier 2 — `OnPair16Layout` (analog of `DictLayout`).** A layout with two children:

```rust
pub struct OnPair16Layout {
    /// The dictionary; typically Arc-shared across many sibling OnPair16Layouts
    /// in a ChunkedLayout, exactly like DictLayout's `values`. Concretely a
    /// BinaryView of n_tokens entries (one per token), allowing the compressed-
    /// domain predicate machinery to query the dictionary directly via Vortex's
    /// existing BinaryView compute kernels.
    dict: LayoutRef,
    /// Codes + per-string offsets + optional splits for this chunk.
    data: LayoutRef,
}
```

Sharing across chunks is identical in shape to `DictLayout`: the writer trains one dict on the corpus, emits one `dict` `LayoutRef`, and every chunk's `OnPair16Layout` Arc-shares that same reference. The reader uses an `OnceLock`-cached materialized dictionary view — the dict array is small enough (≤1 MiB at 16-bit codes) to stay resident in L2/L3 between calls, amortized over the millions of strings the dict applies to.

This Tier-2 mode is what makes corpus-wide dictionary training viable without inventing any new file-format machinery; it also lets the encoder ingest an externally-trained dictionary (whatever the source) and apply it uniformly across all chunks.

**Why BinaryView for the shared `dict` LayoutRef:** the `DictLayout` pattern stores values as a typed array because predicate pushdown operates on typed values. The OnPair16 compressed-domain automata operate on token byte sequences; BinaryView is the natural typed representation and lets the automata be implemented as standard Vortex compute kernels over a BinaryView column. A plain `Buffer` would be more compact but would lose this query-time integration.

### `code_width_bits`: the load-bearing knob

The OnPair paper's natural recommendation, from Section 3.6 ("Dictionary Size Trade-Offs"), is *"across our experiments, we found that allocating 16 bits per token strikes a practical and robust compromise"*; they also note that *"both compression and decompression speeds degrade significantly beyond 16 bits."* The 16-bit width is the paper-recommended default *on a CPU-only deployment*.

This RFC departs from that recommendation in one direction: **on Hopper-class GPUs and below, the dictionary doesn't fit in shared memory at 16 bits.** The flattened decode table at 65,536 tokens × ~17 bytes/entry (16-byte symbol + 1-byte length) is ~1 MiB, which exceeds every current GPU's per-SM SMEM budget. Forcing 16 bits would either trap GPU decoding in the slower dict-in-global-memory mode (Mode B below; ~150–200 GB/s on H100 vs ~250 GB/s expected for SMEM-resident) or give up GPU acceleration entirely. The CPU-only Stage 1 of this RFC adopts the paper's 16-bit recommendation; Stage 2 adds the parametric width with a 12-bit default that is GPU-friendly.

The bitstream shape is identical at every width — only the dictionary capacity and the code-stream packing change. Recommended values:

| `code_width_bits` | Dict capacity | Decode-time table footprint (16-byte stride + length) | Recommended for |
|---|---|---|---|
| **12 (default, Stages 2+)** | **4,096** | **~68 KiB** | **Mixed CPU/GPU; SMEM-resident decode on Hopper-class GPUs.** |
| **16 (Stage 1 default)** | **65,536** | **~1 MiB** | **CPU-only; matches the OnPair paper recommendation.** |
| 14 | 16,384 | ~272 KiB | CPU-primary; L2-friendly on GPU |
| 11 | 2,048 | ~34 KiB | Ada-class GPUs (RTX 4090, L40) at higher occupancy |
| 10 | 1,024 | ~17 KiB | Extreme GPU occupancy |

Note that this is the **decode-time** table footprint (16 byte slots padded for over-copy + 1 byte length per entry). The **on-disk** footprint is OnPair's flat-bytes + offsets format, which at 4K tokens is ~76 KiB (~60 KiB byte data per paper §3.6 + 16 KiB offsets), not ~68 KiB. The disk and decode-time footprints are within ~10% of each other; the decode-time figure is what gates GPU SMEM occupancy.

CPU-only deployments can opt up to 14- or 16-bit codes and get a tighter dictionary; GPU-only deployments on narrower hardware can opt down to 11- or 10-bit codes for higher occupancy. The encoder picks the width once per array (or per corpus, in Tier 2); the decoder dispatches on it.

### Training algorithm

OnPair16's published training algorithm is used unchanged at every code width. At the default 12-bit width, the dictionary fills before the training loop exhausts the input sample; only the most-frequent merges win slots.

**Why not FSST12's own training at the same 4K-symbol budget?** The FSST paper directly addresses this in Section 4.1 ("The Dependency Issue"). Boncz et al. tried both a single-pass suffix-array gain-based selection ("the first symbol picked will indeed have the highest compression gain. However, the compression gain of subsequent symbols depends on earlier symbols. Correcting for the dependencies on earlier symbols is very difficult and, depending on how it is done, leads to large over- or underestimates") and a single-pass dynamic-programming approach ("the compression factor only marginally improves, while encoding performance is severely affected"). They settled on a 5-round evolutionary algorithm that re-tokenizes the sample with the current dict each round and recomputes gains accordingly.

OnPair's pair-merging sidesteps the dependency issue by a different mechanism: it doesn't *select* symbols by gain at all — it *merges* the most-frequent adjacent token pair and immediately replaces it everywhere, so subsequent pair-counting is already conditioned on the new token. The dependency between symbols is resolved incrementally rather than estimated. Whether this resolves the dependency issue *better* than FSST's 5-round approach at the same 4K-symbol budget is an empirical question — the validation campaign in §[Validation](#validation-campaign) tests it explicitly (sub-benchmark 2). The RFC's prior is that OnPair-style pair-merging is competitive or better at 4K codes because (a) the merging avoids FSST's "longest-prefix-first ordering" workaround (FSST paper §4.3: *"we store the (real) symbols in lexicographical order, but when one string prefixes the other, the longest is first"* — a constraint OnPair doesn't impose, expanding the design space), and (b) OnPair training is single-pass and so spends compute on more samples rather than re-tokenizing fewer samples five times.

**A note on speed.** The OnPair paper Table 5 reports training as a small fraction of total compression time (parsing dominates). At 4K codes the dictionary fills earlier, so training cost should be ≤ the paper's numbers; specifically the URL benchmark (paper's slowest) gets faster at smaller dict caps because the training loop terminates earlier.

### CPU decoder

OnPair16's published decode path. The on-disk dictionary is loaded directly; no symbol-table reconstruction phase is required. For a token `t`:

```c
// dict_offsets is u32[n_tokens + 1]; dict_bytes is the padded flat byte buffer
const uint8_t  *src = dict_bytes + dict_offsets[t];
const uint32_t  len = dict_offsets[t+1] - dict_offsets[t];
memcpy(out, src, 16);                  // unconditional 16-byte over-copy (safe due to padding)
out += len;                            // advance by actual length
```

This is OnPair16's Algorithm 3 (paper §3.5, Algorithm 3) applied to every code in the input stream. At 12-bit codes the code-stream is read in the FSST12 layout (2 codes per 3 bytes); at 16-bit codes it's plain `u16[]`; at 10/11/14-bit codes it's LSB-first bit-packed.

The OnPair paper reports CPU decode throughput on this design at **6.5–7.8 GB/s** across five workloads (Book Reviews, Book Titles, News Headlines, Tweets, URLs; Table 3, Intel Core Ultra 7 265K, AVX2 only — the hardware doesn't have AVX-512). FSST CPU decode on the same hardware is 4964–5683 MiB/s, so OnPair16 is comparable to or faster than FSST without AVX-512.

**SIMD acceleration.** The OnPair reference implementation ([`include/onpair/decoding/detail/decode_all.h`](https://github.com/gargiulofrancesco/onpair_cpp/blob/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/decoding/detail/decode_all.h)) uses compiler auto-vectorized `memcpy` (the unconditional 16-byte copy maps directly onto a single vector load/store on any architecture with ≥16-byte vector width). Explicit AVX-512 / AVX2 / NEON paths are an implementation detail and not part of the format; the OnPair paper §3.2.2 explicitly notes that AVX-512 can copy up to 64 bytes per instruction, and an AVX-512 implementation would amortize the 16-byte copy across multiple tokens per vector store. The format guarantees scalar decodability at any code width.

**Per-string random access.** `code_offsets[i]..code_offsets[i+1]` gives the slice of codes for string `i`; the per-token decode above is byte-by-byte position-independent.

### GPU decoder (CUDA, Hopper-class and newer)

Primary GPU targets are Hopper (H100, H200), Blackwell (B100, B200), and Ada (RTX 4090, L40). Ampere (A100) is supported but not the design point; older GPUs are not in scope.

The decode-time table is two arrays — a 16-byte-padded symbol slot per code plus a 1-byte length per code — totalling ~17 bytes per entry. Per-SM shared-memory capacities (max user-shared carveout, per NVIDIA tuning guides):

| GPU | Generation | SMEM/SM | Achievable blocks/SM for the flattened table |
|-----|------------|---------|----------------------------------------------|
| | | | 1K dict (~17 KiB) / 2K (~34 KiB) / 4K (~68 KiB) / 8K (~136 KiB) / 16K (~272 KiB) |
| **H100/H200** | Hopper (primary) | 228 KB | ≥8 / 6 / **3** / 1 / does not fit |
| **B100/B200** | Blackwell (primary) | 228 KB+ | ≥8 / 6 / **3** / 1 / does not fit |
| **RTX 4090 / L40** | Ada (primary) | 100 KB | 5 / 2 / 1 / does not fit / does not fit |
| A100 (comparison) | Ampere | 164 KB | ≥8 / 4 / 2 / 1 / does not fit |

(Bolded entries are the recommended `code_width_bits = 12` row.)

This is the structural reason GSST hits 191 GB/s with FSST8's ~2 KiB symbol table: at 2 KiB, the dict is essentially free in SMEM, so the GPU isn't trading occupancy for table size. And it's the reason 12-bit is the GPU sweet spot for Hopper-class hardware — 4K × ~17 bytes lands at 3 blocks/SM on H100, decent occupancy. The 16-bit (64K-entry, ~1 MiB) flattened table does not fit in any current GPU's SMEM and forces the slower dict-in-global-memory mode.

The decode-time table is reconstructed at kernel start by gathering bytes from `dict_bytes` via `dict_offsets`. For a 4K dict this is 4K gather operations, each ≤16 bytes; the cost is O(n_tokens × MAX_SYMBOL_LEN) ≈ 64 KB of memory traffic per kernel launch — negligible relative to a chunk-sized decode.

**Adopting GSST's split parallelism format.** The structural insight from GSST (Vonk thesis §4.2.2) is that the writer stores per-split metadata (uncompressed size of each split, where a split is a contiguous run of compressed codes) in the block header, so multiple GPU threads in a single SM can begin decoding at known output offsets without serial dependency on prior splits. The thesis evaluates two split designs:

- *Constant uncompressed size* (Figure 4.2a) — each split outputs the same number of bytes; the block header stores the *number of codes* per split (variable). Pros: uniform write footprint per thread. Cons: variable codes per split means thread-time divergence.
- *Constant number of codes* (Figure 4.2b) — each split consumes the same number of input codes; the block header stores the *uncompressed size* per split (variable). Pros: uniform input footprint per thread, simpler bit-unpacking. Cons: variable output sizes.

GSST chose **constant number of codes** for its implementation (thesis §4.2.2: "decompression algorithms are expected to write more data than they read, which is why it's more important to balance the write operations than the read operations"). The OnPair16 encoding adopts the same choice for compatibility with the GSST kernel shape. The optional Coalesced Memory Access Format (thesis §4.2.3) is described separately below.

#### Split-size analysis

This section is a prior, not a measurement; the validation campaign in §[Validation](#validation-campaign) tests its conclusions explicitly.

The competing pressures are amortizing per-split overhead vs. extracting enough parallelism to fill the GPU. The per-split overhead is one global-memory read of the split's uncompressed size plus implicit input/output offsets (~16 bytes total; L2-cached after warmup) plus a few cycles of bookkeeping — call it ~30 ns minimum. The per-warp-iteration work, at 12-bit codes with 128 codes/iter (4 codes per thread, 256 bytes coalesced input), is **estimated** at ~25–40 ns when input/output bandwidth and SMEM dict lookups are pipelined behind enough warp parallelism per SM. This estimate is a rough instruction-mix accounting on H100 priors; the real number will be measured in the validation campaign and is a load-bearing input to the table below. If the real number is materially different (say, 100 ns), the recommended split size changes.

Under those priors, warp-per-split setup amortization:

| Split size (codes) at 12-bit | Warp iters | Estimated work | Setup | Setup % |
|---|---|---|---|---|
| 256 | 2 | ~60 ns | ~30 ns | **33% — bad** |
| 512 | 4 | ~120 ns | ~30 ns | **20% — marginal** |
| 1024 | 8 | ~240 ns | ~30 ns | **11% — OK** |
| 2048 | 16 | ~480 ns | ~30 ns | **6% — good** |
| 4096 | 32 | ~960 ns | ~30 ns | **3% — great** |

So under the prior, warp-per-split at 12-bit codes wants `codes_per_split ≥ ~1024`, ideally 2K–4K. The recommended default `codes_per_split = 1024`.

**Two kernel-mapping options.** Vonk's thesis (§4.2.2) recommends sizing the thread block to **≥128 threads, each thread owning one split** (matching the SM's core count on modern NVIDIA GPUs); thesis §3.3 motivates this by composability with the broader ingestion pipeline (chunked host→device transfers overlapped with kernel execution; persistent kernels make this harder because they claim SMs for their full lifetime). For Vortex's likely decode shapes (a single chunk decoded all at once, or many chunks across SMs), this is the right reference variant and should ship first.

A **persistent-thread alternative** — launch ~512 warps once; each warp atomicAdds a global counter to claim its next split — is plausibly competitive when a single very-large chunk is decoded in isolation (the block-per-SM mapping under-utilizes SMs at low chunk counts; persistent threads load-balance automatically). The cost is more complex kernel code, atomic contention on the counter, and harder profiling. The validation sub-benchmark in §[Validation](#validation-campaign) measures both.

**Assumed Vortex chunk size.** This analysis assumes a typical Vortex chunk of ~500K codes (~1.5 MB compressed at 12 bits + ~4 MB uncompressed at 4-byte avg symbol expansion). Smaller chunks (≤64K codes) saturate fewer SMs and may favor either many-chunks-in-flight (block-per-SM) or persistent threads with very small `codes_per_split`. Vortex chunk size is configurable per array; the encoder picks `codes_per_split` accordingly.

#### Coalesced Memory Access Format (optional)

GSST's third format optimization (thesis §4.2.3, Figure 4.4) reorders codes within a compressed block so that the *i*-th code of every split is stored contiguously, then the (*i*+1)-th, and so on. With this ordering, when *N* threads each read their *i*-th code in parallel, the reads coalesce into one or two cache lines instead of *N* scattered ones. The thesis reports this reordering is materially helpful on top of the split format.

Whether it helps on OnPair16 codes at 12-bit width and 1024 codes/split is genuinely uncertain — GSST evaluated it on FSST8 with a much smaller dict and a different bit-packing layout. The validation campaign measures whether the reordering pays off on OnPair16 in our setting; until then, the format reserves a `flags` bit (bit 0) for opt-in coalesced-format arrays. **Default off until measured.** A small CPU-decode penalty applies when reading coalesced-format codes scattered, though prefetching helps; so even after validation, the reordering is plausibly best as a GPU-target opt-in rather than a universal default.

#### Mode A — dict-in-shared-memory (`code_width_bits` ≤ 12), GSST one-block-per-SM mapping (reference)

```cuda
__global__ void decode_kernel_smem_blockwise(
    /* per-chunk constants */
    const uint8_t*  dict_bytes,             // padded flat byte buffer
    const uint32_t* dict_offsets,           // [n_tokens + 1]
    uint32_t        n_tokens,
    /* per-chunk variables */
    const uint8_t*  codes,                  // bit-packed code stream
    const uint32_t* split_uncompressed_sizes, // [n_splits]
    uint32_t        n_splits,
    uint32_t        codes_per_split,
    uint8_t*        out)
{
    __shared__ uint8_t  sym_bytes[4096 * 16];   // 64 KiB (or less for smaller widths)
    __shared__ uint8_t  sym_len  [4096];        //  4 KiB
    __shared__ uint32_t output_offsets[N_SPLITS_PER_BLOCK + 1];

    // Phase 1: cooperative dictionary materialization.
    //   - Threads in the block cooperatively memcpy dict_bytes/dict_offsets into SMEM.
    //   - Equivalent to GSST thesis §4.3.1; the dict bytes are over-copied 16 bytes per
    //     entry into a fixed-stride buffer for cache-friendly access.
    //   - Per-split output offsets are precomputed once into shared memory via a
    //     block-wide CUB::ExclusiveScan over split_uncompressed_sizes[].
    __syncthreads();

    // Phase 2: each thread owns one split. Per GSST §4.2.2, this is the canonical mapping.
    //   - thread_id selects split_id within this block's range.
    //   - Input range:  codes[split_id * codes_per_split * code_width_bits / 8 ..]
    //   - Output start: output_offsets[split_id]   (precomputed)
    //   - Walk the split's codes_per_split codes; for each:
    //       - Unpack 12-bit code from the bit-packed stream.
    //       - Load sym_bytes[16 * code] (vectorized).
    //       - memcpy 16 bytes to out[output_offsets[split_id] + local_off].
    //       - local_off += sym_len[code].
}
```

The persistent-thread alternative differs in two places: the grid launch is `~512` warps once (instead of one block per chunk × many SMs), and each warp claims its split via `atomicAdd(g_split_counter)` instead of computing it from `blockIdx + threadIdx`.

**Expected throughput on H100 at 12-bit codes:** GSST measures 191 GB/s on A100 with a 2 KiB FSST8 symbol table. The OnPair16-on-H100 prediction (~250 GB/s) extrapolates from GSST's A100 result by (a) Hopper's higher HBM bandwidth (~3 TB/s vs ~1.5 TB/s) and (b) lack of escape-code divergence, offset by (c) lower occupancy from the 68 KiB SMEM footprint vs GSST's effectively-free 2 KiB and (d) 12-bit bit-unpacking cost that FSST8 doesn't pay. Net effect is uncertain — the 250 GB/s figure is a prior, not a derivation. Realistic range is ~150–300 GB/s; the validation campaign measures.

**Mode B — dict-in-global-memory (`code_width_bits` ∈ {14, 16}).** A separate kernel for when the encoder chose a larger dict for ratio reasons but GPU decode is still wanted. A pre-pass kernel materializes the padded flattened table into global memory once per chunk-group; subsequent decode kernels read it through L2 (50 MB on H100 per the Hopper tuning guide; the ~1 MiB table fits trivially with high hit rate). The lane-level lookup uses `__ldcg` for a cache-global load. Expected throughput: ~150–200 GB/s on H100 — slower than Mode A but still ~30–50× the CPU.

**CUDA-specific risks worth flagging in implementation:**
- *Split metadata is required for GPU mode.* The writer must emit the `splits` buffer (per-split uncompressed sizes; ~4 bytes per split) for the GPU decoder to use the split parallelism format. At `codes_per_split = 1024`, metadata overhead is well under 1% of the compressed bytes. Tier-1 arrays without `splits` fall back to single-thread-per-string CPU-style decode on GPU — much slower.
- *Bank conflicts on dictionary access.* With 16-byte-wide entries, an unlucky access pattern can serialize. Standard fix is a one-word stride or interleaving the low bits of the code with the bank index; CUB scan helpers handle this correctly.
- *Occupancy vs. SMEM tradeoff.* Mode A at 12-bit codes uses ~68 KiB SMEM/block; on H100 this gives ~3 blocks/SM. Dropping to 10-bit codes (~17 KiB SMEM/block) raises occupancy substantially, but the ratio loss is real and must be measured.
- *Per-string boundaries are NOT in the GPU bulk hot path.* They live in a separate buffer that bulk-decode kernels skip entirely; only the random-access path reads them. The GPU bulk-decode path uses `splits`, which is independent of string boundaries.

### Per-string random access and predicate pushdown

**Random access** is direct from the format: `code_offsets[i]..code_offsets[i+1]` gives the codes for string `i`; the dictionary is loaded once and cached. For very-short-string workloads, decode latency is dominated by dictionary materialization rather than the codes themselves; the `OnceLock` cache (Tier 2) or an in-decoder cache (Tier 1) is essential.

**Predicate pushdown.** OnPair16 ships a published compressed-domain predicate machinery in `onpair_cpp` ([`include/onpair/search/automata/`](https://github.com/gargiulofrancesco/onpair_cpp/tree/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/search/automata)). It implements:

- Substring match (`LIKE '%needle%'`) via the Aho–Corasick automaton ([`aho_corasick_automaton.h`](https://github.com/gargiulofrancesco/onpair_cpp/blob/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/search/automata/aho_corasick_automaton.h)) and Knuth–Morris–Pratt ([`kmp_automaton.h`](https://github.com/gargiulofrancesco/onpair_cpp/blob/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/search/automata/kmp_automaton.h)).
- Prefix match (`LIKE 'needle%'`) via the prefix automaton over the lexicographically-sorted dictionary ([`prefix_automaton.h`](https://github.com/gargiulofrancesco/onpair_cpp/blob/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/search/automata/prefix_automaton.h)).
- Equality (`WHERE col = 'value'`) via a hash-keyed automaton ([`eq_automaton.h`](https://github.com/gargiulofrancesco/onpair_cpp/blob/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/search/automata/eq_automaton.h)).
- Multi-pattern via Aho–Corasick.
- Boolean composition (`NOT`, `AND`, `OR`) over any of the above.

All automata operate on the token-id stream, not the decoded byte stream. Each query compiles its automaton once against the column's dictionary (exploiting the lexicographic ordering for prefix-range computation) and then scans an arbitrary number of rows without re-tokenization. Vortex's integration is to adopt these as compute kernels — the same role its existing FSST DFA plays for FSST arrays — rather than reimplementing the technique from scratch. Vortex's existing FSST LIKE-pushdown DFA generalizes the "string matching as future work" idea from FSST paper §3.2 (Boncz et al. did not ship this in the original paper); the OnPair16 path is shorter because the upstream automata library is already written.

**Out of scope for this RFC**: Vortex's compute kernel surface (filter/scalar_at/take) and how the OnPair16 automata bind into it. That's an integration discussion to resolve during implementation, not a format-level question.

### Compression

OnPair16's single-pass LPM training; the on-disk format inherits the upstream wire format directly. Compression speed (paper Table 3, single-core AVX2, no AVX-512) is **137–229 MiB/s** for OnPair16 across the five public workloads, vs **325–504 MiB/s** for FSST AVX2 — about ½× FSST AVX2. The paper's hardware doesn't have AVX-512, so the OnPair16 numbers are "no AVX-512 yet" rather than a fundamental ceiling; the FSST AVX-512 path reports ~2 GB/s (FSST paper §3.2). An AVX-512-vectorized OnPair16 should close most of the 2× gap to FSST at parity — the hot path is longest-prefix matching against a hash map plus bytewise copies, both of which have known SIMD treatments (the OnPair paper §3.2.2 explicitly notes AVX-512's 64-bytes-per-instruction copy capability). The engineering isn't done in the public reference impl yet.

**A note on the apples-to-apples comparison: FSST12 vs OnPair-at-12-bit.** The numbers cited above are FSST8 vs OnPair16. At the 12-bit default, both encodings have a 4K-token dictionary, which collapses two of the three structural reasons OnPair16 is slower than FSST8: (a) larger dictionary → more LPM probe cost (collapses at parity) and (b) longer average symbol length → more bytes touched per match (largely collapses; OnPair at 4K tokens has avg symbol ~3 bytes per paper Table 1, and FSST12's average is plausibly similar though not published). The remaining factor is (c) OnPair's heavier per-probe data structure (`robin_hood` hash for ≤8-byte patterns, bucketed 8-byte-prefix-keyed structure for longer patterns, per paper §3.4) vs FSST's lossy perfect hash on the 3-byte prefix (FSST paper §4.3) — a deliberate ratio-for-speed trade in OnPair's design. There is no published FSST12-vs-OnPair-at-12-bit head-to-head measurement in the public literature; the validation campaign sub-benchmark 9 measures it directly. Our prior is that OnPair-at-12-bit is **~50–60% of FSST12's compression speed at AVX2 parity**, a meaningfully smaller gap than the OnPair16-vs-FSST8 ratio but still measurable. With AVX-512 on both, the gap narrows further; with AVX-512 on neither (the public state today), the gap is largest. Compression throughput is parsing-dominated for typical-sized inputs (OnPair paper Table 5: training is <10% of total compression time on most workloads); the smaller training cost of OnPair's single pass vs FSST's 5-round evolutionary algorithm wins meaningfully only on very small inputs where training fixed cost matters.

Note that FSST12 was added to the `cwida/fsst` source tree as an experimental variant after the original FSST paper landed; its 5-round training was tuned for FSST8's 255-code budget, and (to our knowledge) nobody has re-tuned the round count, sampling fraction, or candidate-selection heuristic for the 4096-code budget. Sub-benchmark 2 should be careful to either use the FSST12 source as-is (apples-to-apples vs the shipped artifact) or document any re-tuning, since "OnPair training beats FSST12 training" against an untuned FSST12 is a weaker claim than against a retuned FSST12.

In Tier-2 (shared-dict) mode, training is a one-time corpus-wide cost amortized across all chunks; per-chunk compression is then a parsing pass (which dominates OnPair16's total compression time per paper Table 5) and should run ~2–5× faster than per-chunk training.

### Why this is the right design

A reasonable reader will ask: why a new array? Why not just extend `FSSTArray`, or just adopt OnPair as-is, or just port GSST?

- **Why a cousin Array rather than a flag on `FSSTArray`?** The two encodings differ in their codebook representation (FSST stores u64 symbols + u8 lengths; OnPair stores flat lex-ordered bytes + Arrow-style offsets) and in their decode hot loops. Folding them into one Array with a discriminant flag conflates two genuinely different layouts in metadata and forces every reader to handle both. Cleaner to express them as two Array types in the same family, sharing the FSST archetype but choosing different codebook construction strategies. This mirrors `vortex-fastlanes`'s pattern.
- **Why not adopt OnPair16 verbatim (no Vortex extensions)?** OnPair16's paper-recommended 16-bit default dictionary doesn't fit in any current GPU's SMEM. Either the GPU decoder runs in the slower global-memory mode, or the format carries a width knob so deployments can pick the GPU-friendly point. The latter is strictly more flexible — the 16-bit width of this RFC is functionally identical to upstream OnPair16, so callers who don't want GPU acceleration get the upstream behavior unchanged. (This is Stage 1 of the rollout.)
- **Why not stop at FSST12?** FSST12's bitstream layout is good and we adopt it at the 12-bit width. FSST12's *training* (FSST's 5-round evolutionary algorithm; FSST paper §4) is the part we replace. The training algorithm was tuned for FSST8's 255 codes; at 4096 codes, OnPair's incremental pair-merging plausibly outperforms FSST's gain-estimation-and-reorder heuristic for the reasons in [Training](#training-algorithm). This is the load-bearing empirical claim of this RFC and the validation campaign measures it directly (sub-benchmark 2). If OnPair's training *doesn't* win at 4096 codes, the right answer is to ship FSST12 instead — strictly a smaller change.
- **Why not GSST alone?** GSST is a decoder over the same FSST8 bitstream — same compression ratio as FSST8, same escape mechanism. This proposal gets GSST's kernel structure with an escape-free bitstream and a better codebook underneath.
- **Why both tiers?** Tier 1 (`OnPair16Array`) covers the common case — per-array independence, the same deployment shape as `FSSTArray`. Tier 2 (`OnPair16Layout`) covers the corpus-wide-dictionary case, which gives non-trivial ratio improvements on cross-chunk-redundant data. Vortex already has this two-tier pattern for raw values (Array encoding + `DictLayout`); the FSST family is structurally well-suited to mirror it (the dictionary is small enough to Arc-share cheaply, the codes are independent per chunk). Tier 2 ships in Stage 3.

### Edge cases

- **Empty strings.** `code_offsets[i] == code_offsets[i+1]`. Decoder writes zero bytes. Trivially handled.
- **Null strings.** Handled by Vortex's standard validity bitmap at the Array level; the OnPair16 encoding never sees a null. The encoding's per-string offsets index into the non-null subset.
- **Zero rows.** `n_strings == 0`, `n_codes == 0`. The dictionary may still be present (Tier 2) or absent (Tier 1 — encoder may emit an empty dict trivially). Decoder is a no-op.
- **Very small arrays** (e.g. ≤16 strings, ≤64 codes). Dictionary materialization dominates decode time. The recommended fallback: for arrays below a writer-configurable threshold (default ~256 strings or ~1 KiB compressed), the encoder falls back to FSST or to plain (unencoded) bytes — OnPair16's ratio advantage is lost on tiny arrays and the materialization overhead is wasted. This is an encoder heuristic, not a format requirement.
- **Extreme low entropy** (e.g., every string is the same value). OnPair16's training produces a single learned token for the repeated value; compression ratio is excellent. No special-case logic needed.
- **Mismatch between Tier-2 dict and a chunk's data.** OnPair16 always covers all 256 bytes (codes 0–255 are reserved single-byte tokens), so any byte sequence is representable. A poorly-matched dict produces a longer code stream; compression ratio degrades but the format remains well-defined.
- **Single very long string** (e.g., a 10 MB blob in one row). `code_offsets[1] - code_offsets[0]` is the whole code stream; decode is one big sequential operation. The split-parallel GPU decoder treats the codes the same way regardless of how they're partitioned into strings.

### Hidden assumptions

The design rests on a small set of assumptions that aren't proven elsewhere in the doc; if any breaks, the recommendation set changes.

- **GPU decompression is a Vortex priority within the RFC's time horizon.** Stages 2 and 3 are GPU-focused; if Vortex's roadmap defers GPU decode indefinitely, Stage 1 is the only stage that needs to land and the RFC's "GPU-friendly" framing is overstated for that timeline. The author's understanding from outside-the-RFC context is that GPU is on the roadmap; readers from the broader Vortex maintainer set should sanity-check this is shared.
- **Typical Vortex chunk size is ~500K codes (~1.5 MB compressed at 12-bit).** Stated in §[Split-size analysis]; the GPU saturation argument depends on it. Smaller chunks (≤64K codes) saturate fewer SMs and may favor different `codes_per_split` defaults or persistent-thread mappings.
- **OnPair's reference implementation will remain available and reasonably stable across the validation timeframe.** The README pins to pre-1.0 status. Format and API drift in upstream OnPair could complicate interop testing (validation sub-benchmark #3, two-way decoder cross-check). Mitigation: the RFC's wire format is pinned to commit `ae590713`; any subsequent upstream change is a coordinated bump, not an automatic adoption. Plus implementer can reimplement entirely if upstream becomes unworkable (see Drawbacks).
- **The audience for OnPair16's CPU decode performance values AVX2 results.** OnPair paper Table 3 is AVX2-only. Many production deployments have AVX-512 available; for those, the apples-to-apples comparison is FSST AVX-512 vs an as-yet-unwritten OnPair16 AVX-512 path (§[Compression](#compression) discusses). If AVX-512-only deployments dominate, the case for OnPair16-vs-FSST on CPU compression-side weakens until the AVX-512 OnPair16 path is written.

### Staging

The RFC proposes a three-stage rollout. Each stage is independently mergeable and validatable; later stages depend on earlier stages landing but the design of earlier stages stands on its own if later stages are deferred.

**Stage 1 — `OnPair16Array` at 16-bit codes, CPU-only.** No `splits` buffer, no GPU kernel, no Tier 2. This is essentially a port of upstream OnPair16 into Vortex with the standard `Array`/`Metadata`/compute integration. It validates that OnPair16 beats FSST on CPU decode and matches FSST on compression speed, exercises the writer + reader plumbing, and gives the Vortex maintainers production data on the encoding before the harder pieces ship. The 16-bit width matches the OnPair paper's recommended default.

**Stage 2 — Parametric `code_width_bits` + the `splits` buffer + Mode A GPU kernel.** Adds the 10/11/12/14-bit widths to the encoder, the optional `splits` buffer, and the CUDA decoder (GSST one-block-per-SM reference variant). This is where the novel design lives. Validates the split-parallel GPU decode and the 12-bit-vs-16-bit ratio/throughput tradeoff.

**Stage 3 — `OnPair16Layout` (Tier 2 shared dict) + Mode B + Coalesced Memory Access Format.** Adds the cross-chunk-shared-dictionary layout, the 14/16-bit GPU decoder kernel, and the optional code-stream reordering for GPU memory coalescing. Validates the corpus-wide-dict ratio uplift and the Mode A/B crossover.

A monolithic merge of all three is also possible; the staged form is what we recommend.

### Diagrams

**On-disk format of an `OnPair16Array` (Tier 1):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ metadata (Prost)                                                            │
│   code_width_bits, n_tokens, n_strings, n_codes, n_splits, codes_per_split, │
│   uncompressed_bytes, dict_bytes_size, flags                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ buffer [0] dict_bytes : u8[dict_bytes_size + 16]                            │
│                          ┌────┬────┬────┬───────┬───────┬─────────┬────────┐│
│                          │ 0  │ 1  │... │ token │ token │  ...    │padding ││
│                          │byte│byte│    │  256  │  257  │         │ (16B)  ││
│                          └────┴────┴────┴───────┴───────┴─────────┴────────┘│
│                              ↑ lexicographically sorted by byte sequence    │
├─────────────────────────────────────────────────────────────────────────────┤
│ buffer [1] dict_offsets : u32[n_tokens + 1]                                 │
│   dict_offsets[t]..dict_offsets[t+1] = byte range of token t in dict_bytes  │
├─────────────────────────────────────────────────────────────────────────────┤
│ buffer [2] code_offsets : u32[n_strings + 1]                                │
│   code_offsets[i]..code_offsets[i+1] = code range of string i in `codes`    │
├─────────────────────────────────────────────────────────────────────────────┤
│ buffer [3] codes : bit-packed, LSB-first, code_width_bits per code          │
├─────────────────────────────────────────────────────────────────────────────┤
│ buffer [4] splits (OPTIONAL) : u32[n_splits]                                │
│   splits[s] = uncompressed bytes produced by split s of codes_per_split     │
│                consecutive codes.                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**GPU split-parallel decode (Mode A, one block per chunk):**

```
                    GPU SM (1 block, ~128 threads cooperating)
   ┌──────────────────────────────────────────────────────────────┐
   │  shared memory:                                              │
   │    ┌──────────────────┐    ┌──────────────┐                  │
   │    │ sym_bytes[N×16]  │    │ sym_len[N]   │                  │
   │    │ (cooperatively   │    │ (cooperatively                  │
   │    │  loaded from     │    │  loaded)     │                  │
   │    │  dict_bytes via  │    │              │                  │
   │    │  dict_offsets)   │    │              │                  │
   │    └──────────────────┘    └──────────────┘                  │
   │    ┌─────────────────────────────────────────┐               │
   │    │ output_offsets[~128] = exclusive_scan   │               │
   │    │   over splits[...] (per-split outputs)  │               │
   │    └─────────────────────────────────────────┘               │
   │                                                              │
   │  threads:                                                    │
   │    thread 0 → split 0 → codes[0..codes_per_split]            │
   │             → out[output_offsets[0]..]                       │
   │    thread 1 → split 1 → codes[codes_per_split..2*]           │
   │             → out[output_offsets[1]..]                       │
   │    ...                                                       │
   │    thread T-1 → split T-1                                    │
   └──────────────────────────────────────────────────────────────┘
                              ↓
   Global memory:                                                
     ┌──────────────────────────────────────────┐                
     │ out[0 .. uncompressed_bytes-1]           │                
     │  (decompressed bytes, coalesced writes)  │                
     └──────────────────────────────────────────┘                
```

## Compatibility

This is a new pair of types in the existing FSST encoding crate, not a change to existing arrays or layouts. The `FSSTArray` and `DictLayout` wire formats and APIs are untouched. The encoding registry gains two new entries — `OnPair16` (Array) and `OnPair16Layout` (Layout) — and readers without OnPair16 support will be unable to read arrays/files written with it. `FSSTArray` remains the recommended default for string compression until `OnPair16Array` has accumulated production miles; both ship from the same crate.

There are no migrations required. Files using `FSSTArray` stay using `FSSTArray`; files using `OnPair16Array` are a deliberate per-writer choice.

The format-stability commitments for the OnPair16 encoding:

- The metadata schema is Prost-encoded and follows Vortex's existing convention for forward-compatible additions (optional fields with explicit defaults).
- `code_width_bits` is part of the format and must be honored exactly by any conforming decoder. Adding new allowed values in the future (e.g., 13-bit, 15-bit) is a forward-compatible addition only if older readers reject the unknown width cleanly rather than misinterpreting it.
- The `dict_offsets` array is `u32[n_tokens + 1]` and the `dict_bytes` buffer must end with at least 16 zero-padding bytes (so unconditional 16-byte over-copies at any valid token offset stay in bounds). These are wire-format invariants.
- `flags` bit 0 indicates the codes buffer is in coalesced-memory-access order (GSST §4.2.3). Future flag bits are reserved.
- **Upstream OnPair stability caveat.** The reference implementation at `gargiulofrancesco/onpair_cpp@ae590713` is pre-1.0; its README explicitly states the on-disk format may change. This RFC's wire format is pinned to the commit cited; any subsequent format change in upstream OnPair is a coordinated upstream-and-Vortex bump, not an automatic adoption.

## Drawbacks

- **Decoder dictionary footprint.** ~76 KiB on-disk for 4K-token dicts; up to ~1 MiB for 64K-token dicts. Per-thread decoder scratch is similar. Tier 2 amortizes across all chunks sharing the dict.
- **CUDA-only GPU.** This RFC scopes GPU decode to NVIDIA. Targeting AMD (ROCm/HIP) or vendor-neutral compute (SYCL, Vulkan) is out of scope — see Future Possibilities.
- **Compression speed gap to FSST.** Reference OnPair16 is ~½× FSST8's AVX2-only compression speed (137–229 MiB/s vs 325–504 MiB/s) and ~⅙× FSST8's AVX-512 compression speed (~150–230 MiB/s vs ~1–3 GB/s). At 12-bit parity (this RFC's default vs FSST12), our prior is a smaller gap — OnPair-at-12-bit ≈ 50–60% of FSST12's speed at AVX2 parity (no public head-to-head exists yet; validation sub-benchmark 9 measures). The structural reasons after dict-size and symbol-length collapse at parity: OnPair's `robin_hood`-style hash + bucketed long-pattern structure is heavier per LPM probe than FSST's lossy perfect hash. An AVX-512 OnPair16 path is plausible and would close most of the residual gap; that engineering work isn't done in the public reference impl.
- **Tier-1 metadata is larger than FSST8's.** The OnPair16 dictionary (a few tens of KiB to 1 MiB) is larger than FSST8's ~2 KiB. Negligible at array-scale but worth noting for very-small arrays. Tier 2 makes this a one-time corpus-level cost.
- **Predicate pushdown precompute is larger than FSST8's.** OnPair16's compressed-domain automata operate on a larger dict and use specialized data structures (Aho–Corasick trie, KMP failure tables, prefix range lookups). Precompute is O(pattern_length + dict_size) per query; for short patterns and 4K-token dicts this is sub-millisecond. For point-lookup queries (decode one row), the precompute is unjustified — fall back to decode-then-filter.
- **More moving pieces than a single fixed-width encoding.** `code_width_bits` is a real configuration knob; `codes_per_split`, the coalesced-format flag, the Tier-1-vs-Tier-2 choice, the optional `splits` buffer — each must be picked by the writer or by encoder heuristics. The validation campaign recommends defaults.
- **Encoder dependency choice is unresolved.** Either Vortex depends on `onpair_rs@ae590713` (an external pre-1.0 crate; small implementation effort, ongoing upstream coordination, format-pinning needed at every upstream bump) or reimplements OnPair16's training + bit-packing + decoder in-tree (medium-to-large effort, no upstream coordination, full local control). A hybrid path — depend now, reimplement when upstream stabilizes — is also viable. Each choice has different maintenance and reproducibility profiles. The format itself is independent of this choice (the wire bytes are the same either way), but the project plan must commit to one. Marked unresolved; resolution in implementation review.

## Alternatives

- **Adopt FSST12 directly and stop there.** Smallest delta from the current encoding; would deliver a real ratio + decode-speed lift over FSST8 and the SMEM-fits property is the same as OnPair16-12. But the training algorithm decision at 4K codes is empirical; OnPair-style pair-merging is plausibly better (sub-benchmark 2 measures). Also, FSST12 retains the escape mechanism, where OnPair16 is escape-free; on diverse data the escape-rate cost recurs.
- **Adopt GSST directly and stop there.** Best GPU throughput of the published encodings, but inherits FSST8's ratio and escape-branch CPU cost. Solves only the GPU surface.
- **Fixed 12-bit format with no `code_width_bits` parameter.** Simpler. But Vortex deployments differ — some don't care about GPU; some are GPU-first on Ada hardware where 11-bit is the sweet spot. Single-width forces a global compromise. The parameter is cheap to support (same bitstream shape, different table sizes) and the flexibility is real. Also, Stage 1 wants 16-bit; Stage 2 wants the parameter.
- **Use Vortex's existing `DictLayout` directly for string compression.** `DictLayout` works at the value level — each unique full string is one dictionary entry — which is excellent for low-cardinality categorical columns and bad for high-cardinality text. OnPair16 works at the byte/merge level, complementary to `DictLayout`. The two coexist; the writer picks the right one per column.

## Prior Art

The encoding proposed here is a small step in a five-year arc of FSST-family work. The four substantive contributions, listed in the order they shape the design:

### OnPair (the core algorithmic contribution this RFC carries)

*OnPair: Short Strings Compression for Fast Random Access*, **Francesco Gargiulo & Rossano Venturini** (University of Pisa). [arXiv:2508.02280v1](https://arxiv.org/abs/2508.02280v1), August 2025.

OnPair is the recent contribution this RFC is built around. It introduces (1) a single-pass longest-prefix-matching pair-merge codebook construction that's an order of magnitude cheaper than classical BPE and avoids FSST's "dependency issue" (FSST §4.1) via incremental merging rather than gain-estimation; (2) the **OnPair16** bounded-symbol-length variant ([paper §3.4.2, Algorithm 3](https://arxiv.org/abs/2508.02280v1)) whose tight `memcpy(buf, src, 16); buf += len[t]` decode loop achieves 6.5–7.8 GB/s CPU decode on AVX2 hardware (paper Table 3); (3) the lexicographically-ordered flat-bytes-with-Arrow-offsets dictionary representation ([paper §3.5](https://arxiv.org/abs/2508.02280v1); [`include/onpair/core/dictionary.h`](https://github.com/gargiulofrancesco/onpair_cpp/blob/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/core/dictionary.h)); and (4) a published compressed-domain predicate machinery — KMP, Aho–Corasick, prefix, equality, boolean composition — over the token stream ([`include/onpair/search/automata/`](https://github.com/gargiulofrancesco/onpair_cpp/tree/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/search/automata)). The combined design is a meaningfully different point in the random-access-string-compression design space than FSST occupies, and the experimental work in the paper is the strongest external evidence for the design choices in this RFC.

OnPair has received less attention in the database systems community than it warrants. We hope its adoption in a production columnar format helps that. The substantive technical credit for the encoding proposed in this RFC is OnPair's; Vortex's contributions are GPU decoding, parametric width selection, and the standard system-integration plumbing.

- C++ reference implementation: [`gargiulofrancesco/onpair_cpp@ae590713`](https://github.com/gargiulofrancesco/onpair_cpp/tree/ae590713515c7bb7893e14a757b484545e5339c3) (MIT). Pre-1.0; the README explicitly says *"The on-disk format and the public API may change without notice."*
- Rust reference implementation: [`gargiulofrancesco/onpair_rs@ac663abe`](https://github.com/gargiulofrancesco/onpair_rs/tree/ac663abe92ef3de26a65d9684e78f2aea28366ff) (MIT, pre-1.0).

### FSST (the family this encoding belongs to)

*FSST: Fast Random Access String Compression*, **Peter Boncz, Thomas Neumann, Viktor Leis**. PVLDB Vol 13, 2020. https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf.

FSST established the field-level random-access string compression archetype that this RFC's encoding inherits: small static symbol table, per-string decode, late-decompression-with-pushdown discipline. Vortex's current `FSSTArray` is FSST embedded as Vortex buffers + a Vortex-implemented DFA pushdown.

- Reference implementation: [`cwida/fsst@e638d4cf`](https://github.com/cwida/fsst/tree/e638d4cf8c26129d73c242a4127b42b975de5b63) (MIT). Ships the FSST8 variant from the paper and a 12-bit FSST12 variant introduced in the source tree post-paper (`fsst12.h`, `libfsst12.cpp`). The FSST12 source comment ([`fsst12.h:46`](https://github.com/cwida/fsst/blob/e638d4cf8c26129d73c242a4127b42b975de5b63/fsst12.h#L46)) reads *"12-bits FSST often does not work better dan 8-bits, but it will outperform it on datasets that are more chaotic, such as JSON and widely diverse URLs"* (the "dan/than" typo is in the original).

### GSST (the GPU decoder shape adopted here)

*GSST: Parallel string decompression at 191 GB/s on GPU*, **Robin Vonk, Joost Hoozemans, Zaid Al-Ars** (TU Delft). ACM SIGOPS Operating Systems Review, Vol. 59 No. 1, pp. 55–61, 2025. https://dl.acm.org/doi/10.1145/3759441.3759450 (paywalled).

GSST's full design — block parallelism, split parallelism, coalesced memory access; shared memory, alignment, async transfers — is documented in Robin Vonk's MSc thesis *"GSST: High Throughput Parallel String Decompression on GPU"* (TU Delft, 2024), available via the TU Delft repository at https://repository.tudelft.nl/ (search "Robin Vonk GSST"). Chapter 4 has the format-optimization details; Figures 4.1–4.4 are the canonical diagrams. The thesis abstract states GSST source will be released on GitHub.

### GPU-side FSST encoding (parallel work)

*High Throughput GPU-Accelerated FSST String Compression*, **Tim Anema, Joost Hoozemans, Zaid Al-Ars, H. Peter Hofstee** (TU Delft). VLDB 2025 ADMS Workshop. https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/ADMS/ADMS25-01.pdf.

The encode-side counterpart to GSST. 74 GB/s on RTX 4090. Not adopted by this RFC (we keep compression on the CPU), but cited here as the natural future complement to the decoder.

- Source: [`timanema/fsst-gpu@a0b639c3`](https://github.com/timanema/fsst-gpu/tree/a0b639c33bb0d6d6272c04a2e1c83877d1941f2f) (Apache-2.0).

### Vortex integration templates

- [`vortex-data/vortex@6f54d3d6/encodings/fsst`](https://github.com/vortex-data/vortex/tree/6f54d3d6dcc3d9b405658b4afcb4483616daa76f/encodings/fsst) (Apache-2.0). The existing `FSSTArray` — the integration template for `OnPair16Array`. The DFA LIKE pushdown there solves the same problem the OnPair16 automata solve via a different (FSST-tailored) technique.
- [`vortex-data/vortex@6f54d3d6/vortex-layout/src/layouts/dict`](https://github.com/vortex-data/vortex/tree/6f54d3d6dcc3d9b405658b4afcb4483616daa76f/vortex-layout/src/layouts/dict) (Apache-2.0). The integration template for `OnPair16Layout`.
- [`vortex-data/vortex@6f54d3d6/encodings/fastlanes`](https://github.com/vortex-data/vortex/tree/6f54d3d6dcc3d9b405658b4afcb4483616daa76f/encodings/fastlanes) (Apache-2.0). The crate-organization precedent — a family-named crate containing multiple algorithm-named Array types.

## Unresolved Questions

Each item below names its falsification criterion. The validation campaign in §[Validation](#validation-campaign) is the gate; "validation will tell us" without an upfront hypothesis is dodge, not science.

- **Encoder code-width selection policy (Stage 2+).** For Stage 1 the policy is fixed: 16-bit codes, matching the OnPair paper's §3.6 recommendation. For Stage 2+ the open question is: (a) caller specifies `code_width_bits` explicitly per array; (b) caller specifies a deployment-target label (`cpu_only`, `mixed`, `gpu_first`) and the encoder picks; (c) encoder picks dynamically from data characteristics (entropy, distinct n-gram count, expected ratio at each width on a sample). Current prior: (b) as the default with (a) as an escape hatch; (c) is appealing but requires the data-characteristics heuristic to be measured before committing. **Falsification:** if Stage 2 validation shows a single width dominates across all tested workloads on H100, drop to a fixed default with (a) as the only override.
- **`codes_per_split` default.** Recommended at 1024 codes for 12-bit codes from the split-size analysis priors in §[Split-size analysis]. **Falsification:** any of {512, 2048, 4096} wins by ≥20% throughput on H100 at 12-bit codes across the workload-shape sweep.
- **Coalesced Memory Access Format default.** Recommended as opt-in via `flags` bit 0 until measured to be a net win on OnPair16-at-12-bit. **Falsification:** the coalesced layout wins by ≥10% on at least two of three workload shapes (short / mixed / long) with no CPU-decode regression beyond 5%.
- **Persistent threads vs GSST block-per-SM mapping.** Reference variant is GSST's block-per-SM mapping (Vonk thesis §4.2.2 + §3.3 composability with ingestion pipeline). **Falsification:** persistent threads wins by ≥10% throughput on single-chunk decode at any chunk size, and the gain isn't reversed when ingestion-pipeline composability is measured.
- **Encoder dependency strategy.** Does Vortex depend on `onpair_rs@ae590713` (the external pre-1.0 crate) or reimplement OnPair16 in-tree alongside `FSSTArray`? This is a project-plan decision, not a format decision; documented in [Drawbacks](#drawbacks) but not resolved here. Resolution: implementation review; falsification criteria don't apply.

### Validation campaign

Every design choice above is a prior, not a conclusion. Before any production commitment, the following measurements must be done on representative Vortex workloads (TPC-H/TPC-DS string columns, JSON columns, ClickBench string columns, the existing Public BI benchmark string columns, plus the five OnPair-paper datasets — Book Reviews, Book Titles, News Headlines, Tweets, URLs — for direct comparability with the upstream paper):

1. **CPU decode throughput head-to-head.** FSST8 vs FSST12 vs OnPair16 across `code_width_bits ∈ {12, 14, 16}` (with the 16-bit row being upstream OnPair16-as-published). Scalar + AVX-512. Establishes the CPU baseline. Cross-check against OnPair paper Table 3 for the 16-bit case.
2. **Training algorithm at 12-bit codes.** Hold the bitstream constant (12-bit, 4096 dict, FSST12 layout) and vary only the training: (a) FSST12's 5-round evolutionary algorithm (FSST paper §4.1), (b) OnPair's single-pass pair-merge with early termination at 4096 codes (OnPair paper §3.2), (c) OnPair-16 trained to full capacity then truncated to top-4096-by-coverage. The plan recommends (b); this benchmark tests that prior. **This is the load-bearing measurement for the whole RFC.**
3. **GPU decode throughput sweep.** Port GSST to the same test harness as a baseline; then benchmark the OnPair16 encoding at `code_width_bits ∈ {10, 11, 12, 14, 16}` on H100, Ada (RTX 4090), and A100 (for comparison). Confirms the Mode A vs Mode B crossover and the per-GPU code-width recommendation.

   **Sub-benchmark: split sizing, kernel design, coalesced format.** At the recommended `code_width_bits = 12`: measure two kernel designs (GSST block-per-SM, persistent-thread warp-per-split) × four `codes_per_split` values (~512, ~1024, ~2048, ~4096) × two on-disk layouts (with and without coalesced reordering) × three workload shapes (short strings ~30 bytes mean, mixed bimodal, long free-text bodies >1 KiB). The proposal recommends (GSST mapping or persistent threads — whichever measures better) at `codes_per_split = 1024` with coalesced format opt-in.
4. **Ratio vs throughput Pareto frontier on GPU.** For each `code_width_bits`, plot decompression throughput against compression ratio. Visualize where the design lives in tradeoff space; identify the right defaults per GPU class.
5. **Dictionary materialization cost.** Cold and warm decode-start time at all code widths. Drives the random-access strategy and the Tier-2 caching policy.
6. **Per-string decode latency.** OnPair16 vs FSST for short strings (names, URLs, UUIDs). Verify SIMD ramp-up doesn't make short-string decode worse than FSST's tight scalar loop. The OnPair paper Table 3 random-access column already shows OnPair16 at 156–335 ns per random access vs FSST at 154–333 ns — comparable. Vortex-specific re-measurement on representative columns is the validation.
7. **Predicate pushdown end-to-end.** Vortex's existing FSST DFA vs OnPair16's compressed-domain automata (KMP, Aho–Corasick, prefix), on representative LIKE / equality / substring workloads (e.g., ClickBench Q19/Q20). The OnPair paper Section 4 reports per-pattern throughput; we measure the Vortex-integration overhead.
8. **Tier-2 ratio uplift.** Train one dict on each whole dataset and Arc-share across chunks; measure ratio improvement vs. per-array (Tier-1) training. Compare against the OnPair paper Table 3 ratios for sanity.
9. **Compression speed.** Per-array LPM training cost, including cold-cache effects. **The apples-to-apples comparison is FSST12 vs OnPair-at-12-bit**, both at the 4K-token dictionary capacity — there's no public head-to-head measurement of this, and the structural argument in §[Compression](#compression) predicts ~50–60% of FSST12 at AVX2 parity. Also measure an AVX-512-vectorized OnPair16 hot path (a focused engineering investment, ~1–2 weeks) to confirm the predicted gap-narrowing.
10. **Tier-2 dict materialization cost.** Confirm `OnceLock`-cached dict reuse pays for itself across realistic query patterns. Decide if `OnPair16Layout` needs a different caching policy than `DictLayout`'s default.

## Future Possibilities

- **Non-CUDA GPU support.** ROCm/HIP for AMD, SYCL for Intel, Vulkan/Metal for cross-vendor compute. The kernel shape generalizes; only the SMEM/L2 budgets and the primitive names change. A natural follow-on.
- **Encoder heuristics for code-width selection.** Once the validation data exists, derive an encoder heuristic that picks `code_width_bits` automatically from data characteristics (n-gram entropy, distinct-prefix count, observed compression at each width on a sample). Removes one knob from the caller.
- **AVX-512 acceleration of OnPair16 compression.** As discussed in §[Compression](#compression), an AVX-512-vectorized OnPair16 encoder is plausible engineering. Worth pursuing if compression-throughput-on-write matters in deployment.
- **Cross-encoding pushdown.** The token-stream-automata technique generalizes to any fixed-width-code dictionary encoding. Once OnPair16 lands, Vortex's compute layer can apply the same machinery to other encodings that materialize a code-to-bytes table at decode start.
- **Wider codes.** 20- or 24-bit codes for highly-redundant corpora where 16-bit isn't enough. Probably never wanted, but the format leaves room — the offsets array would widen to u64 and the decoder would need a new packing layout.
- **Adaptive recompression.** Vortex chunks could re-train their dictionaries at compaction time based on observed access patterns. Out of scope for this RFC; worth noting as a possibility.

---

## Implementation Specification

The sections above are written for human expert reviewers — the design narrative, the lineage, the comparisons, the tradeoffs. This section is the implementer's reference: the byte-exact wire format, error model, public Rust APIs, registry wiring, test plan specifics, and worked examples needed to implement the encoding without re-deriving design decisions from cited papers. A maximum-effort engineer (or model) given this section + the cited sources + the Vortex codebase should be able to produce a working, interop-correct implementation in a single pass.

The content below is normative for implementation but does not contradict the design sections above. If an inconsistency surfaces, the design section wins and this section is a bug.

### Wire format (byte-exact)

**Endianness.** Little-endian throughout. All multi-byte integer fields in metadata, `dict_offsets`, `code_offsets`, `splits`, and the bit-packed `codes` stream are little-endian. Matches Arrow and Vortex's other encodings.

**`DECOMPRESS_BUFFER_PADDING` = 16 = `MAX_SYMBOL_LEN`**. The `dict_bytes` buffer is required to have at least 16 zero-valued bytes past `dict_bytes_size`, so that an unconditional 16-byte over-copy from any offset `dict_offsets[t]` (where `t < n_tokens`) reads only allocated memory. The padding bytes must be zero; readers may rely on this for some SIMD shortcuts.

**`code_width_bits` validity.** Allowed values: 10, 11, 12, 14, 16. Other values are reserved and a conforming reader must reject them with an error (not interpret).

**Bit-packing layout for `code_width_bits` ∈ {10, 11, 12, 14, 16}.**

- *16-bit:* Each code is a `u16` little-endian. `codes[k]` lives at byte offset `2*k`.
- *12-bit (FSST12 layout):* Each pair of codes lives in 3 bytes. For codes `c_2i` and `c_{2i+1}` packed into bytes `b_3i`, `b_{3i+1}`, `b_{3i+2}`:
  ```
  bytes[3i]   = c_2i       & 0xFF     // low 8 bits of even code
  bytes[3i+1] = (c_2i      >> 8) & 0xF  |  (c_{2i+1} & 0xF) << 4  // top 4 of even, bottom 4 of odd
  bytes[3i+2] = (c_{2i+1}  >> 4) & 0xFF  // top 8 bits of odd code
  ```
  This matches the `cwida/fsst@e638d4c/libfsst12.cpp` decode loop (Algorithm 2 in the source). At end-of-stream, if `n_codes` is odd, the trailing half-byte of the last byte is zero-padded.
- *10-bit, 11-bit, 14-bit:* LSB-first bit-packed into a `u8` stream. Code `k` of width `w` occupies bits `[k*w, k*w + w)` counting from the LSB of byte 0, spilling into subsequent bytes. Equivalently: `code[k] = (load_u64_le(bytes[k*w/8 .. k*w/8 + 8]) >> (k*w % 8)) & ((1 << w) - 1)`. Trailing bits in the final byte beyond `n_codes * w` are zero-padded.

**Code-stream alignment.** The `codes` buffer is 16-byte-aligned at its start (Vortex buffer convention). No internal alignment requirements within the stream; the bit-packing fills contiguously. Implementations using AVX-512 / SIMD loads larger than 16 bytes should over-allocate the read end appropriately.

**`splits` array semantics.** `splits[s]` (for `s ∈ [0, n_splits)`) is the number of *uncompressed bytes* produced by decoding the codes `[s * codes_per_split, (s+1) * codes_per_split)`. The final split may have fewer than `codes_per_split` codes if `n_codes` is not a multiple of `codes_per_split`; in that case `splits[n_splits - 1]` reflects the actual uncompressed bytes produced by the remaining `n_codes - (n_splits - 1) * codes_per_split` codes. The relationship `n_splits = ⌈n_codes / codes_per_split⌉` is an invariant a conforming reader must check.

**Metadata invariants enforced at read time.**

| Invariant | Action on violation |
|---|---|
| `code_width_bits ∈ {10, 11, 12, 14, 16}` | Reject with `OnPair16Error::UnsupportedCodeWidth(u8)` |
| `n_tokens >= 256` | Reject with `OnPair16Error::InsufficientDictionary` |
| `n_tokens <= 2^code_width_bits` | Reject with `OnPair16Error::DictionaryExceedsCodeSpace` |
| `dict_offsets.len() == n_tokens + 1` and monotonically non-decreasing | Reject with `OnPair16Error::CorruptDictOffsets` |
| `dict_offsets[n_tokens] == dict_bytes_size` | Reject with `OnPair16Error::DictSizeMismatch` (dict_offsets is authoritative; dict_bytes_size is a redundant check) |
| `code_offsets.len() == n_strings + 1`, monotonically non-decreasing, with `code_offsets[n_strings] == n_codes` | Reject with `OnPair16Error::CorruptCodeOffsets` |
| if `n_splits > 0`: `n_splits == ⌈n_codes / codes_per_split⌉` and `sum(splits) == uncompressed_bytes` | Reject with `OnPair16Error::CorruptSplitMetadata` |
| `dict_bytes` has ≥16 trailing zero bytes past `dict_bytes_size` | Reject with `OnPair16Error::MissingDecodeBufferPadding` |

The validation pass runs once on Array construction; subsequent decode operations may assume the metadata is correct.

### Worked example

Input: `["hello", "world", "hello"]`, `code_width_bits = 12`. Assume training produced a 258-token dictionary: codes 0–255 are the single bytes, code 256 is the 5-byte token `"hello"`, code 257 is the 5-byte token `"world"`. (A realistic 4K-token training would produce a different dictionary; this minimal example shows the wire format.)

```
metadata:
  code_width_bits   = 12
  n_tokens          = 258
  n_strings         = 3
  n_codes           = 3
  n_splits          = 0          # GPU-decode not enabled in this example
  codes_per_split   = 0
  uncompressed_bytes = 15        # "hello" + "world" + "hello" = 5+5+5
  dict_bytes_size   = 266        # 256 single-byte + 5 + 5
  flags             = 0

buffer [0] dict_bytes (266 + 16 = 282 bytes, lexicographically ordered):
  // Lex order interleaves the single-byte and multi-byte tokens. For this example
  // (single-byte tokens 0x00..0xFF plus "hello" and "world"):
  //   "hello" sorts after the single byte 0x67 ('g') and before 0x68 ('h')
  //   "world" sorts after 0x76 ('v') and before 0x77 ('w')
  // The implementer must sort the *combined* dictionary lexicographically and
  // assign code IDs in sorted order. (Single-byte tokens at their natural
  // byte-value positions; merge-pair tokens at their lex-sorted positions.)
  // For brevity, the example below shows the byte values; the code IDs are
  // the *positions in the sorted order*.
  bytes:    [0x00, 0x01, ..., 0x67, 'h','e','l','l','o', 0x68, ..., 0x76, 'w','o','r','l','d', 0x77, ..., 0xFF, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                                    ^^^^^^^^^^^^^^^^^^^                   ^^^^^^^^^^^^^^^^^^^                  ^^^^^^^^ 16-byte padding ^^^^^^^^

buffer [1] dict_offsets (259 × u32 = 1036 bytes):
  offsets: [0, 1, 2, ..., 104, 109, 110, ..., 124, 129, 130, ..., 266]
                          ^ start of "hello" at byte 104, ends at 109
                                          ^ start of "world" at byte 124, ends at 129
  // dict_offsets[hello_code] = 104, dict_offsets[hello_code + 1] = 109 → length 5
  // dict_offsets[world_code] = 124, dict_offsets[world_code + 1] = 129 → length 5

  // The code IDs in this sorted order:
  hello_code = 104   // position in lex-sorted dict
  world_code = 124

buffer [2] code_offsets (4 × u32 = 16 bytes):
  [0, 1, 2, 3]    // string 0: codes[0..1] = "hello"
                  // string 1: codes[1..2] = "world"
                  // string 2: codes[2..3] = "hello"

buffer [3] codes (3 codes × 12 bits = 36 bits = 5 bytes, last 4 bits zero-padded):
  // codes[0] = hello_code = 104
  // codes[1] = world_code = 124
  // codes[2] = hello_code = 104
  //
  // Pack two codes per three bytes (FSST12 layout):
  //   bytes[0] = 104 & 0xFF = 0x68
  //   bytes[1] = (104 >> 8) & 0xF  |  (124 & 0xF) << 4 = 0x0 | 0xC0 = 0xC0
  //   bytes[2] = (124 >> 4) & 0xFF = 0x07
  // Then odd-numbered trailing code:
  //   bytes[3] = 104 & 0xFF = 0x68
  //   bytes[4] = (104 >> 8) & 0xF = 0x0  (high half-byte zero-padded)
  bytes: [0x68, 0xC0, 0x07, 0x68, 0x00]
```

Note: in a real-world array with a learned 4096-token dict, the single-byte tokens 0x00..0xFF would not all be present as codes 0..255 in the lex order; the lex order interleaves single-byte and multi-byte tokens. The implementer must sort the combined dictionary lexicographically by byte sequence and assign code IDs by sorted position. This matters because the prefix automaton (see [`prefix_automaton.h`](https://github.com/gargiulofrancesco/onpair_cpp/blob/ae590713515c7bb7893e14a757b484545e5339c3/include/onpair/search/automata/prefix_automaton.h)) relies on this lex ordering for O(log n_tokens) prefix-range computation.

### Public API (Rust)

These signatures mirror `FSSTArray`'s in [`vortex-data/vortex@6f54d3d6/encodings/fsst`](https://github.com/vortex-data/vortex/tree/6f54d3d6dcc3d9b405658b4afcb4483616daa76f/encodings/fsst) wherever possible. Where the OnPair16 design differs (parametric width, optional splits, automata-based pushdown), the API extends.

```rust
/// Tier 1 Array. Implements VTable<Encoding = OnPair16>.
pub struct OnPair16Array {
    metadata: OnPair16Metadata,
    buffers: [ByteBuffer; 4 /* or 5 if splits present */],
    validity: Validity,
}

/// Prost-encoded metadata, transparent on disk.
#[derive(prost::Message)]
pub struct OnPair16Metadata {
    #[prost(uint32, tag = "1")] pub code_width_bits: u32, // 10/11/12/14/16
    #[prost(uint32, tag = "2")] pub n_tokens: u32,
    #[prost(uint32, tag = "3")] pub n_strings: u32,
    #[prost(uint32, tag = "4")] pub n_codes: u32,
    #[prost(uint32, tag = "5")] pub n_splits: u32,
    #[prost(uint32, tag = "6")] pub codes_per_split: u32,
    #[prost(uint64, tag = "7")] pub uncompressed_bytes: u64,
    #[prost(uint32, tag = "8")] pub dict_bytes_size: u32,
    #[prost(uint32, tag = "9")] pub flags: u32, // bit 0: coalesced format
}

/// Tier 2 Layout. Implements LayoutVTable<Layout = OnPair16Layout>.
pub struct OnPair16Layout {
    dict: LayoutRef,  // BinaryView of n_tokens entries
    data: LayoutRef,  // codes + code_offsets + optional splits
}

/// Encoder entry point (Tier 1).
pub fn compress(
    strings: &dyn Array,                  // a UTF-8 or Binary Array
    config: &OnPair16Config,
) -> VortexResult<OnPair16Array>;

pub struct OnPair16Config {
    /// Caller-specified code width. None → encoder picks per deployment_target.
    pub code_width_bits: Option<u8>,
    pub deployment_target: DeploymentTarget,
    /// codes_per_split for the optional splits buffer. None → splits buffer absent.
    pub codes_per_split: Option<u32>,
    /// Apply GSST-style code reordering for GPU memory coalescing.
    pub coalesced_format: bool,
    /// Training hyperparameters (default to OnPair16 paper §3.2 values).
    pub training: TrainingConfig,
}

pub enum DeploymentTarget {
    CpuOnly,       // → encoder picks 16-bit
    Mixed,         // → encoder picks 12-bit (default)
    GpuFirst,      // → encoder picks 11-bit or per-GPU-class heuristic
}

pub struct TrainingConfig {
    /// Threshold for promoting a frequent pair to a new token.
    /// Default: max(2, ⌊log₂(sample_size_mib)⌋) per OnPair paper §3.2.1.
    pub pair_frequency_threshold: u32,
    /// Sample size as fraction of total input (1.0 = full input).
    /// Default: 0.05 (5%) for inputs >10 MiB; 1.0 for smaller inputs.
    pub sample_fraction: f64,
    /// RNG seed for reproducible sampling.
    pub seed: u64,
}

pub enum OnPair16Error {
    UnsupportedCodeWidth(u8),
    InsufficientDictionary,           // n_tokens < 256
    DictionaryExceedsCodeSpace,       // n_tokens > 2^code_width_bits
    CorruptDictOffsets,
    DictSizeMismatch,
    CorruptCodeOffsets,
    CorruptSplitMetadata,
    MissingDecodeBufferPadding,
    InputTooLarge { actual: u64, max: u64 },  // > u32::MAX codes
    // ... wraps VortexError for transport errors
}
```

All public types are `Send + Sync`, matching Vortex's Array convention. Internal mutability (the `OnceLock`-cached materialized symbol table in `OnPair16Layout`'s reader) uses `std::sync::OnceLock<Arc<[u8]>>` for the flattened decode table, scoped to a per-`LayoutRef` lifetime.

### Encoder algorithm

The training algorithm follows the OnPair paper §3.2 unchanged. Hyperparameter defaults pinned here:

- **Pair-frequency threshold:** `max(2, ⌊log₂(S)⌋)` where `S` is sample size in MiB, per OnPair paper §3.2.1 ("The threshold is set as a slowly growing function of dataset size"). Override via `TrainingConfig::pair_frequency_threshold`.
- **Sample size:** 5% of input for inputs >10 MiB; full input for smaller. Override via `TrainingConfig::sample_fraction`. The OnPair paper notes the sample is "randomly selected … shuffled" — implementers should use a deterministic seeded shuffle (`TrainingConfig::seed`) for reproducibility.
- **Training termination:** dictionary fills to `2^code_width_bits` tokens, OR no pair in the sample exceeds the frequency threshold, OR sample exhausted. Whichever comes first.
- **Token sorting:** after training completes, the dictionary is sorted lexicographically by byte sequence. Token IDs are reassigned to match sorted position. This is the OnPair paper §3.5 invariant and is load-bearing for the prefix automaton.

The encoder's parsing phase (paper §3.3) runs longest-prefix-matching against the final sorted dictionary for every input string. Implementation note: the LPM data structures (paper §3.4) — short-pattern hash + long-pattern bucket structure — are the hot path; reuse from `onpair_cpp` if the dependency decision lands that way (see Drawbacks).

### GPU decoder details

**Thread → split mapping.** One thread per split. Block dimension is `min(n_splits, 256)`. If `n_splits > 256`, the kernel loops each thread over multiple splits via `for (split_id = threadIdx.x; split_id < n_splits; split_id += blockDim.x)`. The 256 thread upper bound matches H100's preferred block size with 68 KiB SMEM per block.

**SMEM allocation (12-bit codes).** Static allocation per block:

| Allocation | Size | Note |
|---|---|---|
| `sym_bytes[4096 * 16]` | 64 KiB | Flattened symbol table (over-copied from `dict_bytes`) |
| `sym_len[4096]` | 4 KiB | Length of each token in bytes |
| `output_offsets[257]` | 1 KiB | Per-split exclusive-scan of `splits[]` (256 splits + sentinel) |
| Total | ~69 KiB | Within H100's 228 KB/SM budget at 3 blocks/SM |

For `code_width_bits < 12`, sizes scale down proportionally. For Mode B (14/16-bit codes), the dictionary is in global memory and the SMEM allocation drops to just `output_offsets` + scratch.

**Prefix-scan primitive.** Use `cub::BlockScan<uint32_t, BLOCK_DIM>` with `BLOCK_SCAN_WARP_SCANS` algorithm. Input: `splits[blockIdx.x * splits_per_block .. ]`. Output: exclusive scan into `output_offsets[]`.

**Dictionary materialization (Phase 1 of the kernel).** Each thread cooperatively memcpys a slice of `dict_bytes` into `sym_bytes` with 16-byte stride padding. Pseudocode:

```cuda
for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
    uint32_t off = dict_offsets[t];
    uint32_t len = dict_offsets[t + 1] - off;
    // Copy up to 16 bytes (over-copy safe due to dict_bytes padding)
    *(uint4*)&sym_bytes[t * 16] = *(uint4*)&dict_bytes[off];
    sym_len[t] = len;
}
__syncthreads();
```

**Decode (Phase 2 of the kernel).** Per-thread split decoder:

```cuda
uint32_t split_id = threadIdx.x;  // (or loop if n_splits > blockDim.x)
uint32_t code_start = split_id * codes_per_split;
uint32_t code_end = min(code_start + codes_per_split, n_codes);
uint8_t* out_ptr = &out_global[output_offsets[split_id]];

for (uint32_t k = code_start; k < code_end; k++) {
    uint32_t code = unpack_12bit(codes, k);  // bit-unpack from packed stream
    uint4 sym = *(uint4*)&sym_bytes[code * 16];
    *(uint4*)out_ptr = sym;
    out_ptr += sym_len[code];
}
```

The `unpack_12bit` function implements the two-codes-per-three-bytes layout from the Wire format section.

### Memory layout and allocation

- **Decoder output buffer:** caller-allocated. Caller must allocate at least `uncompressed_bytes + DECOMPRESS_BUFFER_PADDING` (= `uncompressed_bytes + 16`) bytes so the unconditional 16-byte over-copy on the final token is in-bounds.
- **CPU decode-time symbol-table cache:** allocated by `OnPair16Array` lazily on first decode call into an `Arc<[u8]>` scratch buffer of size `16 * n_tokens + n_tokens` (symbol bytes + lengths). Cached for the lifetime of the Array. For Tier 2 (`OnPair16Layout`), the cache lives in the layout reader's `OnceLock<Arc<FlatTable>>` and is shared across all chunks referencing the same dict.
- **GPU SMEM:** sized as above; per-block, kernel-local.

### Registry and dispatch

- **Encoding IDs:** assigned at PR merge from Vortex's encoding-ID registry. The RFC reserves the *name* `OnPair16` (Array) and `OnPair16Layout` (Layout); the *numeric* IDs are assigned by the registry maintainer to avoid conflict.
- **VTable impls:** `OnPair16: VTable<Encoding = OnPair16>` and `OnPair16Layout: LayoutVTable` mirror `FSST` and `Dict` respectively. Method-by-method, the implementations follow the corresponding FSST / Dict entries.
- **Static registration:** via the existing `inventory::submit!` pattern that other Vortex encodings use. Lives at the top of `encodings/fsst/src/lib.rs`.

### Crate boundaries

- **Crate location:** `encodings/fsst` (the existing FSST crate; `OnPair16Array` is a cousin to `FSSTArray`, both shipped from the same crate). No new crate.
- **External dependencies:** the dependency-strategy choice (depend on `onpair_rs` vs reimplement) is open and discussed in Drawbacks. If reimplementing, no new external dependencies; if depending, add `onpair_rs = "= 0.x.y-something"` with an exact pin to commit `ac663abe` until upstream tags a stable release.
- **CUDA feature flag:** Vortex's existing CUDA convention applies. Recommend adding the OnPair16 GPU decoder behind the same feature gate that wraps any other CUDA code. If Vortex doesn't yet have one, add `cuda` as an off-by-default Cargo feature in the `encodings/fsst` crate.

### Concurrency model

- `OnPair16Array` and `OnPair16Layout` are `Send + Sync`.
- The `OnceLock`-cached flattened symbol table in `OnPair16Layout`'s reader is `Arc<FlatTable>`; the `OnceLock` provides one-time-initialization safety across multiple concurrent readers of the same layout.
- The per-array decode-time scratch in `OnPair16Array` is owned by the Array (`Arc<[u8]>`); concurrent decodes against the same Array share the scratch read-only after first init.

### Test plan (specifics)

- **Pinned corpora:** the five OnPair-paper datasets (Book Reviews, Book Titles, News Headlines, Tweets, URLs) at the specific versions cited in the OnPair paper §4.1 (paper [21], [27], [16], [13], [15]); plus ClickBench (latest tag at PR merge), TPC-H scale-factor 1 with `lineitem`'s `l_comment`, TPC-DS scale-factor 1 with `web_returns`'s `wr_returned_date_sk`-adjacent text columns, Public BI benchmark at its latest published snapshot.
- **Two-way interop:** for `code_width_bits = 16`, the Vortex-emitted bytes must decode correctly under `onpair_cpp@ae590713`'s decoder, and `onpair_cpp`-emitted bytes must decode correctly under the Vortex decoder. Property test: random-input round-trip-and-cross-decode.
- **Property tests:** `proptest::collection::vec(any::<u8>(), 0..(1 << 20))` for raw byte arrays; `proptest::collection::vec(arbitrary_utf8(), 0..16384)` for UTF-8 strings. Round-trip invariant: `decode(encode(input)) == input` for all `code_width_bits` settings.
- **Regression tests:** none in OnPair's known-bug list at commit `ae590713`; pre-emptively add tests for: empty strings, all-empty rows, single-char strings, MAX-symbol-length (16-byte) repeating patterns, strings containing all 256 byte values.

### Performance budgets

These are the targets an implementer can measure during development to confirm they're on track. Failure to hit a budget isn't a hard stop, but is a signal to investigate.

| Budget | Target | Verifiable with |
|---|---|---|
| CPU decode throughput at 12-bit | ≥4 GB/s on Zen 4 / Sapphire Rapids, scalar (compiler auto-vectorized) | criterion benchmark with the OnPair-paper corpora |
| CPU decode throughput at 16-bit | ≥5 GB/s on Zen 4 / Sapphire Rapids, scalar | criterion |
| CPU decode L1 hit rate | ≥95% for the symbol table at 12-bit; ≥90% at 16-bit | `perf stat -e L1-dcache-load-misses,L1-dcache-loads` |
| GPU decode throughput at 12-bit (H100) | ≥200 GB/s | nsight compute, runtime divided by uncompressed bytes |
| GPU kernel launch overhead | ≤50 µs per chunk | nsight systems |
| Dict materialization (cold) | ≤500 µs for 4K-token dict, ≤8 ms for 64K-token dict | criterion micro-benchmark |
| Dict materialization (warm via OnceLock) | ≤100 ns per access | criterion |
| Encoder compression throughput | ≥150 MiB/s at 12-bit on the OnPair-paper corpora, scalar (auto-vectorized) | criterion; matches OnPair paper Table 1 row 12 |
| Encoder dict-training cost | ≤10% of total compression time on inputs >1 MiB | criterion, separate-phase timing |

If a budget can't be met after reasonable optimization, raise it as a finding in the implementation PR and discuss before merge.
