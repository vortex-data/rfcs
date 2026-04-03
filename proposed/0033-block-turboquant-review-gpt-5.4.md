# Review of `0033-block-turboquant.md`

## Scope

This review checks the RFC against:

- the TurboQuant paper (`arXiv:2504.19874`)
- the PDX paper (`arXiv:2503.04422`)
- the cited SORF / ORF paper (`arXiv:1610.09072`)
- the cited PQ / OPQ papers
- the referenced open-source implementations and publicly available discussions that could be located

The goal of this review is not to argue against the proposal direction. The goal is to make the RFC maximally defensible when read by experts who will check claims, citations, and wording very closely.

## Executive Summary

The proposal direction is plausible, and several technical points in the RFC are solid, especially:

- the Theorem 1 constant correction
- the distinction between orthogonal MSE rotation and Gaussian QJL projection
- the rationale for treating SORF as an approximation rather than as a theorem-preserving drop-in replacement

The largest problems are not in the core block-decomposition idea. They are in the rhetoric and sourcing around it:

1. The RFC currently overclaims that community evidence supports dropping QJL for **ANN ranking**, when the located evidence is primarily about **KV-cache attention**.
2. The RFC overstates the PDX paper's speedup claim.
3. The PQ comparison contains an unsupported superiority claim that is likely to irritate reviewers.
4. The ADSampling integration discussion makes a nontrivial compatibility question sound easy.
5. The citation hygiene for `[7]` and especially `[8]` is not strong enough for external review.

## Primary Findings

### 1. Overclaim: evidence does not currently justify the ANN-ranking conclusion

The most serious issue is the scope of the QJL claim. The current RFC says:

> Community findings from 6+ independent TurboQuant implementations consistently show that MSE-only outperforms MSE+QJL for attention and ANN ranking in practice.

The evidence I could verify does support a strong claim for **KV-cache attention**:

- `tonbistudio/turboquant-pytorch` explicitly argues that QJL hurts because softmax amplifies variance.
- `scos-lab/turboquant` also reports MSE beating Prod/QJL for attention-like workloads.
- other community sources appear to be in the same family of KV-cache experiments.

However, that is not the same thing as evidence for ANN ranking. In fact, one of the strongest located community sources explicitly distinguishes the two and says QJL may still work for vector search because there is no softmax nonlinearity.

That means the current wording is too strong in two ways:

- it extends **attention evidence** to **ANN ranking**
- it uses that extension to justify a product decision for Vortex's search/storage use case

For outside review, the RFC should either:

- narrow the claim to KV-cache attention only, or
- add actual ANN experiments and cite those directly

### 2. Mis-citation: the PDX paper is overstated

The RFC currently says PDX achieves "on average 2x speedups over SIMD-optimized row-major kernels."

The PDX paper's abstract says:

- PDX beats SIMD-optimized horizontal kernels by **average 40%**
- pruning approaches recover **2-7x** benefit when used with PDX

Those are different claims. The RFC currently mixes them together in a way that overstates what the paper says.

### 3. Unsupported comparison: TurboQuant is presented as likely superior to PQ on uniform embeddings

The RFC currently says:

> For uniformly distributed embeddings, TurboQuant's analytically optimal centroids should match or exceed PQ's learned codebooks.

This is not supported by the cited PQ/OPQ literature, and it is not obviously true. PQ uses learned **vector** codebooks in subspaces, while TurboQuant uses rotated **scalar** quantization. The correct contrast is:

- TurboQuant is training-free, data-oblivious, and analyzable.
- PQ/OPQ are data-dependent and require training.
- PQ/OPQ may still be empirically stronger because vector codebooks are more expressive.

The current sentence sounds like a theorem-shaped statement without theorem-level support.

### 4. ADSampling integration is presented too casually

The RFC suggests that TurboQuant and ADSampling might share the same rotation.

That is not obviously compatible with the proposed Stage 2 design:

- ADSampling relies on a single full-dimensional random orthogonal projection whose coordinates can be sequentially sampled.
- Stage 2 proposes per-block rotations with blockwise norms and blockwise accumulation.

A blockwise-rotated representation is not automatically interchangeable with the globally rotated representation assumed by ADSampling's pruning logic. This may still be possible, but it is a research question, not a straightforward integration detail.

### 5. Citation hygiene is too weak for external review

Two issues stand out:

- `[8]` is a prose bundle of repos and issue references rather than an auditable citation.
- `[7]` was not publicly discoverable under the cited title during review.

For a document going to experts, `[8]` should be expanded into explicit entries with:

- repository / issue / PR URL
- commit SHA or tag if relevant
- workload type: KV attention vs ANN search
- metric: perplexity, recall@k, cosine, etc.
- conclusion actually supported by that source

If `[7]` is intended as a public citation, it should have a public URL. If it is private, the RFC should not lean on it heavily in externally circulated form.

### 6. GPU section uses CPU instruction terminology

The GPU section references `VPDPBUSD`, which is an x86 CPU instruction, not a GPU tensor-core primitive. The section needs either:

- CPU wording, or
- GPU-native terminology

Otherwise it looks like a hardware-model mix-up.

### 7. One worked-example note contradicts the design

The Stage 2 worked example for `d=768, B=256, k=3` is labeled "zero padding" in the notes column. That should be removed or changed; Stage 2 is explicitly avoiding padding in that case.

## Secondary Notes

These items looked good or at least defensible:

- The Theorem 1 constant appears correctly interpreted as `sqrt(3) * pi / 2`.
- The QJL scale-factor correction appears correct.
- The distinction between QR/Haar rotation for MSE and Gaussian `S` for QJL is correctly emphasized.
- The revised VIBE citation is now correct.

## Recommended Editorial Strategy

Before sharing this RFC externally, the safest editorial move is:

1. Keep the proposal structure.
2. Tighten all empirical claims to exactly what the evidence shows.
3. Replace suggestive superiority language with narrower, falsifiable wording.
4. Mark ADSampling integration as speculative / future investigation.
5. Strengthen citations, especially `[8]`.

## Proposed Redline

This redline is intentionally targeted. It focuses on the passages that most need correction before external circulation.

### 1. Summary: narrow the QJL claim

#### Proposed replacement

```diff
-QJL correction is deferred to a later stage and may ultimately be dropped.
-Community findings from 6+ independent TurboQuant implementations consistently
-show that MSE-only outperforms MSE+QJL for attention and ANN ranking in
-practice [8].
+QJL correction is deferred to a later stage and may ultimately be dropped.
+Community findings from multiple independent TurboQuant implementations
+consistently show that MSE-only outperforms MSE+QJL for KV-cache attention in
+practice [8]. For ANN ranking and vector-search workloads, the evidence is
+currently less complete, so QJL should remain an empirical question rather than
+a settled conclusion.
```

### 2. PQ comparison: remove unsupported superiority language

#### Proposed replacement

```diff
 TurboQuant trades PQ's flexibility (data-dependent codebooks can exploit
 structure) for data-obliviousness (no training, provable bounds, zero indexing
 time).
-For uniformly distributed embeddings, TurboQuant's analytically optimal
-centroids should match or exceed PQ's learned codebooks. For highly structured
-data, PQ may still win empirically.
+In return, PQ and OPQ retain a major advantage in expressivity: they learn
+sub-vector codebooks from data rather than applying an analytic scalar
+quantizer. In practice this means TurboQuant is attractive when training-free
+operation, simple deployment, and theoretical guarantees matter most, while PQ
+or OPQ may still win empirically when a learned vector codebook can exploit
+dataset-specific structure.
```

### 3. Community QJL section: separate attention from ANN

#### Proposed replacement

```diff
 ### Community findings on QJL
 
 Multiple independent TurboQuant implementations have converged on a
-significant practical finding: **MSE-only consistently outperforms MSE+QJL for
-attention and ANN ranking**. The mechanism is a variance-bias tradeoff:
-TurboQuant's QJL correction eliminates bias but increases variance, and softmax
-attention (and cosine/L2 ranking) amplifies variance more than bias. At the same
-total bit budget, allocating all bits to MSE (more centroids, lower variance)
-beats splitting between MSE + QJL (fewer centroids + 1-bit correction). This has
-been confirmed by 6+ groups across Python, C, and Rust implementations [8].
+significant practical finding for **KV-cache attention**: MSE-only often
+outperforms MSE+QJL at the same bit budget. The likely mechanism is a
+variance-bias tradeoff: QJL removes bias in raw inner-product estimation but
+adds variance, and the softmax nonlinearity can amplify variance more than it
+penalizes bias. In that setting, allocating all bits to MSE (more centroids,
+lower variance) can beat splitting the budget between MSE + QJL. This behavior
+has been reported by multiple groups across Python, C, and Rust implementations
+[8].
 
-This finding strongly supports making MSE-only the default strategy for our
-columnar storage use case (ANN search, cosine similarity ranking).
+For ANN search, cosine ranking, and other non-softmax vector-search workloads,
+the evidence is currently less settled. MSE-only is still a reasonable default
+because it is simpler and better supported by the current implementation work,
+but the RFC should treat the ANN question as empirical until evaluated on ANN
+datasets with recall@k and ranking metrics.
```

### 4. PDX section: correct the speedup claim

#### Proposed replacement

```diff
 PDX [4] is a data layout for vector similarity search. The paper (SIGMOD '25)
 describes a dimension-major layout within fixed-size blocks of 64 vectors,
 enabling the compiler to auto-vectorize the inner distance loop over vectors
-rather than dimensions, achieving on average 2× speedups over SIMD-optimized
-row-major kernels on modern CPUs. The block size of 64 is empirically optimal
+rather than dimensions. In the paper, this yields average speedups of about 40%
+over SIMD-optimized row-major kernels for the direct-kernel comparison, while
+dimension-pruning methods recover much larger gains when paired with the PDX
+layout [4]. The block size of 64 is empirically optimal
 across AVX-512, AVX2, and NEON architectures [4].
```

### 5. ADSampling integration: mark as speculative

#### Proposed replacement

```diff
 **Shared rotation with ADSampling.** Both TurboQuant and ADSampling apply a
 random orthogonal rotation to make coordinates independent. If we integrate
 ADSampling-style dimension pruning (see Stage 3), the same rotation could serve
 both purposes: producing the Beta distribution for quantization AND enabling
-hypothesis-testing for early pruning. This would avoid rotating the data twice.
-Note that the query must also be rotated at query time with the same rotation
-matrix (stored as a shared child); ADSampling already requires this.
+hypothesis-testing for early pruning. However, this is not automatic under the
+Stage 2 block-decomposed design: ADSampling is formulated around a single
+full-dimensional random projection, whereas Stage 2 introduces per-block
+rotations and per-block norm weighting. Reusing one rotation across both systems
+should therefore be treated as a future research direction that requires either
+new analysis or direct empirical validation. If it proves viable, it would avoid
+rotating the data twice. The query would also need to be rotated at query time
+with the same stored transform.
```

### 6. Worked examples: fix the contradictory note

#### Proposed replacement

```diff
-| 768           | 256  | 3   | 3×256×5 + 3×32 = 3936 | 6.2×  | Block decomp; zero padding |
+| 768           | 256  | 3   | 3×256×5 + 3×32 = 3936 | 6.2×  | Block decomp; no padding   |
```

### 7. GPU section: remove the CPU/GPU terminology mix

#### Proposed replacement

```diff
 At b_mse=8, codes are uint8 indices (0-255). Direct int8 tensor core GEMM
-(using codes as the unsigned operand in VPDPBUSD) requires approximately linear
+or byte-dot-product execution on low-precision hardware requires approximately linear
 centroids — but at high B the Max-Lloyd centroids are already near-uniform
 (the Beta distribution is highly concentrated, approaching Gaussian, for which
 high-resolution optimal quantization is approximately uniform). Whether the
 existing Max-Lloyd centroids are "linear enough" for hardware dot-product
 instructions is an empirical question worth testing before introducing a
 separate linear quantization mode.
```

If you want to be more explicit, you could instead split this into separate CPU and GPU paragraphs.

### 8. Reference `[8]`: make it auditable

#### Proposed replacement

Replace the current bundled prose citation with something like:

```diff
-[8] Community TurboQuant implementations and findings. Key sources:
-tonbistudio/turboquant-pytorch (PyTorch, V3 MSE-only findings),
-ggml-org/llama.cpp#20969 (C/C++, quantized attention analysis),
-0xSero/turboquant (Triton kernels), vivekvar-dl/turboquant (pip package),
-scos-lab/turboquant (reference reproduction). Consensus: MSE-only beats
-MSE+QJL for attention and ANN ranking at all tested bit widths.
+[8] Community TurboQuant implementation reports. These sources primarily study
+KV-cache attention rather than ANN search, and should be cited individually
+with exact URLs and workload scope in the final external draft. Representative
+examples include:
+- tonbistudio/turboquant-pytorch, issue #10 and README discussion of V2
+  (MSE+QJL) vs V3 (MSE-only) behavior on attention and generation.
+- scos-lab/turboquant README discussion of MSE vs Prod/QJL for KV-cache
+  attention workloads.
+- 0xSero/turboquant README and validation scripts for paper checks and
+  implementation behavior.
+These sources support a strong claim for KV-cache attention. They do not, by
+themselves, establish the same conclusion for ANN ranking.
```

This version is intentionally conservative. If you have additional ANN-specific sources, add them here explicitly and then strengthen the main text accordingly.

### 9. Reference `[7]`: either publish it or weaken dependence on it

#### Proposed replacement note

Not a text diff, but a release recommendation:

- If `[7]` is public, add a direct URL.
- If `[7]` is private or unstable, reduce dependence on it in externally
  circulated prose.

For example, this sentence is fine if the report is public:

```diff
-The Eviox corrections study [7] identified six material bugs in the paper's
+A third-party implementation review [7] identified six material bugs in the paper's
 reference Python implementation.
```

But the best fix is still to make the citation resolvable.

## Optional Stronger Rewrite

If you want the RFC to sound maximally careful in front of skeptical reviewers, the simplest global substitution is:

- replace `consistently outperforms` with `has often outperformed`
- replace `consensus` with `reported behavior`
- replace `supports making MSE-only the default for ANN` with `supports evaluating MSE-only first, while keeping ANN ranking as an empirical question`

That wording preserves the proposal but removes the most attackable overclaims.

## Suggested Next Pass

If you want a tighter external-facing RFC, the next revision should:

1. apply the redline above
2. expand `[8]` into exact citations
3. add one explicit sentence saying which claims are backed by theorem, which by implementation, and which remain hypotheses
4. add ANN-specific experiments before claiming ANN superiority for MSE-only over QJL
