# Peer review synthesis: RFC 0033 Block-Decomposed TurboQuant with PDX

**Document reviewed:** `proposed/0033-block-turboquant.md`  
**Review date:** 2026-04-03  
**Purpose:** Consolidated findings from a detailed technical review (citations, papers, and spot-checks against arXiv HTML and GitHub).

---

## Executive summary

The RFC is unusually strong for an implementation plan: staged delivery, explicit approximation boundaries (SORF vs Haar, QJL vs MSE-only), and credible linkage to TurboQuant [1] and PDX [4]. For an expert audience, the highest-impact gaps are **broken or unverifiable citations**, **PDX speedup wording that does not match the PDX abstract**, **under-specified conditions for quantized dot product between two stored columns**, and **the blockwise MSE composition paragraph mixing deterministic algebra with probabilistic bounds**. Addressing those items—and making community claims auditable—would make the document review-resistant.

---

## Citations and bibliographic issues

### Broken GitHub reference

- **Finding:** `ggml-org/llama.cpp#20969` returns **404** (issue does not exist or was removed).
- **Action:** Replace with a **resolvable** link (e.g. [issue #20977](https://github.com/ggml-org/llama.cpp/issues/20977) “Feature Request: TurboQuant support,” or [discussion #21155](https://github.com/ggml-org/llama.cpp/discussions/21155)) and quote the **exact** claim about MSE-only vs MSE+QJL.

### Eviox report [7]

- **Finding:** “Eviox Tech Report v1.2.0, March 2026” has **no URL or DOI** in the RFC. Expert readers cannot verify bugs, Theorem 1 constant discussion, or QJL scale claims against that source.
- **Action:** Publish a stable PDF/link, **or** rephrase to “we verified against reference implementation at commit …” with reproducible steps.

### Community list [8]

- **Finding:** A list of repos plus “6+ groups” and “consensus” is **not** literature-grade evidence without commits, experiment definitions, and metrics.
- **Action:** Add a small **table** (source, commit or version, workload, bit width, metric, outcome) or move strong claims to “anecdotal / preliminary.”

### TurboQuant paper internal references

- **Lemma 1 / Theorem 2:** arXiv HTML aligns with “marginal density” material and **Definition 1** for QJL scaling; **theorem numbering** may differ in the ICLR 2026 camera-ready PDF. **Action:** Reconcile lemma/theorem numbers with the **final** PDF before wide distribution.

- **QJL scale (Definition 1):** The paper gives \(Q_{\text{qjl}}^{-1}(\mathbf z) := \frac{\sqrt{\pi/2}}{d}\mathbf S^\top\mathbf z\). The RFC’s contrast of `√(π/(2d))` vs `√(π/2)/d` is **correct** (ratio involves **√d**).

### PDX [4] speedup claims

- **Finding:** The PDX **abstract** reports beating horizontal SIMD layouts by **~40%** on average (order **1.4×** end-to-end for that comparison), and **2–7×** when **combining PDX with dimension-pruning** (ADSampling/BSA). The RFC’s blanket “**on average 2×**” for PDX vs row-major **overstates** the abstract’s headline scalar-scan claim unless restricted to a specific figure/setup.
- **Action:** Quote **40%** for the core PDX-vs-horizontal result; cite **2–7×** only for **PDX + pruning** (with section/figure reference when possible).

### Flash-KMeans [6]

- **Finding:** Flash-KMeans is a **GPU k-means** paper (assignment/update kernels), not TurboQuant decode. Referring to “following the double-buffered streaming pattern” suggests direct algorithmic lineage.
- **Action:** Clarify **analogy** (IO-aware fused kernels), not the same problem or method.

---

## Mathematics and methodology

### Theorem 1 and related quantities

- The **dimension-free** MSE bound \(D_{\text{mse}} \le (\sqrt{3}\,\pi/2)\,4^{-b}\) matches the arXiv HTML (intro + Theorem 1 region). The RFC’s **Eviox vs \(\sqrt{3\pi}/2\)** argument is directionally correct: **\(\sqrt{3}\pi/2 \approx 2.72\)** is not **\(\sqrt{3\pi}/2 \approx 1.535\)**.

- The proof chain also introduces quantities such as \(\mathcal C(f_X,b)\) with a **\(1/d\)** factor in intermediate steps. The RFC can briefly note **\(\mathcal C\)** vs **\(D_{\text{mse}}\)** so readers see the full proof stack was considered.

### Block decomposition and composed MSE bound

- The **algebraic** identity partitioning \(\|\mathbf x - \hat{\mathbf x}\|^2/\|\mathbf x\|^2\) by orthogonal blocks is **correct**.

- The step from per-block **probabilistic** guarantees to a global bound should be stated in terms of **expectations** (linearity) and assumptions on randomization, not as a purely **pointwise** weighted average unless the theorem is worst-case (it is not, as stated).

- **Conceptual gap:** TurboQuant’s analysis uses **one** global Haar rotation and **high-\(d\)** near-independence across coordinates. **Independent SORF per block** with **smaller \(B\)** may weaken the “coordinates act like independent scalar sources” story even when the **marginal** after Haar in \(\mathbb R^B\) is correct. The RFC already plans empirical validation; **explicitly call out \(B\)-dependence of near-independence**.

### Centroids and block dimension

- Centroids must use the **\(B\)-dimensional** marginal (exponent **\((B-3)/2\)**). The RFC states this; good.

- **Minimum block size:** Global **\(d \ge 3\)** avoids Beta singularities; state that **each block** satisfies **\(B \ge 3\)** under the chosen policy (**\(B \ge 64\)**), so the marginal is well-defined.

### DCT discussion

- In the “Why not DCT?” paragraph, the marginal is written with **\((d-3)/2\)**; for per-block discussion, **\((B-3)/2\)** is the relevant exponent to avoid confusion.

---

## Systems and integration

### Quantized dot product / cosine: two stored columns

- For **column vs query** re-encoded with the **column’s** rotation and centroids, the story is clear.

- For **two TurboQuant-encoded columns**, a fast quantized inner product requires **identical** rotation parameters (**bit-identical `mse_rotation_signs`**, same seeds/structure), not only the same **\(B\)** and centroids. The RFC should **require rotation identity** for the two-sided fast path or **mandate exact fallback**.

### Mixed precision (f64 norms, f32 directions)

- Generally sound; a **brief** note on numerical ordering or tiny norms avoids pedantic corner-case questions.

### PDX layout and indexing

- Implementers will want a **clear mapping** from logical dimension index (spanning TQ blocks) to **PDX transposed offsets**—either a formula or a short diagram.

### Slice/take with PDX

- Full **un-transpose to FSL** is simple but can imply **large transient cost** on small slices. Worth noting **worst-case behavior** and optional **64-row-aligned** fast paths.

### FLOP table

- Label counts as **heuristic**; real cost is often **memory bandwidth** and constant factors in butterflies.

### GPU / VPDPBUSD

- **VPDPBUSD** is a **specific** mixed int8 dot-product idiom, not arbitrary uint8×uint8. Max-Lloyd centroids are **not** naturally constrained to byte-quantized linear scales; treat “linear enough for tensor cores” as a **strong** empirical hypothesis.

---

## Experimental plan and datasets

### Gaussian “pessimistic baseline”

- For **isotropic** Gaussians, a random orthogonal transform is **distributionally neutral**, but that does not make the baseline “pessimistic” for **all** error modes; it can be **misaligned** with heavy-tailed or clustered embeddings. **Soften** wording to: theory anchor / sanity check, not a proxy for worst-case production.

### DEEP \(d=96\)

- Correctly noted: **no** power-of-two **\(\ge 64\)** divides 96, so the RFC’s block rule forces **padding / straggler** path. Good.

### Popular dimensions

- Optional: add rows for dimensions such as **2560** or **1280** if the RFC targets “common model dims” broadly.

---

## Compression ratio section

- **“30% storage improvement”** is easy to misread: the worked example is roughly **29% higher compression ratio** (4.8× → 6.2×) and about **24% fewer compressed bits per vector** for \(d=768\), \(b_{\text{mse}}=5\). **Disambiguate** ratio vs bit reduction.

- **Shared** centroids and SORF signs: remind readers that shared cost is **amortized over \(N\)**; **small** columns are metadata-sensitive.

---

## Minor editorial nits

- Prefer **“greatest”** over **“largest”** for “power-of-two that divides \(d\)” (standard math English).

- PQ row: “8 bits per sub-vector” is a **typical** configuration, not the definition of PQ; qualify as such.

- “Indexing time: Zero” vs PQ training: fair as **no k-means training**, but **encode-time** work remains; soften **“zero”** to avoid pedantic pushback.

- **QJL variance scaling (“\(d/B\) times more”):** align wording with **Lemma 4**’s **exact** statement in the PDF (variance of **averaged** estimators, constants).

---

## Positive highlights (worth preserving)

- Clear **staging** (MSE-only → blocks → PDX → optional QJL).

- Honest **SORF vs Haar** and **SORF for QJL** vs Gaussian **S**.

- **Theorem 1 constant** clarification vs mistaken \(\sqrt{3\pi}/2\) interpretation.

- **PDX open-source delta** (SQ8 tiling, ADSampling, zones) is valuable context.

- **Migration / single array ID** story is clean for a greenfield encoding.

---

## Suggested priority order before external expert send-out

1. Fix **llama.cpp** link and verify **PDX** speedup sentences against [4].  
2. Tighten **block MSE** subsection (expectations, \(B\)-dependence).  
3. Specify **rotation-parameter identity** (or fallback) for **two-column** quantized dot.  
4. Make **[7]** and **[8]** **auditable** or soften claims.  
5. Add **Flash-KMeans** analogy disclaimer, **compression ratio** disambiguation, **slice/PDX** cost note.

## Deliverables (this review)

| File | Purpose |
| ---- | ------- |
| `proposed/0033-block-turboquant-review-synthesis.md` | This document: consolidated findings and recommended actions. |
| `proposed/0033-block-turboquant-revised.md` | Full RFC text with proposed edits applied (does not replace `0033-block-turboquant.md`). |
