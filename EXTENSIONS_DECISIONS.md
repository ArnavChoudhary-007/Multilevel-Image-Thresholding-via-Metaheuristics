# Extension decisions

Applied Statistics & Stochastic Processes project: three extensions to the study.
Every parameter, image and hypothesis below was fixed **before** any extension
experiment was run. Later changes are added under "Amendments" with a date and a
reason. Nothing here is edited silently.

**Status:** fixed on 9 Nov 2026, before the main runs of Parts A, B and C.

**History of this file.** An earlier draft was written by a previous agent session and
never committed. Before committing, it was corrected as follows:

- **(a) Unverified claims about the Tsallis source and q were removed** (see C1 and C2).
- **(b) The histogram descriptions in the image table are now computed** by
  `describe_images.py` instead of written by eye.
- **(c) The DP for Part B is re-timed on this laptop.** The draft reused `bitdepth.csv`,
  which was timed on a different machine.
- **(d) The relative gap is computed exactly as in `runs.csv`, i.e. gap/|J\*|.** The
  draft's runners used gap/max(1, |J\*|).
- **(e) The exact pseudo-additive Tsallis caveat was added** (C3).

**What was seen before this file was fixed:**

- Smoke runs, at budget 1,000 with 2 seeds on 2 images, were executed only to test the
  code, then deleted without analysis.
- The test suites print the exhaustive Tsallis optima for camera and clock, plus the
  counterexample in `test_tsallis.py` §5. No decision below depends on them.

---

## Global

| ID | Decision | Reason |
|---|---|---|
| G1 | **Protocol unchanged:** 5,000 evaluations per run, enforced by `CountedObjective`; seeds 0–29 (Part B: 0–9); the per-cell median over seeds is the response; Friedman + Kendall's W + Nemenyi; 21 Wilcoxon signed-rank tests with Holm correction and rank-biserial r; ART ANOVA for algorithm × k with image as the block. `PARAMS` in `optimisers.py` unchanged. | Comparability with the 13,440-run study. |
| G2 | **Gap and hit are defined exactly as in `run_experiments.py`:** G = (J\* − J)/\|J\*\|, and a hit means J\* − J ≤ 1e-9 · max(1, \|J\*\|). For the scores in this study (\|J\*\| > 1), the hit rule is G ≤ 1e-9. | Same definition as the existing `runs.csv`, so the two sets of runs can be combined. |
| G3 | **Read-only:** `results/runs.csv` and every existing figure and table. All new outputs use new file names; figures go in `figures/extensions/`. | Integrity of the reported study. |
| G4 | **Branch:** `extended` (the user's instruction). Small commits, **no push**. | The user's instruction. |
| G5 | **Floating point across machines:** `runs.csv` was made on Linux with NumPy 2.4.4, the extensions on this laptop with NumPy 2.0.2. Re-running sampled rows of `runs.csv` here gives identical thresholds; Kapur scores can differ in the last binary digit, which the original code also shows on this laptop. `test_extensions.py` checks thresholds exactly and scores to 1e-12. | About 1e-15 relative, far below the 1e-9 hit tolerance. It cannot change a hit or a rank. |

## Part A: 24 images

| ID | Decision | Reason |
|---|---|---|
| A1 | **Selection rules:** natural, microscopy, medical or texture images; at least 16 occupied grey levels; no synthetic or binary images (`horse`, `checkerboard`, `shepp_logan_phantom`, `colorwheel`, `logo`, `binary_blobs`); no duplicates (`cat` is pixel-identical to `chelsea`, so only `chelsea`); no tiny crops (`lfw_subset`, 25 × 25). | Thresholding benchmarks need real, varied histograms. |
| A2 | **Images:** the 11 qualifying images bundled with scikit-image (brick, chelsea, grass, gravel, hubble_deep_field, immunohistochemistry, microaneurysms, retina, rocket, text, stereo_motorcycle), plus 5 from scikit-image's own hosted data, downloaded once via `pooch` and hash-checked (eagle, human_mitosis, nickel_solidification, skin, vortex). This is option (b) of the plan. A previous agent session downloaded the files. No explicit user approval of option (b) is on record in this session: **to be confirmed by the user.** | Only 11 bundled images qualify. Using scikit-image's hosted data keeps one source and one loader. |
| A3 | **Fixed frames:** stereo_motorcycle → left view; nickel_solidification → frame 0 (uint16, reduced by `>> 8`); vortex → frame 0. Colour images go through `rgb2gray` → `img_as_ubyte` as before. | Fixed in advance, not chosen by looking at results. |
| A4 | **Image table:** produced by `describe_images.py` → `results/ext_image_table.csv`. It covers size, levels, range, and the computed modes, skewness and dark share. | No hand-written descriptions. |
| A5 | **Grid:** 16 new images × k ∈ {2, 3, 4, 5} × {Otsu, Kapur} × 7 algorithms × 30 seeds = **26,880 runs**, written to `results/runs_extended.csv` with `run_experiments.py --extended`. | Same grid as the original study. |
| A6 | **Analysis:** the 8 + 16 = 24 images together, with the protocol unchanged. Nemenyi CD = q·√(K(K+1)/(6N)) with K = 7 and N = 24. With q = 2.948 that gives 1.84; the code uses the studentised-range value, as `analyse_stats.py` does. Also reported: the effective N of each Wilcoxon pair (pairs with zero difference are dropped), and Kendall's τ between the 8-image and 24-image average-rank orders. | The brief. |
| A7 | **Hypotheses:** H1: the ranking agrees with the 8-image ranking (Kendall's τ > 0). H2: Friedman stays significant in every (criterion, k). H3: at least one Wilcoxon–Holm pair reaches p < 0.05 in at least one condition. Each may fail and will be reported as found. | Stated before the data exist. |

## Part B: bit depth

| ID | Decision | Reason |
|---|---|---|
| B1 | **Search range:** thresholds in [1, L − 2]. `optimise()` reads L from the objective. At L = 256 the behaviour is identical to the original code (G5, `test_extensions.py`). | A single code path for all L. |
| B2 | **SA:** step sd = 8 · L/256. **PSO** (vmax = 0.2 · range) and **GA** (mutation sd = 0.1 · range) were already relative to the range. **DE** and **GWO** move by differences between population members, so they scale automatically. **Hill climber** stays at ±1, because that defines the method; it is expected to collapse at large L. | Relative step sizes, fixed a priori; none were tuned. |
| B3 | **Inputs:** `deep_histogram(L, seed=0)` from `bitdepth_scaling.py` (dithered camera), L ∈ {256, 1024, 4096, 16384, 65536}, k ∈ {2, 3, 5}, Otsu and Kapur, 7 algorithms, seeds 0–9 → 2,100 runs. | The brief, and the same inputs as the existing Table 8. |
| B4 | **DP timing:** the DP optimum and time are measured once per (L, k, criterion) on this laptop (`results/bitdepth_dp_laptop.csv`). Timing uses the minimum of 3 repeats for L ≤ 4096 and 1 run above, the same rule as `bitdepth_scaling.py`. | A crossover needs both sides timed on the same CPU. `bitdepth.csv` has no optima. |
| B5 | **Timing environment:** everything runs in one process, one job at a time, with BLAS/OpenMP threads set to 1. Optimiser time covers `optimise()` only; building the objective is timed separately. The DP time includes building its tables. | Fair, single-threaded timings. |
| B6 | **Crossover:** the smallest **measured** L at which the DP time exceeds the median optimiser run time. No interpolation. | Only measured numbers are reported. |
| B7 | **Native 16-bit check:** `cells3d` slice z = 30, nuclei channel, native L = 65,536, k = 2, both criteria, 10 seeds. The image was downloaded automatically by `skimage.data` through `pooch`; it is used, and the report says so. | A sanity check on a real 16-bit histogram. |

## Part C: Tsallis criterion

| ID | Decision | Reason |
|---|---|---|
| C1 | **Criterion:** S_q(t) = Σ_c S_c + (1 − q) · Π_c S_c, with S_c = (1 − Σ_{i∈c} (p_i/ω_c)^q)/(q − 1). A zero-mass class gives −∞ (rejected), as for Otsu and Kapur. **Sources:** Tsallis thresholding goes back to M. Portes de Albuquerque, I. A. Esquef, A. R. Gesualdi Mello, "Image thresholding using Tsallis entropy", *Pattern Recognition Letters* 25(9):1059–1065, 2004, DOI 10.1016/j.patrec.2004.03.003. A multilevel Tsallis method is Y. Zhang, L. Wu, "Optimal Multi-Level Thresholding Based on Maximum Tsallis Entropy via an Artificial Bee Colony Approach", *Entropy* 13(4):841–859, 2011, DOI 10.3390/e13040841. **Verified** (Crossref, Semantic Scholar): titles, journals, volumes, pages, years and DOIs. **Not verified:** the exact multilevel formula each paper uses, the q values they report, and whether the 2004 paper is bilevel only. The full texts were not readable (publisher pages returned 403). The author lists also need checking against the PDFs: Crossref repeats the first author of the 2004 paper, and spells the second author of the 2011 paper "Lenan Wu" where Semantic Scholar has "Le-Nan Wu". | The brief specifies this form. The citation can only claim what was verified. |
| C2 | **q = 0.7**, a single value, fixed a priori. **It was not derived from a verified literature value**, because no source's q could be confirmed (C1). Reasons: q < 1 makes (1 − q) > 0, so the coupling term is active and non-trivial; and a value of q away from 1 keeps the criterion clearly different from Kapur. | Honest a-priori choice. Open item: check the source PDFs and state whether 0.7 matches. |
| C3 | **The DP caveat, which matters for the paper.** The true pseudo-additive Tsallis entropy of k+1 independent classes satisfies 1 + (1 − q)·S = Π_c A_c, with A_c = Σ_{i∈c} (p_i/ω_c)^q > 0. That is separable after taking logs, so the DP solves it exactly (`dp_tsallis_pseudoadditive`, checked against full enumeration in `test_tsallis.py`). It equals C1 only at k = 1. So "Tsallis cannot be solved by DP" is true **only** for the C1 form. The study runs C1 as its non-separable case and reports this caveat. | Correctness of the paper's Q3 claim. |
| C4 | **References:** k = 2, 3: exact optimum by vectorised exhaustive search (`bruteforce_tsallis`), with its wall time saved in `results/tsallis_exhaustive.csv`. k = 4, 5: no exact optimum; the reference is the best value found by any of the 210 runs of the cell, labelled **"best known (not proven optimal)"** everywhere. | Exhaustive search is about 2.7 million vectors at k = 3, but about 8.5 × 10⁹ at k = 5. |
| C5 | **Grid:** 24 images × k ∈ {2, 3, 4, 5} × 7 algorithms × 30 seeds = **20,160 runs**, written to `results/runs_tsallis.csv`. The same statistical protocol (G1); at k = 4, 5 gaps are against the best known value. | The brief. |
| C6 | **Hypotheses:** H4: at k = 2, 3, hit rates against the exact optimum are below 100% for every algorithm, as for Otsu and Kapur. H5: at k = 2, 3, one exhaustive solve takes longer than one median optimiser run. If H5 holds, this is the regime where a metaheuristic is a defensible choice. If it fails, it is not, and the report says so. | Stated before the data exist. |

---

## Amendments

*(none yet)*
