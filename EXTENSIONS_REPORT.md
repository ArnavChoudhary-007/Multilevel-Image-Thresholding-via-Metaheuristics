# Extensions report: 24 images, bit depth, and a non-separable criterion

Applied Statistics & Stochastic Processes project, branch `extended`.
Every parameter was fixed in `EXTENSIONS_DECISIONS.md` (commit 0f71109) **before** any extension run.
Every number below was printed by a script in this repo that was actually run; the script is named next to it.
`results/runs.csv` and all earlier figures and tables were not modified.

**Short answers**

| Question | Answer |
|---|---|
| Part A: does the 8-image ranking survive on 24 images? | **Yes.** Kendall's τ between the 8-image and 24-image rank orders is +0.63 to +1.00 in all 10 (criterion, k) conditions. PSO is still first overall for both criteria. |
| Part B: at what bit depth does the exact DP lose to a metaheuristic on time? | **L = 1,024 to 4,096 on this laptop**, depending on the algorithm. At L = 65,536 the DP takes 34–204 s; one optimiser run takes 0.02–0.25 s. Speed is not the whole story: see the quality columns. |
| Part C: is non-separable Tsallis the regime where metaheuristics are justified? | **Only partly.** At k = 2 an exhaustive search (0.016 s) beats nearly every optimiser run. At k = 3 (0.44 s) optimisers are faster but find the exact optimum only 33–94% of the time. From k = 4 there is no exact reference, so the question can no longer be answered by comparison with the optimum. See §3.4 for two caveats that limit this claim. |

---

## 1. What changed in the code (methods)

| Change | File | Effect on the original study |
|---|---|---|
| 16 new images (`BENCHMARK_16_NEW`, `BENCHMARK_24`); `load_grey` handles RGBA, uint16, and fixed frames | `thresholding.py` | The 8 original images load pixel-identically (checked in `verify_extended.py`). |
| Optimisers take a search range `[low, high]`; SA step sd = 8·L/256 | `optimisers.py` | At L = 256 behaviour is identical (`test_extensions.py` re-runs rows of `runs.csv`). |
| Tsallis criterion `tsallis_score/table`, `bruteforce_tsallis` (k ≤ 3), exact pseudo-additive DP | `thresholding.py` | New functions only. |
| `--extended` flag | `run_experiments.py` | The default grid is unchanged. |
| Runners and analysis for Parts A–C | `run_tsallis.py`, `bitdepth_metaheuristics.py`, `analyse_ext.py`, `describe_images.py`, `make_figures_ext.py` | New files. |
| UI shows 24 images and a Tsallis option | `app.py` | UI only. |

`thresholding.py` changed, so the verification suites were re-run after the last change. Final counts: `verify_week1` 56, `verify_week2` 179, `test_optimisers` 134, `verify_extended` 104, `test_extensions` 56, `test_tsallis` 44. **All 573 pass.**

The protocol is unchanged: 5,000 evaluations per run (hard-stopped by `CountedObjective`), seeds 0–29 (Part B: 0–9), per-cell median over seeds as the response, the image as the unit of replication, Friedman + Kendall's W + Nemenyi, 21 Wilcoxon tests per family with Holm correction and rank-biserial r, ART ANOVA.

---

## 2. Part A: 24 images

**Runs.** 13,440 (original) + 26,880 (new) = 40,320, no duplicates, no negative gaps (`analyse_ext.py --part A` → `results/ext24_report.txt`).
**Image table:** `results/ext_image_table.csv` (from `describe_images.py`; histogram descriptions are computed, not hand-written).

### 2.1 Results

- **Friedman is significant in every condition** (10 of 10; the largest p is 5.4e-16, Kapur overall). Kendall's W ranges from 0.58 (Kapur overall) to 0.94 (Otsu k = 2).
- **Critical difference** falls from 3.18 (N = 8) to **1.84** (N = 24). At N = 24, random search is separated from the field in every condition; GA is separated at k ≥ 3; hill climber and DE separate from PSO in a few conditions (Otsu k = 5 and overall: hill climber; Kapur k = 5: DE).
- **Ranking agreement (H1).** Kendall's τ, 8 vs 24 images:

  | | k = 2 | k = 3 | k = 4 | k = 5 | overall |
  |---|---|---|---|---|---|
  | Otsu | +1.00 | +1.00 | +1.00 | +0.90 | +1.00 |
  | Kapur | +0.63 (p = 0.098) | +0.81 | +0.78 | +0.98 | +0.90 |

  The best algorithm changed in two Kapur cells (k = 2: hill climber → SA; k = 4: PSO → DE); in both, the two algorithms are within the critical difference.
- **Are the new images harder?** Hit rates on the 16 new images are close to those on the original 8 (PSO 93% / 88% vs 94% / 93%, Otsu / Kapur). The largest change is GWO on Kapur (58% on the original 8, 67% on the new 16).
- **Hit rate by k, all 24 images** (`ext24_hit_by_k.csv`, `fig_hit_vs_k_24.png`): PSO stays above 73% up to k = 5; DE is best at k ≤ 3 but falls to 31% (Kapur) and 51% (Otsu) at k = 5; GA is at 0–1% at k = 5.

### 2.2 The effective-N problem (this matters for the paper)

The Wilcoxon signed-rank test drops image pairs with a zero difference, and many pairs of good algorithms tie exactly (both reach the optimum on most images). Across the 210 tests, **114 lost at least one image and 59 had fewer than 10 usable images**; with n_eff < 10 the smallest attainable exact p is above Holm's first threshold of 0.05/21, so those tests cannot be significant whatever the data. At k = 2 the median pair has n_eff of only 1–2. So "not significant" at low k between the strong algorithms is a statement about ties, not about evidence. The report should say this next to the Wilcoxon table.

### 2.3 Hypotheses

| | Outcome |
|---|---|
| H1: τ > 0 | **Held** in all 10 conditions (smallest +0.63). |
| H2: Friedman significant everywhere | **Held.** |
| H3: some Wilcoxon–Holm pair p < 0.05 | **Held** (many; mostly pairs involving random search or GA). |

---

## 3. Part C: Tsallis criterion (non-separable, q = 0.7)

**Runs.** 20,160, no duplicates; 0 runs scored above the exact optimum (`analyse_ext.py --part C` → `results/tsallis_report.txt`). Exhaustive references for k = 2, 3 (`results/tsallis_exhaustive.csv`). For k = 4, 5 the reference is the best of the 210 runs in the cell: **best known, not proven optimal**.

### 3.1 Hit rates (`tsallis_hit_by_k.csv`, `fig_tsallis_hit.png`)

| | k = 2 (exact) | k = 3 (exact) | k = 4 (best known) | k = 5 (best known) |
|---|---|---|---|---|
| random search | 17% | 1% | 0% | 0% |
| hill climber | 85% | 80% | 57% | 45% |
| SA | 99% | 86% | 72% | 54% |
| PSO | 100% | 87% | 86% | 77% |
| GA | 98% | 33% | 5% | 0% |
| DE | 100% | 94% | 91% | 38% |
| GWO | 100% | 82% | 55% | 31% |

On the same 24 images the exact-hit rates for Otsu / Kapur at k = 3 are 98% / 94% (PSO) and 100% / 97% (DE): Tsallis hit rates at k = 3 (PSO 87%, DE 94%, SA 86%, GWO 82%) are below the Otsu rates for all four; against Kapur the picture is mixed (SA and GWO are higher on Tsallis, PSO and DE lower). This is a descriptive comparison pooled over images; no test was run.

### 3.2 Statistics

Friedman p ≤ 1.8e-18 in all five families; W = 0.67–0.90. Ranks at k = 5 (best known): PSO 1.67, SA 2.77, GWO 3.44, hill climber 3.54, DE 4.08, GA 5.62, random search 6.88; Nemenyi separates PSO from hill climber, DE, GA and random search. 52 of 105 Wilcoxon–Holm tests are significant; 61 of 105 lost images to zero differences, and 30 had n_eff < 10 (same caveat as §2.2).

### 3.3 Cost of one exhaustive solve vs one optimiser run (H5)

| k | exhaustive solve, median over 24 images | optimiser median run (range over algorithms) | Is a run faster than exhaustive? |
|---|---|---|---|
| 2 | 0.016 s | 21–352 ms | Rarely: 1 of 24 images for SA, PSO, random search; 9 of 24 for hill climber; none for GA, DE, GWO |
| 3 | 0.439 s | 26–371 ms | Yes for all seven (median over images); 24 of 24 images for random search, hill climber, SA, PSO; 15 of 24 for GA, 20 of 24 for DE and GWO |

### 3.4 Answer, with its limits

**Is this the regime where metaheuristics are justified?** The honest answer is "only at k ≥ 3 for this implementation, and the evidence is thinner than it looks":

1. **At k = 2 no.** Exhaustive search is faster and exact.
2. **At k = 3 on speed, yes, but the speed-up is small (1.2×–15×) and between 6% (DE) and 67% (GA) of runs of the six non-random algorithms miss the optimum.** If the user needs the exact answer, exhaustive search at 0.44 s is cheap.
3. **At k = 4, 5 the exact solve is infeasible under the study form** (about 2.7 million candidate vectors at k = 3 grows to about 8.5 × 10⁹ at k = 5; the cost at k = 4, 5 was not measured). There a metaheuristic is the only practical route. But there is **no proven optimum**: a "hit" means "matched the best of 210 runs", and PSO, the best algorithm, matches it in only 77% of k = 5 runs.

**Two caveats that weaken the claim.**

- **The time comparison is not like-for-like.** The exhaustive search is vectorised NumPy and ran single-process; the optimisers are pure Python and ran 4 at a time, so contention inflated their times (this works against the optimisers). Compiled code on either side would change the ratios. The k = 3 ratios are therefore indicative, not definitive.
- **"The DP cannot do Tsallis" is true only for the study's form** S_q = ΣS_c + (1−q)ΠS_c. The pseudo-additive entropy of independent classes satisfies 1 + (1−q)S = Π_c A_c, which is separable after taking logs, so the DP solves it exactly (`dp_tsallis_pseudoadditive`, checked against enumeration in `test_tsallis.py`; the two forms coincide only at k = 1 and differ in 11 of 16 tested cases). The paper's Q3 claim must say which form it means; a reader who knows the pseudo-additive identity will otherwise object.

### 3.5 Hypotheses

| | Outcome |
|---|---|
| H4: at k = 2, 3 every algorithm's hit rate is below 100% | **Partly false.** At k = 2, DE, GWO and PSO found the exact optimum in 720 of 720 runs. At k = 3 it holds (best is DE, 93.6%). |
| H5: at k = 2, 3 one exhaustive solve is slower than a median optimiser run | **False at k = 2, true at k = 3** (see §3.3). |

---

## 4. Part B: bit depth (L up to 65,536)

**Setup.** Dithered camera histograms (`deep_histogram`, same as the existing Table 8) at L ∈ {256, 1,024, 4,096, 16,384, 65,536}, k ∈ {2, 3, 5}, both criteria, 7 algorithms, seeds 0–9 → 2,100 runs, plus a native 16-bit check (`cells3d`, z = 30, nuclei channel, k = 2) → 140 runs. Single process, BLAS threads pinned to 1, one job at a time. The DP optimum and time were measured on this laptop (`results/bitdepth_dp_laptop.csv`); a crossover needs both sides timed on the same CPU.
Outputs: `results/bitdepth_metaheuristics.csv`, `results/bitdepth_crossover.csv`, `fig_bitdepth_crossover.png`, `fig_bitdepth_quality.png`.

### 4.1 Time

| L | DP, Otsu k = 3 | DP, Otsu k = 5 | One run, PSO | One run, DE |
|---|---|---|---|---|
| 256 | 0.002 s | 0.002 s | 0.042 s | 0.162 s |
| 1,024 | 0.048 s | 0.064 s | 0.043 s | 0.168 s |
| 4,096 | 0.35 s | 0.71 s | 0.043 s | 0.165 s |
| 16,384 | 3.2 s | 6.4 s | 0.043 s | 0.166 s |
| 65,536 | 60 s | 110 s | 0.045 s | 0.170 s |

Optimiser time is flat in L because 5,000 evaluations cost the same whatever the histogram length; building the objective grows (0.0002 s at L = 256 to 0.006 s at L = 65,536) and is reported separately. DP time grows with L², as expected. The Kapur DP is slower (k = 5 at 65,536: 204 s).

**Crossover** (smallest *measured* L where the DP is slower than the median optimiser run; no interpolation): hill climber 1,024; PSO 1,024 at k ≥ 3 (4,096 at k = 2); random search, SA, DE and GWO 4,096; GA 4,096 (k ≥ 3) or 16,384 (k = 2). The two criteria give the same crossover in every case. The native 16-bit slice (L = 65,536, k = 2) is slower for the DP (30 s Otsu, 46 s Kapur) than every optimiser (0.04–0.25 s).

### 4.2 Quality: what the faster answer is worth

- **Hill climber collapses as predicted** (it moves ±1): Otsu k = 3 hit rate falls from 100% at L = 256 to 40% at 4,096 and 0% at 16,384; its median relative gap at L = 65,536 is 6.8%. It is fast but not usable at large L.
- **PSO and DE hold the exact optimum at k = 2, 3 for Otsu** (100% at every L). Kapur is harder: at k = 3, L = 65,536 DE hits 60% and PSO 60%; PSO at k = 2 hits 90% at L ≥ 16,384.
- **At k = 5 no algorithm reliably finds the optimum at large L.** Best hit rate at L = 65,536 is 20% (PSO, Otsu) and 0% (Kapur, every algorithm). But the median relative gap of PSO, DE, GWO, SA and GA is 4e-6 to 4e-4, and the worst single run of those five is 1.5e-3: "within 0.04% of optimal at the median, 0.15% at worst". Whether that is acceptable depends on the application; the paper should present it as a trade, not as success.
- **Cheaper answer, not an equal answer.** At L = 65,536, k = 5, the DP gives the exact answer in 110–204 s; DE gives a near-optimal answer in about 0.17 s, roughly 650–1,200× faster (PSO, at 0.045 s, is faster still).

### 4.3 Limits

- The histograms are synthetic dithered versions of one image (camera); only the cells3d slice is a native 16-bit histogram, and only at k = 2.
- 10 seeds per cell. The DP was timed once above L = 4,096 (best of 3 below), so large-L DP times carry some noise.
- The crossover is a property of this DP (pure NumPy, O(k·L²)) on this CPU. A compiled DP, or one exploiting monotone structure to run faster than O(L²) (not tested here), would push the crossover to larger L. Also, two idle Streamlit processes (the project UI) were running during Part B (see Amendments).

---

## 5. What changes in the paper

1. **Q2/Table:** replace the 8-image Nemenyi CD (3.18) with the 24-image CD (1.84) and report τ. State that the ranking was confirmed on 16 further images rather than re-tuned. Report effective N beside every Wilcoxon table (§2.2).
2. **Q3, bit depth:** the DP is not slower than a 5,000-evaluation metaheuristic until L ≈ 1,024–4,096 on this machine, so the "needed at high bit depth" argument holds only for 12-bit and above, and at k = 5 only as a time-for-accuracy trade (§4.2).
3. **Q3, Tsallis:** do not write "the DP cannot solve Tsallis". Write that it cannot solve the sum-plus-product form used here, and that the pseudo-additive form is DP-solvable (§3.4). The metaheuristic case for Tsallis is real only for k ≥ 4, where it cannot be checked against an optimum.
4. **Hypotheses.** H4 partly failed and H5 failed at k = 2; say so.

## 6. Files

| Output | Script |
|---|---|
| `results/runs_extended.csv` (26,880 rows, 52 MB) | `run_experiments.py --extended` |
| `results/runs_tsallis.csv` (20,160 rows), `tsallis_exhaustive.csv` | `run_tsallis.py` |
| `results/bitdepth_metaheuristics.csv`, `bitdepth_dp_laptop.csv`, `bitdepth_crossover.csv` | `bitdepth_metaheuristics.py --cells3d` |
| `results/ext24_*`, `results/tsallis_*` | `analyse_ext.py --part A|C` |
| `results/ext_image_table.csv` | `describe_images.py` |
| `figures/extensions/*.png` (5 figures) | `make_figures_ext.py` |

`results/*.csv` is gitignored, as for `runs.csv`: **keep a copy of the extension CSVs outside the repo.** The text reports (`results/ext24_report.txt`, `tsallis_report.txt`, the run logs) are committed.

## 7. Machine and software

Intel Core i5-1135G7 (4 cores / 8 logical), 15.8 GB RAM, Windows 11 Home (build 26200); Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, scikit-image 0.24.0, pandas 2.3.3, statsmodels 0.14.6, matplotlib 3.9.4. Parts A and C used 4 worker processes; Part B ran single-process.
`runs.csv` was made on Linux with NumPy 2.4.4. Kapur scores can differ from it in the last binary digit (about 1e-15 relative, far below the 1e-9 hit tolerance); thresholds are identical on the rows re-checked (`test_extensions.py`). Part A and C wall-times come from this laptop and are not pooled with the timings in `runs.csv`.

## 8. Not done, and decisions made after seeing results

**Not done / open**
- **Tsallis formula, q and author lists are unverified.** The paper texts could not be fetched (403/429). q = 0.7 is an a-priori choice, not a literature value. Please check the two source PDFs and state whether 0.7 matches.
- The five images downloaded through scikit-image's hosted data (eagle, human_mitosis, nickel_solidification, skin, vortex) were fetched by an earlier session; your approval is not on record. Please confirm they are acceptable.
- No bit-depth view was added to the UI.
- No exact optimum for Tsallis at k ≥ 4 (not feasible); no run of Part B at k = 4.
- Nothing was pushed. The presentation files and `Claude outputs/` were left uncommitted, as instructed.

**Decisions made after seeing results**
- None changed a parameter, an image, a hypothesis or an analysis. Disclosed process events are in the Amendments of `EXTENSIONS_DECISIONS.md`.
- Which findings to highlight in the short answers (e.g. the effective-N problem, the pseudo-additive caveat) was chosen after seeing the output; the numbers themselves are not selected.
