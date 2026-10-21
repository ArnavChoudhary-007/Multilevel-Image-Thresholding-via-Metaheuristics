# CLAUDE.md: ASSP Multilevel Thresholding Project

Course project for Applied Statistics & Stochastic Processes (DSU, CSE 3rd sem, 2026-27).
Owner: Arnav (GitHub: ArnavChoudhary-007). Read this whole file before writing code.

## What the project is

**Title:** "Is Metaheuristic Optimisation Necessary for Multilevel Image Thresholding?
An Exact-Baseline Evaluation with Rigorous Statistical Comparison"

**Problem:** given a greyscale image with normalised histogram p(i), i = 0..255, choose
k thresholds t = (t1 < ... < tk), each in [1, 254], that maximise J(t), which is either
Otsu's between-class variance or Kapur's summed class entropy.

**The argument has three research questions:**

- **Q1, Establish:** is the exact optimum reachable? Yes. Both objectives are additively
  separable over classes, so exact dynamic programming solves them in O(k·L²).
- **Q2, Measure:** what does approximation cost? Run metaheuristics against the exact
  optimum and measure the optimality gap and hit rate, with proper statistics.
- **Q3, Bound:** where is approximation still needed? Two cases: as bit depth grows
  (L = 1024, 4096, 65536), and for criteria the DP can't handle (Tsallis, 2-D histograms,
  spatial terms).

**Paper contributions:**

- (i) Reproduce and extend the exact result.
- (ii) Document the adoption gap.
- (iii) Run an equal-budget statistical comparison.
- (iv) Analyse Simulated Annealing empirically as a Markov chain.

## Environment

- Windows, with a project venv at `.venv` (Python 3.9). Run scripts with `.venv\Scripts\python <script>.py`.
- Dependencies are in `requirements.txt`. numpy, scipy, pandas, scikit-image, opencv,
  statsmodels, scikit-posthocs, matplotlib, seaborn and mealpy (3.0.2) are already installed.
- Git repo on branch `main`. Commits use the style "Day N: short description".
- `figures/*.png` and `results/*.csv` are gitignored, because they are generated outputs.

## Code map (state as of 13 Oct 2026)

| File | Purpose |
|---|---|
| `thresholding.py` | **Shared core library. All new code imports from here.** Has `load_grey`, `histogram`, `cumulative_tables`, `otsu_score`, `kapur_score`, `bruteforce_multilevel` (Otsu only; vectorised over the last threshold; practical up to k=3), `segment` (class-mean reconstruction), `segment_labels` (display only) and `quality` (PSNR, SSIM). |
| `verify_week1.py` | Regression suite with 56 checks: Otsu vs OpenCV (k=1) and vs skimage `threshold_multiotsu` (k=2,3), Kapur sanity checks, cumulative tables vs naive sums. **Re-run it after any change to `thresholding.py`.** |
| `day5_benchmark.py` | Defines the 8-image benchmark `BENCHMARK = [camera, coins, moon, page, clock, astronaut, coffee, cell]` and makes the PSNR/SSIM baseline figures. |
| `Multilevel.py`, `D2_otsu.py`, `D3_kapur_entropy.py`, `day1_histogram.py` | Week 1 day-scripts. They keep their own older copies of the functions. Historical only: don't import from them and don't refactor them. |

**Conventions to keep:**

- Thresholds are a sorted tuple of ints in [1, 254].
- Classes are (-1, t1], (t1, t2], ..., (tk, 255].
- A degenerate split (empty class) returns -1.0 for Otsu and -inf for Kapur.
- Per-class quantities come from `cumulative_tables` in O(1).
- Every script starts with a docstring explaining the why, in the existing teaching style.

## What is NOT done yet (do these in order)

### 1. Exact DP in `thresholding.py`, for both objectives

It is not in this repo yet. It was prototyped elsewhere and has to be rebuilt here.

- Separable form:
  - Otsu: σ_b² = Σ_c m_c²/w_c − μ_T²
  - Kapur: H_c = ln w_c − (Σ_{i∈c} p_i ln p_i)/w_c. Add a cumulative table of p·ln p.
- Recurrence: F[j][t] = max_{s<t} F[j−1][s] + g(s,t), with backtracking. O(k·L²).
- Take L as a parameter (default 256) so it also works for bit-depth scaling.
- Add tests to `verify_week1.py`, or a new `verify_week2.py`:
  - DP matches `bruteforce_multilevel` exactly for k=1..3 on all 8 images (Otsu).
  - Add a Kapur brute force for k≤2 (or k≤3) and check the Kapur DP against it.
  - DP matches skimage `threshold_multiotsu` for k up to 5.
  - The DP score equals `otsu_score`/`kapur_score` evaluated at the DP's thresholds.
    `kapur_score` uses ln(q + EPS), so compare with a small tolerance and treat terms
    with p_i = 0 as 0 in the p·ln p table.
- Honesty: the DP for Otsu is **not novel** (Merzban & Elbayoumi 2019, ref [28] in the paper).
  We have not confirmed whether the Kapur DP is novel. Never call either one new.

### 2. `measure_timing.py`

Wall-clock time for brute force (k=1..3) and DP (k=1..10) on this machine. Record the
CPU, Python version and numpy version. These numbers go into the footnote of Table 1.

### 3. `optimisers.py`

- One interface: `optimise(objective, k, budget, seed) -> (thresholds, score, history)`.
- `objective` is a closure over p. Wrap it in an **evaluation counter**: it hard-stops at
  `budget` calls and logs the best-so-far value at each evaluation for convergence curves.
- Write SA and PSO by hand: they are what the stochastic-process analysis needs.
  The other algorithms can come from mealpy, but each must be forced through the same
  counter and budget.
- Repair step: round, clip to [1, 254], sort, de-duplicate. Seed everything from the
  `seed` argument (`np.random.default_rng`).
- SA must be able to log every proposal as (iteration, temperature, current, proposed,
  accepted) for item 6.

### 4. Optimiser tests

- No optimiser ever beats the DP optimum by more than 1e-9.
- Evaluation count is never above the budget.
- Same seed gives identical output.
- On easy cases (k=2), each optimiser usually reaches the optimum.

### 5. `run_experiments.py`

- Grid: images × objectives × k values × algorithms × seeds. This must total **13,440 runs**,
  the number in the methodology. **Ask Arnav for the exact factor values** (algorithm list,
  k values, seeds, budget) before running. Don't guess them.
- Compute the DP optimum once per (image, objective, k) and cache it.
- CSV columns: image, objective, k, algorithm, seed, budget, best_score, opt_score, gap,
  rel_gap, hit, thresholds, time_s.
- Append each row as it finishes, to `results/runs.csv`.
- Make it resumable by skipping rows already in the CSV.

### 6. `analyse_stats.py`

- **Unit of replication = image, not run.** First aggregate over seeds (mean gap and hit
  rate per image × objective × k × algorithm).
- Pairwise Wilcoxon signed-rank tests with Holm correction (statsmodels `multipletests`).
- Friedman test, then Nemenyi post-hoc (scikit-posthocs), plus a critical-difference diagram.
- Report effect sizes, not just p-values.

### 7. `sa_markov.py`

Treat SA as a time-inhomogeneous Markov chain. Measure:

- empirical acceptance rate against temperature
- time to first hit of the optimum
- state revisit frequency
- how stationary the best-so-far value is

### 8. `bitdepth_scaling.py` (Q3)

Run the DP with L = 256, 1024, 4096 and 65536 on synthetic or rescaled histograms.
Fit and plot time against L² and against k.

### 9. `make_figures.py`

- Gap vs k for each algorithm
- Hit-rate heatmap
- Convergence curves
- Critical-difference diagram
- Bit-depth plot

Write all outputs to `figures/`.

## Non-negotiable rules

- Equal function-evaluation budget for every metaheuristic. Never compare by iterations.
- Never invent, round up or "expected-value" any result, timing or p-value. Every number
  in the report must come from a script in this repo that was actually run.
- Keep `thresholding.py` as the single source of truth. Run the verification suite after
  every change to it, and don't commit if it fails.
- Small, focused commits in the existing "Day N: ..." style. Ask before any git push.
- Arnav is a student presenting this in vivas. Explain the why in docstrings and comments,
  because he has to be able to defend every line.

## Non-code items still open (for context only)

- Verify references [1]–[27] in the paper, and complete [29].
- Fill the placeholder rows in Table 3.
- Write the Results, Discussion and Conclusion once the experiments are done.
