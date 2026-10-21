# ASSP project: completion report, 14 Oct 2026

All coding tasks are done, the full experiment has been run, and the final report is written: `ASSP_Final_Report.docx`, with a PDF copy. Every number in the report comes from a script in this folder.

## What was done

| Task | Files | Result |
|---|---|---|
| Exact DP, Otsu + Kapur | `thresholding.py`, `verify_week2.py` | 179/179 checks pass |
| Timing (report Table 3) | `measure_timing.py` | DP 3.0 ms at k = 10; exhaustive estimate ≈ 3,600 years |
| Seven optimisers | `optimisers.py`, `test_optimisers.py` | 134/134 checks pass |
| Landscape, k = 2 (Table 4) | `landscape_k2.py` | Review 1's Table 5 reproduced exactly; sign test p = 0.0078 |
| Full experiment | `run_experiments.py` → `results/runs.csv` | 13,440 runs, no duplicates, no negative gaps |
| Statistics (Tables 5–6) | `analyse_stats.py` → `results/stats_*.csv`, `stats_report.txt` | Friedman, Nemenyi, Wilcoxon + Holm, ART, PSNR/SSIM |
| SA as a Markov chain (Table 7) | `sa_markov.py` | Metropolis rule confirmed to within 0.0024 |
| Bit depth, Q3 (Table 8) | `bitdepth_scaling.py` | measured up to 16-bit: 35 s to 3 min per solve |
| Figures | `make_figures.py` and the scripts above → `figures/` | 9 new figures |
| References | in the report | [29] completed (authors, year); [25] and [28] DOIs added |
| Final report | `ASSP_Final_Report.docx` / `.pdf` | 13 pages |

`verify_week1.py` still passes 56/56.

## Main results (detail in the report)

- **Q1:** for 8-bit images the exact optimum costs about 2 ms, for both criteria and any k tested.
- **Q2:** no metaheuristic is reliably exact.
  - PSO, the best, found the optimum in 93.5% (Otsu) and 92.8% (Kapur) of runs.
  - GA found it in about a third of runs; random search in under 5%.
  - Every method was less reliable at k = 5 than at k = 2.
  - Each run took 11–220 times longer than the DP.
- **Statistics:**
  - Friedman is significant in every condition. PSO ranks best, but Nemenyi separates it only from GA and random search.
  - With 8 images, Wilcoxon + Holm cannot reach significance by construction, and the report explains why.
- **Q3:** at 16-bit the DP takes 35 s to 3 min per solve. That is where approximation becomes defensible.

## Changes since the 13 Oct report

1. **Simulated Annealing bug fix.** On narrow-histogram images (clock), SA could start from a vector with an empty class and never move. It now starts from a valid vector, and a regression test was added. All 1,920 SA runs were redone with the fix; other algorithms' results were unaffected. Appendix A of the report records this.
2. **`measure_timing.py` reports the median, not the mean.** The mean was too sensitive to one-off slowdowns.
3. **`run_experiments.py`** records an empty result instead of crashing.
4. **New scripts:** `landscape_k2.py`, `analyse_stats.py`, `sa_markov.py`, `bitdepth_scaling.py`, `make_figures.py`.

## Before you submit (please check)

1. **Timings were measured on the cloud machine, not your laptop.** The report says so in Table 3's caption. To quote your own machine:
   - run `.venv\Scripts\python measure_timing.py` and update Table 3;
   - optionally also run `bitdepth_scaling.py` (about 10 minutes).
2. **Reference [29]** still needs its volume and article number from the publisher's page. Table 1 entries marked "not verified" need full-text access.
3. **The name "AshItosh Kumar"** is copied exactly from Review 1. Correct it if that was a typo.
4. **Grey Wolf Optimiser** is the "modern swarm method". Change it now if you prefer another.
5. **Re-run the three check scripts on your laptop** and confirm they all pass.

## Running everything (`.venv\Scripts\python <script>`)

| Order | Script | Time |
|---|---|---|
| 1 | `verify_week1.py`, `verify_week2.py`, `test_optimisers.py` | checks, ~1–2 min each |
| 2 | `run_experiments.py --workers 4` | ~15–60 min; resumable |
| 3 | `analyse_stats.py` | ~3 min |
| 4 | `make_figures.py`, `landscape_k2.py`, `sa_markov.py` | ~1 min each |
| 5 | `measure_timing.py` | ~1 min |
| 6 | `bitdepth_scaling.py` | ~10 min |

`results/` and `figures/` are not tracked by git (see `.gitignore`), so commit the code and the report. Keep `results/runs.csv` somewhere safe, since the analysis depends on it.
