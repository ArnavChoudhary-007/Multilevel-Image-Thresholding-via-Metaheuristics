"""
Extension Part B - what do the metaheuristics cost at 10-16 bits?
Applied Statistics & Stochastic Processes project.

Why
---
bitdepth_scaling.py showed the exact DP grows as ~L^2: about 2 ms at 8 bit,
35 s to 3 min at 16 bit. A metaheuristic evaluation is O(k) whatever L is,
so somewhere the metaheuristic must become CHEAPER than the exact answer.
This script finds that crossover by measurement, and also measures what
the cheaper answer is worth (gap and hit rate against the DP optimum).

What it does
------------
For L in {256, 1024, 4096, 16384, 65536}, k in {2, 3, 5}, Otsu and Kapur:
  1. builds the same dithered camera histogram as bitdepth_scaling.py;
  2. solves it ONCE with the exact DP, timing it on THIS machine
     (bitdepth.csv holds no optima and was timed on a different machine,
     and a crossover is only meaningful if both sides use the same CPU);
  3. runs each of the 7 optimisers for 10 seeds (0-9) at the usual
     5,000-evaluation budget, searching thresholds in [1, L-2].

Everything runs in ONE process, one job at a time, so timings are not
disturbed by parallel workers. BLAS/OpenMP threads are pinned to 1 below.

Timing convention: the DP time includes building its cumulative tables
(it is a complete solve). The optimiser time is optimise() only; building
its objective (also O(L), once per histogram) is timed separately and
saved, so the report can state it.

Optional --cells3d: one 16-bit fluorescence slice from skimage.data.cells3d
(z = 30, nuclei channel), at its native L = 65,536, k = 2, as a check that
the dithered histograms are not misleading.

Outputs (new files only):
  results/bitdepth_dp_laptop.csv          DP optimum + time per (L, k, objective)
  results/bitdepth_metaheuristics.csv     one row per optimiser run
  results/bitdepth_crossover.csv          per (k, objective, algorithm): median/IQR
                                          time, hit rate, gap by L, and crossover L

Run:  python bitdepth_metaheuristics.py [--smoke] [--cells3d]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # single-threaded timings, before numpy loads

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from bitdepth_scaling import deep_histogram
from optimisers import ALGORITHMS, make_objective, optimise
from run_experiments import HIT_TOL
from thresholding import dp_multilevel

RESULTS = Path("results")
LS = [256, 1024, 4096, 16384, 65536]
KS = [2, 3, 5]
OBJECTIVES = ["otsu", "kapur"]
SEEDS = list(range(10))
BUDGET = 5000

DP_FIELDS = ["source", "L", "k", "objective", "dp_time_s", "dp_repeats",
             "opt_thresholds", "opt_score"]
RUN_FIELDS = ["source", "L", "k", "objective", "algorithm", "seed", "budget",
              "best_score", "opt_score", "gap", "rel_gap", "hit", "thresholds",
              "time_s", "objective_build_s"]


def cells3d_histogram():
    """Native 16-bit histogram of one cells3d slice (z = 30, nuclei channel)."""
    from skimage import data
    img = data.cells3d()[30, 1]
    counts = np.bincount(img.ravel(), minlength=65536)
    return counts / counts.sum()


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def append_row(path, fields, row):
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def dp_reference(source, p, L, k, obj, dp_path, cache):
    """DP optimum + time for one cell; computed once and cached in dp_path."""
    key = (source, L, k, obj)
    if key not in cache:
        reps = 3 if L <= 4096 else 1           # same repeat rule as bitdepth_scaling.py
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            t, s = dp_multilevel(p, k, obj)
            times.append(time.perf_counter() - t0)
        row = {"source": source, "L": L, "k": k, "objective": obj,
               "dp_time_s": f"{min(times):.6f}", "dp_repeats": reps,
               "opt_thresholds": " ".join(map(str, t)), "opt_score": repr(s)}
        append_row(dp_path, DP_FIELDS, row)
        cache[key] = (t, s, min(times))
        print(f"  DP  {source} L={L:6d} k={k} {obj:5s}: {min(times):9.3f} s  {t}", flush=True)
    return cache[key]


def run_grid(source, hist_fn, ls, ks, seeds, dp_path, run_path):
    dp_cache = {(r["source"], int(r["L"]), int(r["k"]), r["objective"]):
                (tuple(int(v) for v in r["opt_thresholds"].split()),
                 float(r["opt_score"]), float(r["dp_time_s"]))
                for r in read_rows(dp_path)}
    done = {(r["source"], int(r["L"]), int(r["k"]), r["objective"], r["algorithm"], int(r["seed"]))
            for r in read_rows(run_path)}
    for L in ls:
        p = hist_fn(L)
        for obj in OBJECTIVES:
            t0 = time.perf_counter()
            f = make_objective(p, obj)
            build = time.perf_counter() - t0
            for k in ks:
                _, opt_s, _ = dp_reference(source, p, L, k, obj, dp_path, dp_cache)
                for algo in ALGORITHMS:
                    for seed in seeds:
                        if (source, L, k, obj, algo, seed) in done:
                            continue
                        t0 = time.perf_counter()
                        thr, score, _ = optimise(algo, f, k, BUDGET, seed)
                        elapsed = time.perf_counter() - t0
                        gap = opt_s - score
                        append_row(run_path, RUN_FIELDS, {
                            "source": source, "L": L, "k": k, "objective": obj,
                            "algorithm": algo, "seed": seed, "budget": BUDGET,
                            "best_score": repr(score), "opt_score": repr(opt_s),
                            "gap": repr(gap), "rel_gap": repr(gap / abs(opt_s)),
                            "hit": int(gap <= HIT_TOL * max(1.0, abs(opt_s))),
                            "thresholds": " ".join(map(str, thr)) if thr else "",
                            "time_s": f"{elapsed:.6f}", "objective_build_s": f"{build:.6f}"})
                print(f"  ran L={L:6d} k={k} {obj:5s}: 7 algorithms x {len(seeds)} seeds",
                      flush=True)


def summarise(dp_path, run_path, out_path):
    """Median/IQR time, hit rate, median gap per cell, and the crossover L."""
    import pandas as pd
    dp = pd.read_csv(dp_path)
    runs = pd.read_csv(run_path)
    g = (runs.groupby(["source", "L", "k", "objective", "algorithm"])
         .agg(median_time_s=("time_s", "median"),
              q25_time_s=("time_s", lambda s: s.quantile(0.25)),
              q75_time_s=("time_s", lambda s: s.quantile(0.75)),
              hit_rate=("hit", "mean"),
              median_rel_gap=("rel_gap", "median"),
              max_rel_gap=("rel_gap", "max"),
              n=("seed", "count"))
         .reset_index()
         .merge(dp[["source", "L", "k", "objective", "dp_time_s"]],
                on=["source", "L", "k", "objective"]))
    g["dp_slower"] = g.dp_time_s > g.median_time_s
    # crossover: smallest MEASURED L at which the DP is slower than the median run
    cross = (g[g.dp_slower].groupby(["source", "k", "objective", "algorithm"]).L.min()
             .rename("crossover_L").reset_index())
    g = g.merge(cross, on=["source", "k", "objective", "algorithm"], how="left")
    g.to_csv(out_path, index=False)

    print("\nCrossover: smallest measured L where the exact DP is slower than the median"
          " optimiser run ('-' = DP faster at every L measured)")
    for (src, k, obj), sub in g.groupby(["source", "k", "objective"]):
        first = sub.drop_duplicates("algorithm").set_index("algorithm").crossover_L
        print(f"  {src:7s} k={k} {obj:5s}: " + ", ".join(
            f"{a} {int(first[a]) if first[a] == first[a] else '-'}" for a in ALGORITHMS))
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Part B: optimisers across bit depths")
    ap.add_argument("--smoke", action="store_true", help="L in {256, 1024}, k = 2, 2 seeds")
    ap.add_argument("--cells3d", action="store_true", help="also run the native 16-bit check")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    tag = "smoke_" if args.smoke else ""
    dp_path = RESULTS / f"{tag}bitdepth_dp_laptop.csv"
    run_path = RESULTS / f"{tag}bitdepth_metaheuristics.csv"
    out_path = RESULTS / f"{tag}bitdepth_crossover.csv"

    ls, ks, seeds = ([256, 1024], [2], [0, 1]) if args.smoke else (LS, KS, SEEDS)
    run_grid("camera", deep_histogram, ls, ks, seeds, dp_path, run_path)
    if args.cells3d and not args.smoke:
        run_grid("cells3d", lambda L: cells3d_histogram(), [65536], [2], SEEDS,
                 dp_path, run_path)
    summarise(dp_path, run_path, out_path)


if __name__ == "__main__":
    main()
