"""
Week 3 - the main experiment: 13,440 optimiser runs vs the exact optimum.
Applied Statistics & Stochastic Processes project.

Grid (paper, section 4.2):
    8 images x k in {2,3,4,5} x 2 objectives x 7 algorithms x 30 seeds
    = 13,440 runs, each with a budget of exactly 5,000 evaluations.

The exact DP runs ONCE per (image, objective, k) cell - it is deterministic,
so repeating it would add nothing - and every run is scored against it:

    gap     = optimum - best score found        (>= 0, always)
    rel_gap = gap / |optimum|                   (scale-free, comparable
                                                 between Otsu and Kapur)
    hit     = 1 if the run found the optimum exactly (to rounding), else 0

The DP never enters the rank-based statistics; it is the ruler, not a
competitor.

Seeds: run number s (0..29) uses seed s for every algorithm and cell.
That makes every single run reproducible from its row in the CSV.

Robustness: each finished run is appended to the CSV immediately and the
file is flushed, so a crash or Ctrl+C loses at most the runs in progress.
Re-running the script SKIPS every run already in the CSV and continues.

Run:
    python run_experiments.py --smoke          tiny grid, ~1 min, results/smoke_runs.csv
    python run_experiments.py                  full grid -> results/runs.csv
    python run_experiments.py --workers 4      same, using 4 CPU cores
"""

import argparse
import csv
import os
import time
from multiprocessing import Pool
from pathlib import Path

from thresholding import load_grey, histogram, dp_multilevel
from optimisers import ALGORITHMS, make_objective, optimise

RESULTS = Path("results")

FULL = {
    "images": ["camera", "coins", "moon", "page", "clock", "astronaut", "coffee", "cell"],
    "ks": [2, 3, 4, 5],
    "objectives": ["otsu", "kapur"],
    "algorithms": ALGORITHMS,
    "seeds": list(range(30)),
    "budget": 5000,
    "out": RESULTS / "runs.csv",
}
SMOKE = {
    "images": ["camera", "moon"],
    "ks": [2, 3],
    "objectives": ["otsu", "kapur"],
    "algorithms": ALGORITHMS,
    "seeds": [0, 1],
    "budget": 1000,
    "out": RESULTS / "smoke_runs.csv",
}

CHECKPOINT_EVERY = 50     # convergence curve: best-so-far every 50 evaluations
HIT_TOL = 1e-9            # relative tolerance for "found the optimum"

FIELDS = ["image", "objective", "k", "algorithm", "seed", "budget",
          "best_score", "opt_score", "gap", "rel_gap", "hit",
          "thresholds", "opt_thresholds", "time_s", "curve"]

_HIST_CACHE = {}          # per worker process: image name -> histogram


def _histogram(name):
    if name not in _HIST_CACHE:
        _HIST_CACHE[name] = histogram(load_grey(name))
    return _HIST_CACHE[name]


def run_one(task):
    """One optimiser run. Top-level so it can run in worker processes."""
    image, objective, k, algorithm, seed, budget, opt_t, opt_s = task
    f = make_objective(_histogram(image), objective)
    t0 = time.perf_counter()
    thresholds, score, history = optimise(algorithm, f, k, budget, seed)
    elapsed = time.perf_counter() - t0

    gap = opt_s - score
    rel_gap = gap / abs(opt_s)
    curve = history[CHECKPOINT_EVERY - 1::CHECKPOINT_EVERY]
    return {
        "image": image, "objective": objective, "k": k,
        "algorithm": algorithm, "seed": seed, "budget": budget,
        "best_score": repr(score), "opt_score": repr(opt_s),
        "gap": repr(gap), "rel_gap": repr(rel_gap),
        "hit": int(gap <= HIT_TOL * max(1.0, abs(opt_s))),
        # None only if every evaluation hit an empty class (never observed
        # after the SA fix, but recorded rather than crashing the grid)
        "thresholds": " ".join(map(str, thresholds)) if thresholds else "",
        "opt_thresholds": " ".join(map(str, opt_t)),
        "time_s": f"{elapsed:.4f}",
        "curve": " ".join(repr(v) for v in curve),
    }


def done_keys(path):
    """Runs already in the CSV, so a restart can skip them."""
    if not path.exists():
        return set()
    with open(path, newline="") as fh:
        return {(r["image"], r["objective"], int(r["k"]), r["algorithm"], int(r["seed"]))
                for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--smoke", action="store_true", help="tiny grid for testing")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    args = ap.parse_args()
    cfg = SMOKE if args.smoke else FULL

    RESULTS.mkdir(exist_ok=True)
    out = cfg["out"]

    # exact optimum for every cell, computed once
    print("Computing exact optima (DP)...")
    optimum = {}
    for image in cfg["images"]:
        p = _histogram(image)
        for objective in cfg["objectives"]:
            for k in cfg["ks"]:
                optimum[(image, objective, k)] = dp_multilevel(p, k, objective)

    done = done_keys(out)
    tasks = []
    for image in cfg["images"]:
        for objective in cfg["objectives"]:
            for k in cfg["ks"]:
                opt_t, opt_s = optimum[(image, objective, k)]
                for algorithm in cfg["algorithms"]:
                    for seed in cfg["seeds"]:
                        if (image, objective, k, algorithm, seed) in done:
                            continue
                        tasks.append((image, objective, k, algorithm, seed,
                                      cfg["budget"], opt_t, opt_s))

    total = (len(cfg["images"]) * len(cfg["objectives"]) * len(cfg["ks"])
             * len(cfg["algorithms"]) * len(cfg["seeds"]))
    print(f"Grid: {total:,} runs | already done: {len(done):,} | to run now: {len(tasks):,}")
    if not tasks:
        print("Nothing to do.")
        return

    new_file = not out.exists()
    t_start = time.time()
    with open(out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        if args.workers > 1:
            pool = Pool(args.workers)
            results = pool.imap_unordered(run_one, tasks, chunksize=4)
        else:
            pool = None
            results = map(run_one, tasks)
        try:
            for i, row in enumerate(results, 1):
                writer.writerow(row)
                fh.flush()
                if i % 50 == 0 or i == len(tasks):
                    rate = i / (time.time() - t_start)
                    left = (len(tasks) - i) / rate
                    print(f"  {i:,}/{len(tasks):,} runs  "
                          f"(~{left / 60:.0f} min left)", flush=True)
        finally:
            if pool is not None:
                pool.terminate()

    print(f"Saved {out}  ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
