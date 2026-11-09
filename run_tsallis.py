"""
Extension Part C - the seven optimisers on a NON-separable criterion.
Applied Statistics & Stochastic Processes project.

Why this experiment exists
--------------------------
The paper's argument is that for Otsu and Kapur, metaheuristics are not
needed: both criteria are sums of per-class terms, so an exact DP finds the
optimum in milliseconds. A fair paper must also show the other side: a
criterion where the DP does NOT apply. The multilevel Tsallis criterion

    S_q(t) = sum_c S_c + (1 - q) * prod_c S_c

has a product over ALL classes, so it is not a sum of per-class terms and
the DP's recurrence breaks (test_tsallis.py shows a concrete case).

Grid: 24 images x k in {2,3,4,5} x 7 algorithms x 30 seeds = 20,160 runs,
same 5,000-evaluation budget, same seeds 0-29, q fixed in advance (see
EXTENSIONS_DECISIONS.md).

Reference values - and why they differ by k:
  k = 2, 3 : EXACT optimum by exhaustive search (bruteforce_tsallis; up to
             C(254,3) = 2.7 million vectors, vectorised). Its wall time is
             saved too, because "metaheuristic time vs exhaustive time" is
             the honest cost comparison when no DP exists.
  k = 4, 5 : no exact method is run. The analysis uses the best value found
             by ANY of the 210 runs of that cell as a "best known" reference
             and labels it so everywhere. It is computed AFTER the runs, so
             this script leaves opt_score empty for those rows.

Outputs (never touching results/runs.csv):
  results/runs_tsallis.csv           one row per run (resumable)
  results/tsallis_exhaustive.csv     exact optima and exhaustive-search times

Run:
  python run_tsallis.py --smoke              tiny grid -> results/smoke_runs_tsallis.csv
  python run_tsallis.py --workers 4          full grid
"""

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

from optimisers import ALGORITHMS, make_objective, optimise
from run_experiments import CHECKPOINT_EVERY, HIT_TOL, _histogram, done_keys
from thresholding import BENCHMARK_24, bruteforce_tsallis

RESULTS = Path("results")
Q = 0.7                     # fixed in EXTENSIONS_DECISIONS.md (C2) before any run
EXACT_KS = (2, 3)           # exhaustive search is feasible only up to k = 3

FULL = {
    "images": BENCHMARK_24, "ks": [2, 3, 4, 5], "algorithms": ALGORITHMS,
    "seeds": list(range(30)), "budget": 5000,
    "out": RESULTS / "runs_tsallis.csv", "exact": RESULTS / "tsallis_exhaustive.csv",
}
SMOKE = {
    "images": ["camera", "brick"], "ks": [2, 4], "algorithms": ALGORITHMS,
    "seeds": [0, 1], "budget": 1000,
    "out": RESULTS / "smoke_runs_tsallis.csv", "exact": RESULTS / "smoke_tsallis_exhaustive.csv",
}

# same columns as runs.csv, so the same analysis code can read both,
# plus q and the kind of reference ("exact" or "" = best known, filled later)
FIELDS = ["image", "objective", "q", "k", "algorithm", "seed", "budget",
          "best_score", "opt_score", "gap", "rel_gap", "hit", "opt_kind",
          "thresholds", "opt_thresholds", "time_s", "curve"]


def run_one(task):
    """One optimiser run on the Tsallis criterion (top-level for worker processes)."""
    image, k, algorithm, seed, budget, opt_t, opt_s = task
    f = make_objective(_histogram(image), "tsallis", q=Q)
    t0 = time.perf_counter()
    thresholds, score, history = optimise(algorithm, f, k, budget, seed)
    elapsed = time.perf_counter() - t0

    row = {
        "image": image, "objective": "tsallis", "q": Q, "k": k,
        "algorithm": algorithm, "seed": seed, "budget": budget,
        "best_score": repr(score), "opt_score": "", "gap": "", "rel_gap": "",
        "hit": "", "opt_kind": "best_known_pending",
        "thresholds": " ".join(map(str, thresholds)) if thresholds else "",
        "opt_thresholds": "", "time_s": f"{elapsed:.4f}",
        "curve": " ".join(repr(v) for v in history[CHECKPOINT_EVERY - 1::CHECKPOINT_EVERY]),
    }
    if opt_s is not None:
        # identical definitions to run_experiments.py
        gap = opt_s - score
        row.update(opt_score=repr(opt_s), gap=repr(gap), rel_gap=repr(gap / abs(opt_s)),
                   hit=int(gap <= HIT_TOL * max(1.0, abs(opt_s))), opt_kind="exact",
                   opt_thresholds=" ".join(map(str, opt_t)))
    return row


def exact_optima(cfg):
    """Exhaustive optimum per (image, k <= 3); cached in a CSV so restarts skip it."""
    path = cfg["exact"]
    cache = {}
    if path.exists():
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                cache[(r["image"], int(r["k"]))] = (
                    tuple(int(v) for v in r["thresholds"].split()), float(r["score"]))
    todo = [(i, k) for i in cfg["images"] for k in cfg["ks"]
            if k in EXACT_KS and (i, k) not in cache]
    if todo:
        new = not path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["image", "k", "q", "thresholds",
                                               "score", "exhaustive_time_s"])
            if new:
                w.writeheader()
            for image, k in todo:
                t0 = time.perf_counter()
                t, s = bruteforce_tsallis(_histogram(image), k, q=Q)
                dt = time.perf_counter() - t0
                w.writerow({"image": image, "k": k, "q": Q,
                            "thresholds": " ".join(map(str, t)), "score": repr(s),
                            "exhaustive_time_s": f"{dt:.4f}"})
                fh.flush()
                cache[(image, k)] = (t, s)
                print(f"  exhaustive {image:22s} k={k}: {t}  ({dt:.2f} s)", flush=True)
    return cache


def main():
    ap = argparse.ArgumentParser(description="Part C: Tsallis experiment")
    ap.add_argument("--smoke", action="store_true", help="tiny grid for testing")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    args = ap.parse_args()
    cfg = SMOKE if args.smoke else FULL
    RESULTS.mkdir(exist_ok=True)

    # exhaustive search runs single-process, BEFORE the pool starts, so its
    # timings are not disturbed by the optimiser workers
    print(f"Exact Tsallis optima (q = {Q}) by exhaustive search, k in {EXACT_KS}...")
    exact = exact_optima(cfg)

    out = cfg["out"]
    done = {(i, k, a, s) for (i, _o, k, a, s) in done_keys(out)}
    tasks = []
    for image in cfg["images"]:
        for k in cfg["ks"]:
            opt_t, opt_s = exact.get((image, k), (None, None))
            for algorithm in cfg["algorithms"]:
                for seed in cfg["seeds"]:
                    if (image, k, algorithm, seed) not in done:
                        tasks.append((image, k, algorithm, seed, cfg["budget"], opt_t, opt_s))
    total = len(cfg["images"]) * len(cfg["ks"]) * len(cfg["algorithms"]) * len(cfg["seeds"])
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
        pool = Pool(args.workers) if args.workers > 1 else None
        results = pool.imap_unordered(run_one, tasks, chunksize=4) if pool else map(run_one, tasks)
        try:
            for i, row in enumerate(results, 1):
                writer.writerow(row)
                fh.flush()
                if i % 200 == 0 or i == len(tasks):
                    rate = i / (time.time() - t_start)
                    print(f"  {i:,}/{len(tasks):,} runs  (~{(len(tasks) - i) / rate / 60:.0f} min left)",
                          flush=True)
        finally:
            if pool is not None:
                pool.terminate()
    print(f"Saved {out}  ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
