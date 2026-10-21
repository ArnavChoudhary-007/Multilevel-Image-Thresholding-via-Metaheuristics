"""
Table 1 timings: exhaustive search vs exact dynamic programming.
Applied Statistics & Stochastic Processes project.

Produces the numbers for Table 1 of the report ON THIS MACHINE, plus the
environment line for its footnote. Run it on the laptop you will quote:
timings depend on the CPU, Python and NumPy versions, so a number measured
anywhere else must not be reported as yours.

What is measured
  - exhaustive search (Otsu), k = 1, 2, 3: measured directly
  - exhaustive search, k = 4, 5, 6, 10: EXTRAPOLATED as C(254, k) * c,
    where c = (time at k=3) / C(254, 3) is the cost per candidate vector.
    This is an optimistic lower bound (it ignores the Python loop growing
    with k), and the table marks these values (e) so nobody mistakes them
    for measurements.
  - exact DP, Otsu and Kapur, k = 1..6 and 10: measured directly

Each measured value is the MEDIAN of repeated runs, with its inter-quartile
range (IQR), so the footnote can state the spread honestly.

Run:  python measure_timing.py
Writes results/timing.csv and prints the table + footnote text.
"""

import csv
import platform
import statistics
import sys
import time
from math import comb
from pathlib import Path

import cv2
import numpy as np
import skimage

from thresholding import load_grey, histogram, bruteforce_multilevel, dp_multilevel

RESULTS = Path("results")
IMAGE = "camera"
DP_KS = [1, 2, 3, 4, 5, 6, 10]
BRUTE_MEASURED = {1: 50, 2: 20, 3: 5}       # k -> number of repeats
BRUTE_EXTRAPOLATED = [4, 5, 6, 10]
DP_REPEATS = 30


def timed(fn, repeats):
    """
    Median and inter-quartile range (seconds) of `repeats` calls, after one
    warm-up call. The median, not the mean: a single background hiccup on a
    millisecond-scale call can double a mean, but barely moves a median.
    """
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    if len(times) < 4:
        return statistics.median(times), 0.0
    q = statistics.quantiles(times, n=4)
    return statistics.median(times), q[2] - q[0]


def human(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 120:
        return f"{seconds:.1f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    if seconds < 86400 * 2:
        return f"{seconds / 3600:.0f} h"
    years = seconds / (86400 * 365.25)
    if years < 1:
        return f"{seconds / 86400:.0f} days"
    return f"{years:,.0f} years"


def environment():
    ram = "RAM unknown (pip install psutil to record it)"
    try:
        import psutil
        ram = f"{psutil.virtual_memory().total / 1e9:.1f} GB RAM"
    except ImportError:
        pass
    cpu = platform.processor() or platform.machine()
    return (f"{cpu}, {ram}, {platform.system()} {platform.release()}, "
            f"Python {platform.python_version()}, NumPy {np.__version__}, "
            f"scikit-image {skimage.__version__}, OpenCV {cv2.__version__}")


def main():
    RESULTS.mkdir(exist_ok=True)
    p = histogram(load_grey(IMAGE))
    rows = []

    print(f"Timing on '{IMAGE}' - this takes a minute or two...\n")

    brute = {}
    for k, reps in BRUTE_MEASURED.items():
        mean, sd = timed(lambda: bruteforce_multilevel(p, k), reps)
        brute[k] = (mean, sd, "m")
        print(f"  brute force k={k}: {human(mean)}")
    c = brute[3][0] / comb(254, 3)
    for k in BRUTE_EXTRAPOLATED:
        brute[k] = (comb(254, k) * c, None, "e")

    for k in DP_KS:
        otsu_mean, otsu_sd = timed(lambda: dp_multilevel(p, k, "otsu"), DP_REPEATS)
        kap_mean, kap_sd = timed(lambda: dp_multilevel(p, k, "kapur"), DP_REPEATS)
        thresholds, _ = dp_multilevel(p, k, "otsu")
        b_mean, b_sd, b_kind = brute[k]
        rows.append({
            "k": k,
            "vectors": comb(254, k),
            "brute_s": b_mean, "brute_sd": b_sd, "brute_kind": b_kind,
            "dp_otsu_s": otsu_mean, "dp_otsu_sd": otsu_sd,
            "dp_kapur_s": kap_mean, "dp_kapur_sd": kap_sd,
            "otsu_thresholds": thresholds,
        })
        print(f"  DP k={k}: Otsu {human(otsu_mean)}, Kapur {human(kap_mean)}")

    with open(RESULTS / "timing.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 100)
    print(f"{'k':>3} {'C(254,k)':>20} {'Exhaustive':>20} {'DP Otsu':>12} "
          f"{'DP Kapur':>12}   Otsu thresholds")
    print("-" * 100)
    worst_sd = 0.0
    for r in rows:
        tag = "(m)" if r["brute_kind"] == "m" else "(e)"
        brute_txt = ("≈ " if tag == "(e)" else "") + human(r["brute_s"]) + f" {tag}"
        print(f"{r['k']:>3} {r['vectors']:>20,} {brute_txt:>20} "
              f"{human(r['dp_otsu_s']):>9} (m) {human(r['dp_kapur_s']):>9} (m)   "
              f"{r['otsu_thresholds']}")
        for mean, sd in ((r["dp_otsu_s"], r["dp_otsu_sd"]),
                         (r["dp_kapur_s"], r["dp_kapur_sd"]),
                         (r["brute_s"], r["brute_sd"])):
            if sd is not None and mean > 0:
                worst_sd = max(worst_sd, sd / mean)
    print("=" * 100)

    print("\nFootnote text for Table 1:")
    print(f"  (m) = measured, median of repeated runs (largest IQR = "
          f"{worst_sd * 100:.0f}% of the median); (e) = extrapolated via "
          f"T_k = C(254,k)·c, c = {c * 1e9:.0f} ns fitted at k = 3, an optimistic "
          f"lower bound.")
    print(f"  Environment: {environment()} - single-threaded; timings bound "
          f"feasibility, not optimality.")
    print(f"\nSaved {RESULTS / 'timing.csv'}")
    print(f"(Python executable: {sys.executable})")


if __name__ == "__main__":
    main()
