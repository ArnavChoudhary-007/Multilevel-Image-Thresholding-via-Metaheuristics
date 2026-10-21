"""
Q3 - where does the exact method stop being cheap? Bit-depth scaling.
Applied Statistics & Stochastic Processes project.

The DP costs O(k * L^2), where L is the number of grey levels:
    8-bit  L =    256  ->      65,536 L^2
    10-bit L =  1,024  ->   1,048,576
    12-bit L =  4,096  ->  16,777,216
    16-bit L = 65,536  -> 4,294,967,296   (65,536x the 8-bit cost)

This script MEASURES DP time at L = 256, 1024, 4096, 16384 and 65536
(8, 10, 12, 14 and 16 bit), both objectives, k = 2, 3, 5 - nothing is
extrapolated. It also fits the exponent b in  time ~ a * L^b  between the
two largest sizes: b close to 2 confirms the O(L^2) growth (at small L the
Python loop overhead, which is only O(k*L), hides it).

Takes ~10 minutes, almost all of it at 16 bit.

Histograms: a real image cannot be "made" 16-bit, so we build an L-level
histogram from the 8-bit camera image by spreading each pixel uniformly
within its grey-level bin (dithering) and re-quantising to L levels. The
DP's running time depends only on L and k, not on the histogram's shape,
so this is a fair timing input; the thresholds it returns are not
interpreted.

Run:  python bitdepth_scaling.py
Writes results/bitdepth.csv and figures/bitdepth.png
"""

import csv
import time
from pathlib import Path

import numpy as np

from thresholding import load_grey, dp_multilevel

RESULTS, FIGURES = Path("results"), Path("figures")
LS = [256, 1024, 4096, 16384, 65536]
KS = [2, 3, 5]
TARGET_L = {"8-bit": 256, "10-bit": 1024, "12-bit": 4096, "14-bit": 16384, "16-bit": 65536}


def deep_histogram(L, seed=0):
    img = load_grey("camera").astype(np.float64)
    rng = np.random.default_rng(seed)
    x = (img + rng.uniform(-0.5, 0.5, img.shape) + 0.5) / 256.0     # in (0, 1)
    q = np.clip((x * L).astype(int), 0, L - 1)
    counts = np.bincount(q.ravel(), minlength=L)
    return counts / counts.sum()


def best_time(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)          # min = least disturbed by other processes


def human(s):
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 120:
        return f"{s:.1f} s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    rows = []
    print(f"{'L':>6} {'k':>3} {'objective':>9} {'path':>7} {'time':>10}")
    for L in LS:
        p = deep_histogram(L)
        for k in KS:
            for obj in ("otsu", "kapur"):
                path = "matrix" if L <= 1024 else "loop"
                reps = 3 if L <= 4096 else 1
                t = best_time(lambda: dp_multilevel(p, k, obj, path), reps)
                rows.append({"L": L, "k": k, "objective": obj, "path": path,
                             "seconds": t, "kind": "measured"})
                print(f"{L:>6} {k:>3} {obj:>9} {path:>7} {human(t):>10}", flush=True)

    # growth exponent between the two largest sizes, per (k, objective)
    print("\nGrowth exponent b in time ~ L^b between L = 16,384 and 65,536:")
    for k in KS:
        for obj in ("otsu", "kapur"):
            t = {r["L"]: r["seconds"] for r in rows if r["k"] == k and r["objective"] == obj}
            b = np.log(t[65536] / t[16384]) / np.log(65536 / 16384)
            ratio = t[65536] / t[256]
            print(f"  k={k} {obj:<5} b = {b:.2f};  16-bit takes {human(t[65536])}, "
                  f"{ratio:,.0f}x the 8-bit time")

    with open(RESULTS / "bitdepth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {RESULTS / 'bitdepth.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {2: "#2a78d6", 3: "#eb6834", 5: "#1baf7a"}
        markers = {2: "o", 3: "s", 5: "^"}
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for k in KS:
            meas = [r for r in rows if r["k"] == k and r["objective"] == "otsu"]
            ax.plot([r["L"] for r in meas], [r["seconds"] for r in meas],
                    marker=markers[k], color=colors[k], lw=2, ms=6, label=f"k = {k}")
        Lr = np.array([256, 65536], dtype=float)
        ref = [r for r in rows if r["k"] == 2 and r["objective"] == "otsu" and r["L"] == 65536][0]
        ax.plot(Lr, ref["seconds"] * (Lr / 65536) ** 2, color="#8a8984", ls=":", lw=1.2,
                label="slope of L² (reference)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(list(TARGET_L.values()))
        ax.set_xticklabels([f"{k}\nL={v:,}" for k, v in TARGET_L.items()], fontsize=8)
        ax.set_ylabel("DP time, Otsu (s, log scale)")
        ax.set_title("Exact DP time vs bit depth (all points measured)",
                     fontsize=10, color="#0b0b0b")
        ax.grid(True, which="major", color="#e5e5e2", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "bitdepth.png", dpi=160)
        print(f"Saved {FIGURES / 'bitdepth.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
