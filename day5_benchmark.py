"""
Day 5 - Benchmark image set + segmentation grid.
Applied Statistics & Stochastic Processes project.

Builds the 8-image testbed that Weeks 2-3 will run every algorithm
against, and produces the two figures you'll reuse in the report:

  1. figures/day5_grid.png       - all 8 images segmented at k=3
  2. figures/day5_k_sweep.png    - one image at k=1,2,3 with its histogram

Also prints a PSNR/SSIM table. Those two metrics are how Week 3 measures
"was this segmentation actually any good?", so it's worth having baseline
numbers from exhaustive search now: they are the best any metaheuristic
could possibly do, i.e. the ceiling your PSO/GA/SA results get compared to.

Run:  python day5_benchmark.py
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from thresholding import (
    load_grey,
    histogram,
    bruteforce_multilevel,
    segment,
    segment_labels,
    quality,
)

FIG_DIR = Path("figures")

# Eight bundled scikit-image photos with deliberately varied histograms:
# multimodal, low-contrast, document, textured, biomedical.
BENCHMARK = [
    "camera",      # classic 'Cameraman' - clean trimodal histogram
    "coins",       # objects on an uneven background
    "moon",        # low contrast, narrow histogram
    "page",        # document / text, strongly bimodal
    "clock",       # smooth gradients
    "astronaut",   # colour -> grey, broad histogram
    "coffee",      # colour -> grey, textured
    "cell",        # biomedical - previews the Week 4 MRI application
]


def main():
    FIG_DIR.mkdir(exist_ok=True)

    print(f"{'image':<14}{'k':>3}{'thresholds':>22}{'PSNR (dB)':>12}{'SSIM':>8}{'time':>8}")
    print("-" * 68)

    rows = []
    for name in BENCHMARK:
        img = load_grey(name)
        p = histogram(img)

        for k in (1, 2, 3):
            t0 = time.time()
            thresh, _ = bruteforce_multilevel(p, k)
            elapsed = time.time() - t0

            seg = segment(img, thresh)
            psnr, ssim = quality(img, seg)

            print(f"{name:<14}{k:>3}{str(thresh):>22}{psnr:>12.2f}{ssim:>8.3f}{elapsed:>7.2f}s")
            rows.append((name, k, thresh, psnr, ssim))
        print()

    # ---- Figure 1: all 8 images at k=3 -------------------------------
    fig, axes = plt.subplots(2, 8, figsize=(22, 6))
    for col, name in enumerate(BENCHMARK):
        img = load_grey(name)
        thresh = next(r[2] for r in rows if r[0] == name and r[1] == 3)

        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(segment_labels(img, thresh), cmap="gray", vmin=0, vmax=255)
        axes[1, col].set_title(f"k=3 {thresh}", fontsize=8)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("original")
    fig.suptitle("Benchmark set - exhaustive-search Otsu at k=3", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "day5_grid.png", dpi=140)
    print(f"Saved {FIG_DIR / 'day5_grid.png'}")

    # ---- Figure 2: k sweep on one image + histogram with cut points ---
    name = "camera"
    img = load_grey(name)
    p = histogram(img)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("original")
    axes[0].axis("off")

    for i, k in enumerate((1, 2, 3), start=1):
        thresh = next(r[2] for r in rows if r[0] == name and r[1] == k)
        axes[i].imshow(segment_labels(img, thresh), cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(f"k={k}  {thresh}")
        axes[i].axis("off")

    fig.suptitle(f"{name}: more thresholds = finer partition", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "day5_k_sweep.png", dpi=140)
    print(f"Saved {FIG_DIR / 'day5_k_sweep.png'}")

    # ---- Figure 3: histogram with the k=3 cut points drawn on ---------
    thresh3 = next(r[2] for r in rows if r[0] == name and r[1] == 3)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(256), p, width=1.0, color="#4c78a8")
    for t in thresh3:
        ax.axvline(t, color="crimson", linestyle="--", linewidth=1.6)
        ax.text(t + 2, ax.get_ylim()[1] * 0.9, str(t), color="crimson", fontsize=9)
    ax.set_title(f"{name}: histogram with k=3 thresholds {thresh3}")
    ax.set_xlabel("Grey level i")
    ax.set_ylabel("p(i)")
    ax.set_xlim(0, 255)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "day5_histogram_cuts.png", dpi=140)
    print(f"Saved {FIG_DIR / 'day5_histogram_cuts.png'}")

    plt.show()


if __name__ == "__main__":
    main()