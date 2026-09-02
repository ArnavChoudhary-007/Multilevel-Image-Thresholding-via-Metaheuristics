"""
Day 4 - Multilevel thresholding (k=2, 3) + validation against skimage.
Applied Statistics & Stochastic Processes project.

This is the most important file of Week 1. It generalises Day 2's
single-threshold Otsu to any number of thresholds, brute-forces the true
optimum for k=2 and k=3, and validates against skimage's trusted
implementation. That ground-truth check is what you reuse in Week 2 to
prove your metaheuristics find the RIGHT answer, not just "an" answer.

PERFORMANCE NOTE (worth understanding, it's a real algorithmic idea):
A naive implementation recomputes each class's probability mass and mean
by summing over its grey levels every time - so scoring one candidate
costs O(256), and k=3 costs 2.7M x 256 operations. Instead we precompute
two cumulative-sum tables ONCE:

    P[i] = sum of p(0..i)          (cumulative probability)
    S[i] = sum of j*p(j) for j=0..i (cumulative first moment)

Then any class spanning (lo, hi] has:
    w  = P[hi] - P[lo]          <- O(1) instead of O(256)
    mu = (S[hi] - S[lo]) / w    <- O(1)

Same maths, same answer, ~100x faster. This is a standard trick and it's
worth a sentence in your report.

Run:  python day4_multilevel.py
"""

import time
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import threshold_multiotsu

FIG_DIR = Path("figures")


def load_grey(img=None):
    if img is None:
        img = data.camera()
    if img.ndim == 3:
        img = rgb2gray(img)
    return img_as_ubyte(img)


def histogram(img):
    counts = np.bincount(img.ravel(), minlength=256)
    return counts / counts.sum()


def cumulative_tables(p):
    """
    Precompute the two lookup tables described in the module docstring.

    Both are padded with a leading 0 so that P[hi+1] - P[lo+1] cleanly
    gives the mass of levels lo+1 .. hi with no special-casing at 0.
    """
    levels = np.arange(256)
    P = np.concatenate([[0.0], np.cumsum(p)])          # length 257
    S = np.concatenate([[0.0], np.cumsum(levels * p)])  # length 257
    return P, S


def otsu_score_multilevel(P, S, thresholds, total_mean):
    """
    Generalised Otsu between-class variance, O(k) via cumulative tables.

    `thresholds` is a sorted sequence of cut points, e.g. (87, 176).
    Classes are (-1, t1], (t1, t2], ..., (tk, 255].

        sigma_b^2 = sum over classes of  w_c * (mu_c - mu_total)^2

    Returns -1 for a degenerate split (an empty class), so such a vector
    can never win.
    """
    bounds = (-1,) + tuple(thresholds) + (255,)
    score = 0.0

    for lo, hi in zip(bounds[:-1], bounds[1:]):
        w = P[hi + 1] - P[lo + 1]
        if w <= 0:
            return -1.0
        mu = (S[hi + 1] - S[lo + 1]) / w
        score += w * (mu - total_mean) ** 2

    return score


def bruteforce_multilevel(p, k):
    """
    Try every combination of k sorted, distinct thresholds; return the best.

    k=1: 255 combinations        (instant)
    k=2: ~32,000                 (well under a second)
    k=3: ~2.7 million            (a few seconds with cumulative tables)
    k=4: ~172 million            (impractical - this is why Week 2 exists)
    """
    P, S = cumulative_tables(p)
    total_mean = (np.arange(256) * p).sum()

    best_score = -np.inf
    best_thresh = None

    for combo in combinations(range(1, 255), k):
        s = otsu_score_multilevel(P, S, combo, total_mean)
        if s > best_score:
            best_score = s
            best_thresh = combo

    return best_thresh, best_score


def segment_multilevel(img, thresholds):
    """Map each pixel to its class, spread evenly over [0,255] for display."""
    bounds = [-1] + list(thresholds) + [255]
    out = np.zeros_like(img)
    n_classes = len(bounds) - 1
    for idx, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        level = int(round(255 * idx / (n_classes - 1))) if n_classes > 1 else 255
        out[(img > lo) & (img <= hi)] = level
    return out


def main():
    FIG_DIR.mkdir(exist_ok=True)

    img = load_grey()
    p = histogram(img)

    results = {}

    for k in (2, 3):
        print(f"\n--- k = {k} -----------------------------------------")
        t0 = time.time()
        my_thresh, my_score = bruteforce_multilevel(p, k)
        elapsed = time.time() - t0

        sk_thresh = tuple(int(t) for t in threshold_multiotsu(img, classes=k + 1))
        max_gap = max(abs(a - b) for a, b in zip(my_thresh, sk_thresh))

        print(f"my thresholds         : {my_thresh}   (brute force, {elapsed:.1f}s)")
        print(f"skimage thresholds    : {sk_thresh}")
        print(f"max per-threshold gap : {max_gap}")
        print(f"match (gap <= 2)      : {max_gap <= 2}   <- must be True")

        results[k] = my_thresh

    print("\n--- why brute force stops at k=3 --------------------")
    for k in range(1, 7):
        combos, n = 1, 254
        for i in range(k):
            combos = combos * (n - i) // (i + 1)
        print(f"k={k}: {combos:,} combinations to check")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))

    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(segment_multilevel(img, results[2]), cmap="gray", vmin=0, vmax=255)
    ax[1].set_title(f"k=2  thresholds={results[2]}")
    ax[1].axis("off")

    ax[2].imshow(segment_multilevel(img, results[3]), cmap="gray", vmin=0, vmax=255)
    ax[2].set_title(f"k=3  thresholds={results[3]}")
    ax[2].axis("off")

    fig.tight_layout()
    out = FIG_DIR / "day4_multilevel.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    plt.show()


if __name__ == "__main__":
    main()