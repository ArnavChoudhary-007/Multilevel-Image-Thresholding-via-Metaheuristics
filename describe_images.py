"""
Extension Part A - the 24-image benchmark, described by measurement.
Applied Statistics & Stochastic Processes project.

Every number in the image table of the report comes from here, including
the one-line histogram description. The description is COMPUTED, not
written by eye, so it cannot be argued with in a viva:

  modes  : peaks of the histogram after Gaussian smoothing (sd = 3 grey
           levels), counting only peaks with prominence >= 10% of the
           tallest smoothed bin. Smoothing removes single-bin noise.
  skew   : sample skewness of the grey levels ("right-skewed" if > 0.5,
           "left-skewed" if < -0.5, else "roughly symmetric").
  dark   : share of pixels in the darkest 10% of the occupied range.

Run:  python describe_images.py
Writes results/ext_image_table.csv and prints a Markdown table.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import skew

from thresholding import BENCHMARK_16_NEW, BENCHMARK_ORIGINAL, histogram, load_grey

DOWNLOADED = {"eagle", "human_mitosis", "nickel_solidification", "skin", "vortex"}
NOTE = {"stereo_motorcycle": "left view", "nickel_solidification": "frame 0, uint16 >> 8",
        "vortex": "frame 0"}


def describe(img):
    p = histogram(img)
    sm = gaussian_filter1d(p, 3.0)
    peaks, _ = find_peaks(sm, prominence=0.10 * sm.max())
    g = img.ravel().astype(float)
    sk = float(skew(g))
    shape = "right-skewed" if sk > 0.5 else "left-skewed" if sk < -0.5 else "roughly symmetric"
    lo, hi = int(img.min()), int(img.max())
    dark = float((img <= lo + 0.1 * (hi - lo)).mean())
    n = len(peaks)
    modes = {1: "unimodal", 2: "bimodal", 3: "trimodal"}.get(n, f"{n} modes")
    text = f"{modes} (peaks at {', '.join(map(str, peaks))}), {shape}, {dark:.0%} of pixels in darkest tenth"
    return n, sk, dark, text


def main():
    rows = []
    for name in BENCHMARK_ORIGINAL + BENCHMARK_16_NEW:
        img = load_grey(name)
        n, sk, dark, text = describe(img)
        rows.append({
            "image": name,
            "set": "original" if name in BENCHMARK_ORIGINAL else "new",
            "source": "skimage.data (downloaded once via pooch, hash-checked)"
                      if name in DOWNLOADED else "skimage.data (bundled)",
            "note": NOTE.get(name, ""),
            "height": img.shape[0], "width": img.shape[1],
            "levels": int(np.count_nonzero(histogram(img))),
            "min": int(img.min()), "max": int(img.max()),
            "mean": round(float(img.mean()), 1), "median": int(np.median(img)),
            "modes": n, "skewness": round(sk, 2), "dark_share": round(dark, 3),
            "description": text,
        })
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/ext_image_table.csv", index=False)

    print("| # | Image | Set | Source | Size (h x w) | Levels | Range | Histogram (computed) |")
    print("|---|---|---|---|---|---:|---|---|")
    for i, r in enumerate(df.itertuples(index=False), 1):
        src = "downloaded" if r.image in DOWNLOADED else "bundled"
        name = f"{r.image} ({r.note})" if r.note else r.image
        print(f"| {i} | {name} | {r.set} | {src} | {r.height} x {r.width} | {r.levels} | "
              f"{r.min}-{r.max} | {r.description} |")
    print("\nSaved results/ext_image_table.csv")


if __name__ == "__main__":
    main()
