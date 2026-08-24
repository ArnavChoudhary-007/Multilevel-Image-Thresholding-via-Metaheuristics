"""
Day 1 - Environment check + first histogram.
Applied Statistics & Stochastic Processes project.

Run:  python day1_histogram.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_ubyte
from skimage.color import rgb2gray

FIG_DIR = Path("figures")


def load_grey(img=None):
    """Return a greyscale uint8 image with values in [0, 255]."""
    if img is None:
        img = data.camera()          # 'Cameraman' - ships with scikit-image
    if img.ndim == 3:
        img = rgb2gray(img)
    return img_as_ubyte(img)


def histogram(img):
    """Return raw counts and the normalised probability distribution p(i).

    p(i) = (# pixels with intensity i) / (total pixels)
    This p is THE probability distribution the whole project is built on.
    """
    counts = np.bincount(img.ravel(), minlength=256)
    p = counts / counts.sum()
    return counts, p


def main():
    FIG_DIR.mkdir(exist_ok=True)

    img = load_grey()
    counts, p = histogram(img)

    levels = np.arange(256)
    mean = (levels * p).sum()
    var = ((levels - mean) ** 2 * p).sum()

    print("--- environment check -------------------------")
    print(f"image shape    : {img.shape}")
    print(f"dtype          : {img.dtype}")
    print(f"min / max      : {img.min()} / {img.max()}")
    print(f"total pixels   : {counts.sum()}")
    print(f"sum of p(i)    : {p.sum():.10f}   <- must be 1.0")
    print(f"mean intensity : {mean:.4f}")
    print(f"total variance : {var:.4f}       <- Otsu will decompose this")
    print("-----------------------------------------------")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Cameraman (greyscale)")
    ax[0].axis("off")

    ax[1].bar(levels, p, width=1.0)
    ax[1].set_title("Normalised histogram  p(i)")
    ax[1].set_xlabel("Grey level i")
    ax[1].set_ylabel("p(i)")
    ax[1].set_xlim(0, 255)

    fig.tight_layout()
    out = FIG_DIR / "day1_histogram.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()