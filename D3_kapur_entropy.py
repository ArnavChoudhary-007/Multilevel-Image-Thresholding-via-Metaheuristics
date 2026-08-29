"""
Day 3 - Kapur's entropy method from scratch.
Applied Statistics & Stochastic Processes project.

Run:  python day3_kapur.py
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_ubyte
from skimage.color import rgb2gray

FIG_DIR = Path("figures")
EPS = 1e-12  # guards against log(0) for an empty class


def load_grey(img=None):
    if img is None:
        img = data.camera()
    if img.ndim == 3:
        img = rgb2gray(img)
    return img_as_ubyte(img)


def histogram(img):
    counts = np.bincount(img.ravel(), minlength=256)
    p = counts / counts.sum()
    return p


def kapur_entropy(p, t):
    """
    Total entropy for a single threshold t.

    Class 0 = levels [0, t], Class 1 = levels [t+1, 255].

    For each class, renormalise its probabilities so they sum to 1
    within the class (that's what "conditional on being in this class"
    means), then compute Shannon entropy:

        H_c = - sum( (p_i / w_c) * log(p_i / w_c) )   for i in class c

    Total score = H_0 + H_1. Kapur's method picks t that MAXIMISES this.

    EPS is added inside the log to avoid log(0) when a bin has zero
    probability - those terms contribute ~0 to the sum either way,
    so this doesn't distort the result, it just prevents a crash.
    """
    w0 = p[: t + 1].sum()
    w1 = p[t + 1:].sum()

    if w0 <= 0 or w1 <= 0:
        return -np.inf  # degenerate split - never select this t

    p0 = p[: t + 1] / w0
    p1 = p[t + 1:] / w1

    h0 = -np.sum(p0 * np.log(p0 + EPS))
    h1 = -np.sum(p1 * np.log(p1 + EPS))

    return h0 + h1


def kapur_threshold_bruteforce(p):
    scores = np.array([kapur_entropy(p, t) for t in range(255)])
    best_t = int(np.argmax(scores))
    return best_t, scores


def otsu_between_class_variance(p, t):
    """Same as Day 2 - kept here only for the sanity-check comparison."""
    levels = np.arange(256)
    w0 = p[: t + 1].sum()
    w1 = p[t + 1:].sum()
    if w0 == 0 or w1 == 0:
        return 0.0
    mu0 = (levels[: t + 1] * p[: t + 1]).sum() / w0
    mu1 = (levels[t + 1:] * p[t + 1:]).sum() / w1
    return w0 * w1 * (mu0 - mu1) ** 2


def otsu_threshold_bruteforce(p):
    scores = np.array([otsu_between_class_variance(p, t) for t in range(255)])
    return int(np.argmax(scores))


def segment(img, t):
    return np.where(img <= t, 0, 255).astype(np.uint8)


def main():
    FIG_DIR.mkdir(exist_ok=True)

    img = load_grey()
    p = histogram(img)

    kapur_t, kapur_scores = kapur_threshold_bruteforce(p)
    otsu_t = otsu_threshold_bruteforce(p)
    cv_otsu_t = int(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])

    print("--- Day 3: Kapur's entropy method ------------------")
    print(f"Kapur threshold        : {kapur_t}")
    print(f"Otsu threshold (Day 2) : {otsu_t}")
    print(f"OpenCV Otsu (reference) : {cv_otsu_t}")
    print(f"Otsu self-check (±1)    : {abs(otsu_t - cv_otsu_t) <= 1}   <- must be True")
    print(f"Kapur vs Otsu gap       : {abs(kapur_t - otsu_t)} levels")
    print("     (they're two different objectives - some gap is expected,")
    print("      not a bug. A huge gap, e.g. >40, is worth double-checking.)")
    print("-----------------------------------------------------")

    kapur_seg = segment(img, kapur_t)
    otsu_seg = segment(img, otsu_t)

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    ax[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Original")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(kapur_seg, cmap="gray", vmin=0, vmax=255)
    ax[0, 1].set_title(f"Kapur  (t = {kapur_t})")
    ax[0, 1].axis("off")

    ax[1, 0].imshow(otsu_seg, cmap="gray", vmin=0, vmax=255)
    ax[1, 0].set_title(f"Otsu  (t = {otsu_t})")
    ax[1, 0].axis("off")

    ax[1, 1].plot(range(255), kapur_scores, label="Kapur entropy", color="tab:purple")
    ax[1, 1].axvline(kapur_t, color="tab:purple", linestyle="--", alpha=0.6)
    ax[1, 1].axvline(otsu_t, color="tab:orange", linestyle="--", alpha=0.6, label=f"Otsu t={otsu_t}")
    ax[1, 1].set_title("Kapur objective vs. threshold")
    ax[1, 1].set_xlabel("Threshold t")
    ax[1, 1].set_ylabel("H0 + H1")
    ax[1, 1].legend()

    fig.tight_layout()
    out = FIG_DIR / "day3_kapur.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()