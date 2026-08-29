"""
Day 2 - Otsu's method from scratch + validation against OpenCV.
Applied Statistics & Stochastic Processes project.

Run:  python day2_otsu.py
"""

from pathlib import Path

import cv2
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
    """Return the normalised probability distribution p(i) over grey levels."""
    counts = np.bincount(img.ravel(), minlength=256)
    p = counts / counts.sum()
    return p


def otsu_between_class_variance(p, t):
    """
    Between-class variance for a single threshold t.

    Class 0 = levels [0, t], Class 1 = levels [t+1, 255].

        w0, w1   = class probabilities  (sum of p(i) in each class)
        mu0, mu1 = class means          (weighted by p(i))
        sigma_b^2 = w0 * w1 * (mu0 - mu1)^2

    Returns 0.0 for a degenerate split (an empty class) so it can never
    be picked as the maximum.
    """
    levels = np.arange(256)

    w0 = p[: t + 1].sum()
    w1 = p[t + 1:].sum()

    if w0 == 0 or w1 == 0:
        return 0.0

    mu0 = (levels[: t + 1] * p[: t + 1]).sum() / w0
    mu1 = (levels[t + 1:] * p[t + 1:]).sum() / w1

    return w0 * w1 * (mu0 - mu1) ** 2


def otsu_threshold_bruteforce(p):
    """Try every t in [0, 254] and return the one maximising between-class variance."""
    scores = np.array([otsu_between_class_variance(p, t) for t in range(255)])
    best_t = int(np.argmax(scores))
    return best_t, scores


def segment(img, t):
    """Binary segmentation: pixels <= t -> 0, pixels > t -> 255."""
    out = np.where(img <= t, 0, 255).astype(np.uint8)
    return out


def main():
    FIG_DIR.mkdir(exist_ok=True)

    img = load_grey()
    p = histogram(img)

    # --- our from-scratch Otsu -------------------------------------
    my_t, scores = otsu_threshold_bruteforce(p)

    # --- OpenCV's built-in Otsu, for validation ----------------------
    cv_t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv_t = int(cv_t)

    print("--- Day 2: Otsu's method ------------------------")
    print(f"my threshold      : {my_t}")
    print(f"OpenCV threshold  : {cv_t}")
    print(f"match (±1)        : {abs(my_t - cv_t) <= 1}   <- must be True")
    print("---------------------------------------------------")

    my_seg = segment(img, my_t)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(my_seg, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title(f"My Otsu  (t = {my_t})")
    ax[1].axis("off")

    ax[2].plot(range(255), scores)
    ax[2].axvline(my_t, color="red", linestyle="--", label=f"t = {my_t}")
    ax[2].set_title("Between-class variance vs. threshold")
    ax[2].set_xlabel("Threshold t")
    ax[2].set_ylabel(r"$\sigma^2_{between}$")
    ax[2].legend()

    fig.tight_layout()
    out = FIG_DIR / "day2_otsu.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()