"""
Shared core library for the multilevel thresholding project.

Everything reusable lives here so the day-scripts (and, from Week 2 on,
the metaheuristics and the experiment harness) all call ONE validated
implementation instead of each keeping its own copy.

Import it like:
    from thresholding import load_grey, histogram, bruteforce_multilevel
"""

from itertools import combinations

import numpy as np
from skimage import data, img_as_ubyte
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

EPS = 1e-12


# ----------------------------------------------------------------------
# Images and histograms
# ----------------------------------------------------------------------

def load_grey(name="camera"):
    """Load one of scikit-image's bundled sample images as uint8 greyscale."""
    img = getattr(data, name)()
    if img.ndim == 3:
        img = rgb2gray(img)
    return img_as_ubyte(img)


def histogram(img):
    """Normalised histogram p(i): a probability distribution over grey levels."""
    counts = np.bincount(img.ravel(), minlength=256)
    return counts / counts.sum()


def cumulative_tables(p):
    """
    Precompute cumulative probability and first-moment tables.

        P[i] = sum of p(0 .. i-1)
        S[i] = sum of j*p(j) for j = 0 .. i-1

    Both padded with a leading 0 so a class spanning (lo, hi] is just:
        w  = P[hi+1] - P[lo+1]
        mu = (S[hi+1] - S[lo+1]) / w
    i.e. O(1) per class instead of O(256). Same maths, far less work.
    """
    levels = np.arange(256)
    P = np.concatenate([[0.0], np.cumsum(p)])
    S = np.concatenate([[0.0], np.cumsum(levels * p)])
    return P, S


# ----------------------------------------------------------------------
# Objective functions
# ----------------------------------------------------------------------

def otsu_score(p, thresholds):
    """
    Generalised Otsu between-class variance for any number of thresholds.

        sigma_b^2 = sum over classes of  w_c * (mu_c - mu_total)^2

    Returns -1.0 for a degenerate split (an empty class).
    """
    P, S = cumulative_tables(p)
    mu_T = (np.arange(256) * p).sum()
    bounds = (-1,) + tuple(sorted(thresholds)) + (255,)

    score = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        w = P[hi + 1] - P[lo + 1]
        if w <= 0:
            return -1.0
        mu = (S[hi + 1] - S[lo + 1]) / w
        score += w * (mu - mu_T) ** 2
    return score


def kapur_score(p, thresholds):
    """
    Generalised Kapur entropy: sum of each class's Shannon entropy,
    computed on probabilities renormalised within that class.

    Returns -inf for a degenerate split.
    """
    bounds = (-1,) + tuple(sorted(thresholds)) + (255,)

    total = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg = p[lo + 1: hi + 1]
        w = seg.sum()
        if w <= 0:
            return -np.inf
        q = seg / w
        total += -np.sum(q * np.log(q + EPS))
    return total


# ----------------------------------------------------------------------
# Exhaustive search (ground truth for k <= 3)
# ----------------------------------------------------------------------

def bruteforce_multilevel(p, k):
    """
    Exact optimum for k Otsu thresholds by exhaustive search.

    Implementation note: we loop over the first k-1 thresholds in Python
    and VECTORISE over the last one with numpy. That cuts the number of
    Python-level iterations by a factor of ~254 versus looping over every
    full combination, which is what makes k=3 take seconds rather than
    minutes.

    Uses the identity
        sigma_b^2 = sum_c (m_c^2 / w_c)  -  mu_total^2
    so each class contributes an independent term that vectorises cleanly.
    """
    P, S = cumulative_tables(p)
    mu_T = (np.arange(256) * p).sum()

    def g_scalar(lo, hi):
        w = P[hi + 1] - P[lo + 1]
        if w <= 0:
            return None
        m = S[hi + 1] - S[lo + 1]
        return m * m / w

    best_score = -np.inf
    best_thresh = None

    # candidate values for the final threshold, handled as a numpy array
    for head in combinations(range(1, 255), k - 1):
        bounds = (-1,) + head
        fixed = 0.0
        ok = True
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            term = g_scalar(lo, hi)
            if term is None:
                ok = False
                break
            fixed += term
        if not ok:
            continue

        last_fixed = bounds[-1]
        t = np.arange(last_fixed + 1, 255)
        if t.size == 0:
            continue

        w_mid = P[t + 1] - P[last_fixed + 1]
        m_mid = S[t + 1] - S[last_fixed + 1]
        w_end = P[256] - P[t + 1]
        m_end = S[256] - S[t + 1]

        valid = (w_mid > 0) & (w_end > 0)
        if not valid.any():
            continue

        scores = np.full(t.shape, -np.inf)
        scores[valid] = (
            fixed
            + m_mid[valid] ** 2 / w_mid[valid]
            + m_end[valid] ** 2 / w_end[valid]
        )

        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best_score = scores[idx]
            best_thresh = head + (int(t[idx]),)

    return best_thresh, best_score - mu_T ** 2


# ----------------------------------------------------------------------
# Segmentation and quality metrics
# ----------------------------------------------------------------------

def segment(img, thresholds):
    """
    Replace every pixel with its CLASS MEAN intensity.

    This is the standard reconstruction used in the thresholding
    literature: it makes the segmented image directly comparable to the
    original, so PSNR and SSIM are meaningful quality measures.
    """
    bounds = [-1] + list(sorted(thresholds)) + [255]
    out = np.zeros_like(img, dtype=np.float64)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (img > lo) & (img <= hi)
        if mask.any():
            out[mask] = img[mask].mean()
    return out.astype(np.uint8)


def segment_labels(img, thresholds):
    """Class indices spread evenly over [0,255] - for display only."""
    bounds = [-1] + list(sorted(thresholds)) + [255]
    out = np.zeros_like(img)
    n = len(bounds) - 1
    for idx, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        out[(img > lo) & (img <= hi)] = int(round(255 * idx / (n - 1))) if n > 1 else 255
    return out


def quality(original, segmented):
    """PSNR (dB) and SSIM between the original and its segmented version."""
    return (
        peak_signal_noise_ratio(original, segmented, data_range=255),
        structural_similarity(original, segmented, data_range=255),
    )