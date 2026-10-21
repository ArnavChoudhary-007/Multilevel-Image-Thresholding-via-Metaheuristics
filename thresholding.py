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
# Exact solution by dynamic programming (Otsu AND Kapur, any k)
# ----------------------------------------------------------------------
#
# WHY THIS WORKS (the central idea of the whole project):
#
# Both objectives are a SUM of per-class terms, and each term depends
# only on that class's own two boundaries:
#
#     Otsu :  sigma_b^2 = sum_c  m_c^2 / w_c   -  mu_T^2
#     Kapur:  J         = sum_c  ( ln w_c  -  L_c / w_c )
#
# with  w_c = class mass,  m_c = sum of i*p(i),  L_c = sum of p(i)*ln p(i)
# over the class. (The Kapur form is just  -sum q ln q  with q = p/w
# expanded:  -sum (p/w) ln(p/w) = ln w - (1/w) sum p ln p.)
#
# Because classes are contiguous intervals, the best way to cover levels
# 0..t with j+1 classes is "the best way to cover 0..s with j classes,
# plus one more class (s, t]" for the best s. That is Bellman's principle
# of optimality, and it gives the recurrence
#
#     D[j][t] = max over s < t of ( D[j-1][s] + g(s, t) )
#
# which fills a (k+1) x L table in O(k * L^2) time. For k = 10 that is
# ~650,000 cheap operations instead of C(254,10) ~ 2.6e17 vectors.
#
# NOT NOVEL for Otsu: Merzban & Elbayoumi (2019), ref [28] in the paper.
# The Kapur version uses the identical argument; do not call either new.


def class_tables(p):
    """
    Cumulative tables for O(1) per-class terms, all padded with a leading 0:

        P[i] = sum p(0..i-1)              (mass)
        S[i] = sum j*p(j), j < i          (first moment)
        T[i] = sum p(j) ln p(j), j < i    (for Kapur; 0 ln 0 taken as 0)
    """
    p = np.asarray(p, dtype=np.float64)
    L = p.size
    plogp = np.zeros(L)
    nz = p > 0
    plogp[nz] = p[nz] * np.log(p[nz])
    P = np.concatenate([[0.0], np.cumsum(p)])
    S = np.concatenate([[0.0], np.cumsum(np.arange(L) * p)])
    T = np.concatenate([[0.0], np.cumsum(plogp)])
    return P, S, T


def class_term(objective, tables, lo, hi):
    """
    Per-class term g(lo, hi) for the class (lo, hi] = levels lo+1 .. hi.

    `lo` and `hi` may be numpy arrays (vectorised). A class with zero
    probability mass is invalid and gets -inf, so it can never be chosen.
    """
    P, S, T = tables
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    w = P[hi + 1] - P[lo + 1]
    out = np.full(np.broadcast(lo, hi).shape, -np.inf)
    ok = w > 0
    if objective == "otsu":
        m = S[hi + 1] - S[lo + 1]
        out[ok] = (np.broadcast_to(m, out.shape)[ok] ** 2
                   / np.broadcast_to(w, out.shape)[ok])
    elif objective == "kapur":
        Lc = T[hi + 1] - T[lo + 1]
        wb = np.broadcast_to(w, out.shape)[ok]
        out[ok] = np.log(wb) - np.broadcast_to(Lc, out.shape)[ok] / wb
    else:
        raise ValueError("objective must be 'otsu' or 'kapur'")
    return out


def dp_multilevel(p, k, objective="otsu", method="auto"):
    """
    EXACT global optimum for k thresholds, by dynamic programming.

    Parameters
    ----------
    p : histogram (length L; L = 256 for 8-bit, larger for deeper images)
    k : number of thresholds
    objective : "otsu" or "kapur"
    method : "auto" (matrix for L <= 1024, else loop), "matrix" or "loop".
             Both give identical results; verify_week2.py checks that.

    Returns (thresholds, score) with thresholds a sorted int tuple in
    [1, L-2], using the same class convention as the rest of this file:
    (-1, t1], (t1, t2], ..., (tk, L-1]. The score is on the same scale as
    otsu_score / kapur_score, so the two can be compared directly.
    Returns (None, -inf) if no valid split exists (e.g. k is larger than
    the number of occupied grey levels allows).

    Table meaning:
        D[j, t] = best total of class terms for classes 0..j,
                  where class j ends exactly at level t.
        B[j, t] = where class j-1 ended in that best solution (backpointer).
    """
    p = np.asarray(p, dtype=np.float64)
    L = p.size
    if not 1 <= k <= L - 2:
        raise ValueError(f"k must be between 1 and {L - 2}")
    tables = class_tables(p)

    D = np.full((k + 1, L), -np.inf)
    B = np.full((k + 1, L), -1, dtype=np.int64)

    # class 0 = (-1, t]; its end is the first threshold, so 1 <= t <= L-2
    ends0 = np.arange(1, L - 1)
    D[0, ends0] = class_term(objective, tables, -1, ends0)

    if method == "auto":
        method = "matrix" if L <= 1024 else "loop"
    if method not in ("matrix", "loop"):
        raise ValueError("method must be 'auto', 'matrix' or 'loop'")

    if method == "matrix":
        # FAST PATH (8-bit and 10-bit): precompute every class term once
        # as an L x L matrix G[s, t] = g(s, t), -inf where s >= t, then do
        # each DP layer as one numpy max over a matrix. Same recurrence,
        # same answer as the loop below, ~20x faster in Python.
        s_idx = np.arange(L)[:, None]
        t_idx = np.arange(L)[None, :]
        G = np.full((L, L), -np.inf)
        upper = s_idx < t_idx
        G[upper] = class_term(objective, tables,
                              np.broadcast_to(s_idx, (L, L))[upper],
                              np.broadcast_to(t_idx, (L, L))[upper])
        allowed = np.zeros(L, dtype=bool)
        for j in range(1, k + 1):
            M = D[j - 1][:, None] + G
            allowed[:] = False
            if j == k:
                allowed[L - 1] = True
            else:
                allowed[j + 1: L - 1] = True
            D[j, allowed] = M.max(axis=0)[allowed]
            B[j, allowed] = M.argmax(axis=0)[allowed]
    else:
        # MEMORY-SAFE PATH (12-bit, 16-bit): same recurrence, one column
        # at a time, so memory is O(k*L) instead of O(L^2).
        for j in range(1, k + 1):
            # class j ends at the next threshold, or at L-1 if it is last
            ends = [L - 1] if j == k else range(j + 1, L - 1)
            for t in ends:
                s = np.arange(j, t)          # where class j-1 can end
                cand = D[j - 1, s] + class_term(objective, tables, s, t)
                idx = int(np.argmax(cand))
                D[j, t] = cand[idx]
                B[j, t] = s[idx]

    best = D[k, L - 1]
    if not np.isfinite(best):
        return None, -np.inf

    # walk the backpointers to recover the thresholds
    thresholds = []
    t = L - 1
    for j in range(k, 0, -1):
        t = int(B[j, t])
        thresholds.append(t)
    thresholds = tuple(reversed(thresholds))

    if objective == "otsu":
        mu_T = (np.arange(L) * p).sum()
        best = best - mu_T ** 2
    return thresholds, float(best)


def bruteforce_kapur(p, k):
    """
    Exact Kapur optimum by exhaustive search - ground truth for the DP.

    Same trick as bruteforce_multilevel: loop over the first k-1
    thresholds, vectorise over the last. Practical for k <= 3.
    Uses the separable form, so it shares no DP code but does share
    class_tables (which verify_week2.py checks against kapur_score).
    """
    tables = class_tables(p)
    P, _, T = tables
    Pl, Tl = P.tolist(), T.tolist()       # plain floats: fast scalar maths
    best_score, best_thresh = -np.inf, None

    def g_scalar(lo, hi):
        w = Pl[hi + 1] - Pl[lo + 1]
        if w <= 0:
            return None
        return np.log(w) - (Tl[hi + 1] - Tl[lo + 1]) / w

    for head in combinations(range(1, 255), k - 1):
        bounds = (-1,) + head
        fixed = 0.0
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            term = g_scalar(lo, hi)
            if term is None:
                fixed = -np.inf
                break
            fixed += term
        if not np.isfinite(fixed):
            continue
        last = bounds[-1]
        t = np.arange(last + 1, 255)
        if t.size == 0:
            continue
        scores = (fixed
                  + class_term("kapur", tables, last, t)
                  + class_term("kapur", tables, t, 255))
        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best_score = float(scores[idx])
            best_thresh = head + (int(t[idx]),)

    return best_thresh, best_score


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