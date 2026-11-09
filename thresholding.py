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

# Benchmark image lists. The 16 extension images were fixed before any
# extension experiment ran; selection rules are in EXTENSIONS_DECISIONS.md.
# Five of them (eagle, human_mitosis, nickel_solidification, skin, vortex)
# are not bundled with scikit-image: skimage.data downloads them once via
# pooch, hash-checked, into the local cache.
BENCHMARK_ORIGINAL = [
    "camera", "coins", "moon", "page", "clock", "astronaut", "coffee", "cell"
]

BENCHMARK_16_NEW = [
    "brick", "chelsea", "eagle", "grass", "gravel", "hubble_deep_field",
    "human_mitosis", "immunohistochemistry", "microaneurysms", "nickel_solidification",
    "retina", "rocket", "skin", "stereo_motorcycle", "text", "vortex"
]

BENCHMARK_24 = BENCHMARK_ORIGINAL + BENCHMARK_16_NEW


# ----------------------------------------------------------------------
# Images and histograms
# ----------------------------------------------------------------------

def load_grey(name="camera"):
    """
    Load a scikit-image sample image as uint8 greyscale.

    The 8 original images go through exactly the same path as before
    (colour -> rgb2gray -> img_as_ubyte); verify_extended.py checks they are
    pixel-identical, so results/runs.csv stays valid. Three of the extension
    images are multi-frame, so one frame is fixed in advance:
      stereo_motorcycle -> left view,  nickel_solidification -> frame 0,
      vortex -> frame 0.
    16-bit frames are reduced to 8 bits by dropping the low byte (>> 8).
    """
    if name == "stereo_motorcycle":
        img = data.stereo_motorcycle()[0]
    elif name == "nickel_solidification":
        img = data.nickel_solidification()[0]
    elif name == "vortex":
        img = data.vortex()[0]
    else:
        img = getattr(data, name)()
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = rgb2gray(img)
        return img_as_ubyte(img)
    elif img.ndim == 2:
        if img.dtype in (np.float32, np.float64):
            return img_as_ubyte(img)
        elif img.dtype == np.uint16:
            return (img >> 8).astype(np.uint8)
        else:
            return img.astype(np.uint8)
    raise ValueError(f"Unsupported image shape {img.shape} for {name}")


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
# Multilevel Tsallis entropy (Part C - non-separable criterion)
# ----------------------------------------------------------------------
#
# Tsallis entropy for thresholding goes back to Portes de Albuquerque et al.
# (2004, Pattern Recognition Letters 25(9):1059-1065). The multilevel form
# used here,
#
#     S_q(t) = sum_c S_c + (1 - q) * prod_c S_c,
#
# is the one specified for this study (see EXTENSIONS_DECISIONS.md). Which
# exact form each published paper uses has NOT been verified from full text.
#
# WHY IT IS NOT SEPARABLE: the product term couples every class, so the best
# place for one threshold depends on the entropies of ALL the other classes.
# The DP's recurrence "best prefix + one more class" no longer holds.
#
# CAUTION (important for the paper): the TRUE pseudo-additive rule for many
# independent subsystems is  1 + (1-q) S = prod_c (1 + (1-q) S_c),  which
# expands to sum + pairwise products + triple products + ... For k = 1 (two
# classes) it equals the form above; for k >= 2 it does not. That true form
# IS solvable by DP, because 1 + (1-q) S_c = A_c = sum_{i in c} (p_i/w_c)^q > 0
# and maximising prod_c A_c is maximising sum_c ln A_c (for q < 1).
# See dp_tsallis_pseudoadditive below.

def tsallis_score(p, thresholds, q=0.7):
    """
    Multilevel Tsallis criterion S_q = sum_c S_c + (1 - q) prod_c S_c.
    Reference implementation: a plain loop, used to check the fast versions.

    Parameters
    ----------
    p : normalised histogram over [0, L-1]
    thresholds : sorted sequence of thresholds in [1, L-2]
    q : entropic index (q != 1; default 0.7)

    Returns
    -------
    S_q(t) = sum_c S_c + (1 - q) * prod_c S_c
    where S_c = (1 - sum_{i in C_c} (p(i)/omega_c)^q) / (q - 1)
    and omega_c = sum_{i in C_c} p(i).
    Returns -inf if any class has omega_c <= 0.
    """
    bounds = (-1,) + tuple(sorted(thresholds)) + (len(p) - 1,)
    if abs(q - 1.0) < 1e-12:
        return kapur_score(p, thresholds)

    s_classes = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg = p[lo + 1: hi + 1]
        w = seg.sum()
        if w <= 0:
            return -np.inf
        pos = seg[seg > 0]
        if pos.size == 0:
            return -np.inf
        sum_pq = np.sum((pos / w) ** q)
        s_c = (1.0 - sum_pq) / (q - 1.0)
        s_classes.append(s_c)

    return sum(s_classes) + (1.0 - q) * float(np.prod(s_classes))


def tsallis_table(p, q=0.7):
    """
    Class Tsallis entropy S_c for EVERY class, as an L x L table:
    table[lo, hi] = S_c of the class covering grey levels lo..hi inclusive
    (-inf if that class has zero mass, or hi < lo).

    Uses  sum_i (p_i / w)^q = (sum_i p_i^q) / w^q.  Each row is accumulated
    from its own start level (cumsum of p[lo:]) rather than as a difference
    of two global cumulative sums: the difference form loses ~8 digits to
    cancellation on small classes late in the histogram (seen as a 3.5e-8
    mismatch on 'coins' in test_tsallis.py); this form does not, at the
    same O(L^2) cost. L = 256 -> 65,536 entries, built once per image.
    """
    p = np.asarray(p, dtype=np.float64)
    L = len(p)
    pq = np.zeros(L, dtype=np.float64)
    pos = p > 0
    pq[pos] = p[pos] ** q

    table = np.full((L, L), -np.inf, dtype=np.float64)
    for lo in range(L):
        w = np.cumsum(p[lo:])
        a = np.cumsum(pq[lo:])
        valid = w > 0
        hi_idx = np.where(valid)[0] + lo
        table[lo, hi_idx] = (1.0 - a[valid] / w[valid] ** q) / (q - 1.0)
    return table


def bruteforce_tsallis(p, k, q=0.7):
    """
    Exact exhaustive search for multilevel Tsallis entropy (k in {1, 2, 3}).
    Vectorised over the final threshold for speed.
    """
    if k not in (1, 2, 3):
        raise ValueError("bruteforce_tsallis is practical only for k in {1, 2, 3}")

    L = len(p)
    table = tsallis_table(p, q)
    best_thresh, best_score = None, -np.inf

    if k == 1:
        s0 = table[0, 1:L - 1]
        s1 = table[2:L, L - 1]
        valid = np.isfinite(s0) & np.isfinite(s1)
        scores = np.full_like(s0, -np.inf)
        if np.any(valid):
            scores[valid] = s0[valid] + s1[valid] + (1.0 - q) * s0[valid] * s1[valid]
        idx = int(np.argmax(scores))
        return (idx + 1,), float(scores[idx])

    if k == 2:
        for t1 in range(1, L - 2):
            s0 = table[0, t1]
            if not np.isfinite(s0):
                continue
            s1 = table[t1 + 1, t1 + 1:L - 1]
            s2 = table[t1 + 2:L, L - 1]
            valid = np.isfinite(s1) & np.isfinite(s2)
            if not np.any(valid):
                continue
            scores = np.full_like(s1, -np.inf)
            scores[valid] = s0 + s1[valid] + s2[valid] + (1.0 - q) * s0 * s1[valid] * s2[valid]
            idx = int(np.argmax(scores))
            if scores[idx] > best_score:
                best_score = float(scores[idx])
                best_thresh = (t1, t1 + 1 + idx)
        return best_thresh, best_score

    if k == 3:
        for t1 in range(1, L - 3):
            s0 = table[0, t1]
            if not np.isfinite(s0):
                continue
            for t2 in range(t1 + 1, L - 2):
                s1 = table[t1 + 1, t2]
                if not np.isfinite(s1):
                    continue
                s2 = table[t2 + 1, t2 + 1:L - 1]
                s3 = table[t2 + 2:L, L - 1]
                valid = np.isfinite(s2) & np.isfinite(s3)
                if not np.any(valid):
                    continue
                scores = np.full_like(s2, -np.inf)
                scores[valid] = s0 + s1 + s2[valid] + s3[valid] + (1.0 - q) * s0 * s1 * s2[valid] * s3[valid]
                idx = int(np.argmax(scores))
                if scores[idx] > best_score:
                    best_score = float(scores[idx])
                    best_thresh = (t1, t2, t2 + 1 + idx)
        return best_thresh, best_score


def tsallis_pa_score(p, thresholds, q=0.7):
    """
    TRUE pseudo-additive Tsallis entropy of the k+1 classes:

        1 + (1-q) S = prod_c A_c,   A_c = sum_{i in c} (p_i / w_c)^q,
        so  S = (prod_c A_c - 1) / (1 - q).

    Reference loop implementation (checks dp_tsallis_pseudoadditive).
    Equals tsallis_score when k = 1 (two classes); differs for k >= 2.
    """
    bounds = (-1,) + tuple(sorted(thresholds)) + (len(p) - 1,)
    prod = 1.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg = p[lo + 1: hi + 1]
        w = seg.sum()
        if w <= 0:
            return -np.inf
        prod *= np.sum((seg[seg > 0] / w) ** q)
    return (prod - 1.0) / (1.0 - q)


def dp_tsallis_pseudoadditive(p, k, q=0.7):
    """
    EXACT optimum of the true pseudo-additive Tsallis entropy, by DP.

    Why the DP applies: for q < 1 the factor 1/(1-q) is positive, so
    maximising S is maximising prod_c A_c, i.e. maximising sum_c ln A_c -
    a SUM of per-class terms again. With cumulative tables of p and p^q,
    ln A_c = ln(sum_{i in c} p_i^q) - q ln w_c is O(1) per class, and the
    same recurrence as dp_multilevel gives the optimum in O(k L^2).
    (For q > 1 the sign flips and one would minimise sum ln A_c instead.)

    Returns (thresholds, S) with S from tsallis_pa_score's formula.
    """
    if not q < 1:
        raise ValueError("implemented for q < 1 (the regime used in this study)")
    p = np.asarray(p, dtype=np.float64)
    L = p.size
    pq = np.zeros(L)
    nz = p > 0
    pq[nz] = p[nz] ** q
    P = np.concatenate([[0.0], np.cumsum(p)])
    PQ = np.concatenate([[0.0], np.cumsum(pq)])

    # G[s, t] = ln A for the class (s, t], -inf where empty or s >= t
    s_idx = np.arange(L)[:, None]
    t_idx = np.arange(L)[None, :]
    w = P[t_idx + 1] - P[s_idx + 1]
    a = PQ[t_idx + 1] - PQ[s_idx + 1]
    G = np.full((L, L), -np.inf)
    ok = (s_idx < t_idx) & (w > 0) & (a > 0)
    G[ok] = np.log(np.broadcast_to(a, (L, L))[ok]) - q * np.log(np.broadcast_to(w, (L, L))[ok])
    # class 0 = (-1, t]
    w0, a0 = P[1:], PQ[1:]
    first = np.full(L, -np.inf)
    ok0 = (w0 > 0) & (a0 > 0)
    first[ok0] = np.log(a0[ok0]) - q * np.log(w0[ok0])

    D = np.full((k + 1, L), -np.inf)
    B = np.full((k + 1, L), -1, dtype=np.int64)
    D[0, 1:L - 1] = first[1:L - 1]
    for j in range(1, k + 1):
        M = D[j - 1][:, None] + G
        cols = [L - 1] if j == k else list(range(j + 1, L - 1))
        D[j, cols] = M.max(axis=0)[cols]
        B[j, cols] = M.argmax(axis=0)[cols]
    if not np.isfinite(D[k, L - 1]):
        return None, -np.inf
    thresholds, t = [], L - 1
    for j in range(k, 0, -1):
        t = int(B[j, t])
        thresholds.append(t)
    thresholds = tuple(reversed(thresholds))
    return thresholds, float((np.exp(D[k, L - 1]) - 1.0) / (1.0 - q))


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