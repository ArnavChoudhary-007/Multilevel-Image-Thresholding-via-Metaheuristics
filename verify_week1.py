"""
Day 6 - Week 1 regression check.
Applied Statistics & Stochastic Processes project.

Re-runs every correctness claim made in Days 1-5 against the SHARED
module (thresholding.py) and prints a pass/fail summary.

Why this file exists: on Day 5 the core functions moved out of the
day-scripts into thresholding.py. A refactor like that can silently
break an earlier result, and you would not notice until Week 3 when
your metaheuristics are optimising a subtly wrong objective. Run this
any time you touch thresholding.py.

Run:  python verify_week1.py
"""

import numpy as np
import cv2
from skimage.filters import threshold_multiotsu

from thresholding import (
    load_grey,
    histogram,
    cumulative_tables,
    otsu_score,
    kapur_score,
    bruteforce_multilevel,
    segment,
    quality,
)

BENCHMARK = ["camera", "coins", "moon", "page", "clock", "astronaut", "coffee", "cell"]

passed, failed = 0, []


def check(label, condition, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"  PASS  {label}  {detail}")
    else:
        failed.append(label)
        print(f"  FAIL  {label}  {detail}")


print("=" * 66)
print("WEEK 1 REGRESSION CHECK")
print("=" * 66)

# ---------------------------------------------------------------- Day 1
print("\nDay 1 - histogram is a probability distribution")
for name in BENCHMARK:
    p = histogram(load_grey(name))
    check(f"{name}: sum p(i) == 1", abs(p.sum() - 1.0) < 1e-9, f"sum={p.sum():.12f}")
    check(f"{name}: p(i) >= 0 everywhere", bool((p >= 0).all()))

# ---------------------------------------------------------------- Day 2
print("\nDay 2 - single-threshold Otsu matches OpenCV")
for name in BENCHMARK:
    img = load_grey(name)
    p = histogram(img)
    mine, _ = bruteforce_multilevel(p, 1)
    cv_t = int(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    check(f"{name}: k=1 vs OpenCV", abs(mine[0] - cv_t) <= 1, f"mine={mine[0]} cv2={cv_t}")

# ---------------------------------------------------------------- Day 3
print("\nDay 3 - Kapur entropy behaves sanely")
p = histogram(load_grey("camera"))
check("finite for a valid split", np.isfinite(kapur_score(p, (100,))))
check("rejects degenerate split (t=255)", kapur_score(p, (255,)) == -np.inf)
check("entropy is non-negative", kapur_score(p, (100,)) >= 0,
      f"H={kapur_score(p, (100,)):.4f}")

# ---------------------------------------------------------------- Day 4
print("\nDay 4 - multilevel Otsu matches skimage (ground truth)")
for name in BENCHMARK:
    img = load_grey(name)
    p = histogram(img)
    for k in (2, 3):
        mine, _ = bruteforce_multilevel(p, k)
        sk = tuple(int(t) for t in threshold_multiotsu(img, classes=k + 1))
        gap = max(abs(a - b) for a, b in zip(mine, sk))
        check(f"{name}: k={k} vs skimage", gap <= 2, f"mine={mine} skimage={sk}")

# ------------------------------------------------- internal consistency
print("\nCross-checks - the pieces agree with each other")
p = histogram(load_grey("camera"))

# the brute-force winner must score at least as well as any random vector
best, best_score = bruteforce_multilevel(p, 3)
rng = np.random.default_rng(0)
beaten = False
for _ in range(2000):
    cand = tuple(sorted(rng.choice(range(1, 255), size=3, replace=False)))
    if otsu_score(p, cand) > best_score + 1e-9:
        beaten = True
        break
check("brute-force optimum beats 2000 random vectors", not beaten,
      f"best={best} score={best_score:.2f}")

# thresholds must come back sorted and in range
check("thresholds sorted", list(best) == sorted(best), f"{best}")
check("thresholds in [1,254]", all(1 <= t <= 254 for t in best))

# O(1) cumulative tables must agree with a naive direct sum
P, S = cumulative_tables(p)
levels = np.arange(256)
lo, hi = 50, 180
w_fast = P[hi + 1] - P[lo + 1]
w_slow = p[lo + 1: hi + 1].sum()
mu_fast = (S[hi + 1] - S[lo + 1]) / w_fast
mu_slow = (levels[lo + 1: hi + 1] * p[lo + 1: hi + 1]).sum() / w_slow
check("cumulative tables == naive sums", abs(w_fast - w_slow) < 1e-12
      and abs(mu_fast - mu_slow) < 1e-9, f"w={w_fast:.8f} mu={mu_fast:.6f}")

# more thresholds must never reduce the objective (nested optimum property)
scores = [bruteforce_multilevel(p, k)[1] for k in (1, 2, 3)]
check("objective non-decreasing in k", scores[0] <= scores[1] <= scores[2],
      " <= ".join(f"{s:.1f}" for s in scores))

# ---------------------------------------------------------------- Day 5
print("\nDay 5 - PSNR improves as k increases")
for name in BENCHMARK:
    img = load_grey(name)
    p = histogram(img)
    psnrs = []
    for k in (1, 2, 3):
        th, _ = bruteforce_multilevel(p, k)
        psnrs.append(quality(img, segment(img, th))[0])
    check(f"{name}: PSNR rises with k", psnrs[0] < psnrs[1] < psnrs[2],
          " -> ".join(f"{v:.2f}" for v in psnrs))

# ---------------------------------------------------------------- summary
print("\n" + "=" * 66)
if failed:
    print(f"RESULT: {passed} passed, {len(failed)} FAILED")
    for f in failed:
        print(f"   - {f}")
    print("\nDo not start Week 2 until these pass.")
else:
    print(f"RESULT: all {passed} checks passed. Week 1 is solid.")
print("=" * 66)