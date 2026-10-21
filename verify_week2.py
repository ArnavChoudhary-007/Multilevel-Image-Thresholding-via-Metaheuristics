"""
Week 2 - exact dynamic-programming solver: correctness checks.
Applied Statistics & Stochastic Processes project.

The DP in thresholding.py is the reference point for EVERYTHING that
follows: every metaheuristic's "optimality gap" is measured against it.
If the DP were wrong, every gap in the report would be wrong too. So it
is checked here from four independent directions:

  1. against exhaustive search (Otsu and Kapur, k = 1..3, all 8 images)
  2. against external libraries (OpenCV k=1, scikit-image k=2..4)
  3. against the objective functions themselves (the DP's own score must
     equal otsu_score / kapur_score evaluated at its thresholds)
  4. against basic properties (both DP code paths agree; more thresholds
     never lower the optimum; no random vector beats it)

Run:  python verify_week2.py          (about a minute)
"""

import numpy as np
import cv2
from skimage.filters import threshold_multiotsu

from thresholding import (
    load_grey,
    histogram,
    class_tables,
    otsu_score,
    kapur_score,
    bruteforce_multilevel,
    bruteforce_kapur,
    dp_multilevel,
)

BENCHMARK = ["camera", "coins", "moon", "page", "clock", "astronaut", "coffee", "cell"]

passed, failed, notes = 0, [], []


def check(label, condition, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"  PASS  {label}  {detail}")
    else:
        failed.append(label)
        print(f"  FAIL  {label}  {detail}")


def same_score(a, b, rel=1e-10):
    """Equal up to floating-point rounding, scaled to the objective's size."""
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


print("=" * 72)
print("WEEK 2 - EXACT DP VERIFICATION")
print("=" * 72)

hists = {name: histogram(load_grey(name)) for name in BENCHMARK}

# ------------------------------------------------- 1. vs exhaustive search
print("\n1. DP == exhaustive search (the true optimum), k = 1..3")
for name, p in hists.items():
    for k in (1, 2, 3):
        bt, bs = bruteforce_multilevel(p, k)
        dt, ds = dp_multilevel(p, k, "otsu")
        check(f"{name}: Otsu  k={k}", same_score(bs, ds) and bt == dt,
              f"dp={dt} brute={bt}")

        bt, bs = bruteforce_kapur(p, k)
        dt, ds = dp_multilevel(p, k, "kapur")
        check(f"{name}: Kapur k={k}", same_score(bs, ds) and bt == dt,
              f"dp={dt} brute={bt}")

# ------------------------------------------------- 2. vs external libraries
print("\n2a. DP vs OpenCV Otsu, k = 1")
for name, p in hists.items():
    img = load_grey(name)
    cv_t = int(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    dt, _ = dp_multilevel(p, 1, "otsu")
    check(f"{name}: k=1 vs OpenCV", abs(dt[0] - cv_t) <= 1, f"dp={dt[0]} cv2={cv_t}")

print("\n2b. DP vs scikit-image threshold_multiotsu, k = 2..4")
print("     Rule: identical thresholds, OR (if they differ) the DP's vector must")
print("     score at least as high under OUR objective - the DP is the optimum,")
print("     so a library answer can tie it but never beat it.")
for name, p in hists.items():
    img = load_grey(name)
    # stops at k=4: skimage's routine is itself exhaustive over a lookup
    # table, so k=5 takes minutes per image (k=4 already takes seconds) -
    # a nice illustration of the cost the DP avoids.
    for k in (2, 3, 4):
        sk = tuple(int(t) for t in threshold_multiotsu(img, classes=k + 1))
        dt, ds = dp_multilevel(p, k, "otsu")
        if dt == sk:
            check(f"{name}: k={k} vs skimage", True, f"identical {dt}")
        else:
            sk_score = otsu_score(p, sk)
            ok = ds >= sk_score - 1e-9
            rel = (ds - sk_score) / ds
            notes.append(f"{name} k={k}: dp={dt} ({ds:.6f}) vs skimage={sk} "
                         f"({sk_score:.6f}), relative diff {rel:.2e}")
            check(f"{name}: k={k} vs skimage", ok,
                  f"DIFFERENT dp={dt} {ds:.6f} >= skimage={sk} {sk_score:.6f}")

# ------------------------------------------------- 3. vs objective functions
print("\n3. DP score == objective function evaluated at the DP's thresholds")
for name, p in hists.items():
    for k in (1, 3, 5, 8):
        dt, ds = dp_multilevel(p, k, "otsu")
        check(f"{name}: Otsu  k={k} self-consistent",
              same_score(ds, otsu_score(p, dt)), f"{ds:.6f}")
        dt, ds = dp_multilevel(p, k, "kapur")
        # kapur_score uses ln(q + EPS) with EPS = 1e-12; that shifts each
        # term by ~1e-12, so allow 1e-8 absolute here.
        check(f"{name}: Kapur k={k} self-consistent",
              abs(ds - kapur_score(p, dt)) < 1e-8, f"{ds:.9f}")

# the Kapur table T must agree with a naive sum
p = hists["camera"]
_, _, T = class_tables(p)
lo, hi = 40, 200
seg = p[lo + 1: hi + 1]
naive = (seg[seg > 0] * np.log(seg[seg > 0])).sum()
check("p*ln(p) cumulative table == naive sum", abs((T[hi + 1] - T[lo + 1]) - naive) < 1e-12)

# ------------------------------------------------- 4. properties
print("\n4. Properties")
for name, p in hists.items():
    for obj in ("otsu", "kapur"):
        agree = all(dp_multilevel(p, k, obj, "matrix") == dp_multilevel(p, k, obj, "loop")
                    for k in (1, 2, 4, 7))
        check(f"{name}: {obj} matrix path == loop path", agree)

for obj in ("otsu", "kapur"):
    for name, p in hists.items():
        scores = [dp_multilevel(p, k, obj)[1] for k in range(1, 11)]
        check(f"{name}: {obj} optimum non-decreasing in k (k=1..10)",
              all(a <= b + 1e-9 for a, b in zip(scores, scores[1:])))

rng = np.random.default_rng(0)
for obj, f in (("otsu", otsu_score), ("kapur", kapur_score)):
    p = hists["camera"]
    dt, ds = dp_multilevel(p, 5, obj)
    beaten = any(
        f(p, tuple(sorted(rng.choice(np.arange(1, 255), size=5, replace=False)))) > ds + 1e-8
        for _ in range(5000))
    check(f"camera: {obj} k=5 DP beats 5000 random vectors", not beaten, f"{dt}")

# ------------------------------------------------- summary
print("\n" + "=" * 72)
if notes:
    print("Documented differences from scikit-image (DP scores >= library each time):")
    for n in notes:
        print("   - " + n)
    print()
if failed:
    print(f"RESULT: {passed} passed, {len(failed)} FAILED")
    for f in failed:
        print(f"   - {f}")
else:
    print(f"RESULT: all {passed} checks passed. The exact solver is verified.")
print("=" * 72)
