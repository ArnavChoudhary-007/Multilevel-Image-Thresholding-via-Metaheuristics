"""
Part A Verification: DP vs exhaustive search (k=1..3) for all 16 new benchmark images.
Applied Statistics & Stochastic Processes project.

Confirms that for every new image, the DP's global optimum exactly matches
exhaustive search across all 16 images x k in {1,2,3} x 2 criteria = 96 checks,
plus 8 checks that the original images still load pixel-identically.
Kept separate from verify_week2.py so that suite's 179 checks stay as reported.

Run:  python verify_extended.py
"""

import numpy as np
from skimage import data, img_as_ubyte
from skimage.color import rgb2gray

from thresholding import (
    load_grey,
    histogram,
    bruteforce_multilevel,
    bruteforce_kapur,
    dp_multilevel,
    BENCHMARK_ORIGINAL,
    BENCHMARK_16_NEW,
)

passed, failed = 0, []


def check(label, condition, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"  PASS  {label}  {detail}")
    else:
        failed.append(label)
        print(f"  FAIL  {label}  {detail}")


def same_score(a, b, rel=1e-10):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


print("=" * 72)
print("EXTENDED BENCHMARK (16 NEW IMAGES) - EXACT DP VERIFICATION")
print("=" * 72)

# load_grey gained special cases for the new images; the 8 original images
# must still load pixel-for-pixel as before, or results/runs.csv would no
# longer describe the same inputs. "Before" = the original two-line loader.
print("\n0. Original 8 images load exactly as before the extension")
for name in BENCHMARK_ORIGINAL:
    raw = getattr(data, name)()
    old = img_as_ubyte(rgb2gray(raw) if raw.ndim == 3 else raw)
    check(f"{name:20s}: pixel-identical", np.array_equal(old, load_grey(name)))

hists = {name: histogram(load_grey(name)) for name in BENCHMARK_16_NEW}

print("\n1. Otsu: DP == exhaustive search (k = 1, 2, 3)")
for name, p in hists.items():
    for k in (1, 2, 3):
        bt, bs = bruteforce_multilevel(p, k)
        dt, ds = dp_multilevel(p, k, "otsu")
        check(f"{name:20s}: Otsu  k={k}", same_score(bs, ds) and bt == dt,
              f"dp={dt} brute={bt} s={ds:.2f}")

print("\n2. Kapur: DP == exhaustive search (k = 1, 2, 3)")
for name, p in hists.items():
    for k in (1, 2, 3):
        bt, bs = bruteforce_kapur(p, k)
        dt, ds = dp_multilevel(p, k, "kapur")
        check(f"{name:20s}: Kapur k={k}", same_score(bs, ds) and bt == dt,
              f"dp={dt} brute={bt} s={ds:.4f}")

print("=" * 72)
print(f"RESULT: {passed} checks passed, {len(failed)} failed.")
print("=" * 72)

if failed:
    raise SystemExit(1)
