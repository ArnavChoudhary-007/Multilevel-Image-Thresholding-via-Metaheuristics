"""
Extension checks: the optimiser changes must not alter a single 8-bit result,
and the general-L search (Part B) must behave correctly.
Applied Statistics & Stochastic Processes project.

Why this matters: optimisers.py was changed so it can search [1, L-2] for
any L. results/runs.csv (13,440 runs) was produced by the OLD code. If the
new code drew even one random number differently at L = 256, the old and
new results could not be combined. So:

  1. Regression: re-run a fixed sample of rows from results/runs.csv (two
     per algorithm x objective, chosen by a fixed rule) and require the SAME
     thresholds and the same best score (to 1e-12; see the comment below).
  2. General L: at L = 1024, 4096 and 65536 every optimiser returns a valid
     sorted vector inside [1, L-2], never exceeds the budget, reads L from
     the objective, and never beats the exact DP optimum.
  3. SA's step size scales as 8 * L / 256 (and is exactly 8 at L = 256).

Run:  python test_extensions.py
"""

import math

import numpy as np
import pandas as pd

import optimisers
from bitdepth_scaling import deep_histogram
from optimisers import ALGORITHMS, make_objective, optimise
from thresholding import dp_multilevel, histogram, load_grey

passed, failed = 0, []


def check(label, condition, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"  PASS  {label}  {detail}")
    else:
        failed.append(label)
        print(f"  FAIL  {label}  {detail}")


print("=" * 72)
print("EXTENSIONS: REGRESSION AND GENERAL-L CHECKS")
print("=" * 72)

# ---------------------------------------------------------------- 1
print("\n1. Bit-identical to results/runs.csv at L = 256")
runs = pd.read_csv("results/runs.csv")
# fixed rule (no cherry-picking): for every algorithm x objective, the first
# row in file order with k = 3 and the first with k = 5
sample = pd.concat([runs[runs.k == k].groupby(["algorithm", "objective"]).head(1)
                    for k in (3, 5)])
hist_cache = {}
for r in sample.itertuples(index=False):
    if r.image not in hist_cache:
        hist_cache[r.image] = histogram(load_grey(r.image))
    f = make_objective(hist_cache[r.image], r.objective)
    t, s, _ = optimise(r.algorithm, f, int(r.k), int(r.budget), int(r.seed))
    same_t = " ".join(map(str, t)) == r.thresholds
    # runs.csv was produced on a different machine (Linux, NumPy 2.4.4); the
    # last binary digit of a Kapur score can differ between NumPy builds
    # (the ORIGINAL committed optimisers.py shows the same last-digit change
    # on this laptop), so the score is compared to 1e-12 relative - far below
    # the 1e-9 hit tolerance. The thresholds must match exactly.
    rel = abs(s - float(r.best_score)) / max(1.0, abs(s))
    check(f"{r.image:9s} {r.objective:5s} k={r.k} {r.algorithm:13s} seed={r.seed}",
          same_t and rel <= 1e-12, f"t={t} score diff = {rel:.1e}")

# ---------------------------------------------------------------- 2
print("\n2. General L: valid vectors, budget, L read from the objective, no DP beaten")
for L in (1024, 4096, 65536):
    p = deep_histogram(L)
    f = make_objective(p, "otsu")
    check(f"L={L}: objective knows its L", f.L == L)
    opt_t, opt_s = dp_multilevel(p, 2, "otsu") if L <= 4096 else (None, None)
    for algo in ALGORITHMS:
        t, s, hist = optimise(algo, f, 2, 300, seed=0)
        ok = (len(hist) == 300 and t is not None and list(t) == sorted(set(t))
              and 1 <= t[0] and t[-1] <= L - 2)
        if opt_s is not None:
            ok = ok and s <= opt_s + 1e-9 * max(1.0, abs(opt_s))
        check(f"L={L:5d} {algo:13s}", ok, f"t={t}")
# a vector above 254 proves the search really left the 8-bit range
p = deep_histogram(4096)
t, _, _ = optimise("random_search", make_objective(p, "otsu"), 3, 200, seed=0)
check("L=4096: search reaches thresholds above 254", max(t) > 254, f"t={t}")

# ---------------------------------------------------------------- 3
print("\n3. SA step size scales with L")
seen = []
real_normal = np.random.Generator.normal


class Spy:
    """Records the sd SA asks for, by wrapping the generator it is given."""

    def __init__(self, rng):
        self.rng = rng

    def __getattr__(self, name):
        return getattr(self.rng, name)

    def normal(self, loc, scale, size=None):
        seen.append(scale)
        return self.rng.normal(loc, scale, size)


for L, expected in ((256, 8.0), (4096, 128.0), (65536, 2048.0)):
    seen.clear()
    f = optimisers.CountedObjective(make_objective(deep_histogram(L), "otsu"), 100)
    try:
        optimisers.sa(f, 2, Spy(np.random.default_rng(0)), 100, low=1, high=L - 2, L=L)
    except optimisers.BudgetExhausted:
        pass
    check(f"L={L}: SA step sd = {expected}", seen and all(math.isclose(s, expected) for s in seen),
          f"sd used = {sorted(set(seen))}")

print("=" * 72)
print(f"RESULT: {passed} checks passed, {len(failed)} failed.")
print("=" * 72)
if failed:
    raise SystemExit(1)
