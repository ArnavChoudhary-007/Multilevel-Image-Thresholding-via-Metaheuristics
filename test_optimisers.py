"""
Week 2 - checks for optimisers.py.
Applied Statistics & Stochastic Processes project.

The experiment's conclusions rest on four properties of the optimisers.
Each is tested here, so a bug shows up now rather than as a strange
result in Week 3:

  1. FAIRNESS    every run uses exactly the budget - never more
  2. HONESTY     no optimiser ever reports a score above the exact DP
                 optimum (if one did, either it or the DP would be wrong)
  3. REPRODUCIBILITY   the same seed gives the identical run
  4. SANITY      on easy problems (k = 2) the real optimisers usually
                 find the optimum; random search is the floor

Also checks that the fast objective used by the optimisers computes
the same numbers as thresholding.otsu_score / kapur_score.

Run:  python test_optimisers.py        (about 1-2 minutes)
"""

import math

import numpy as np

from thresholding import load_grey, histogram, otsu_score, kapur_score, dp_multilevel
from optimisers import (
    ALGORITHMS, LOW, HIGH, CountedObjective, BudgetExhausted,
    make_objective, optimise, repair,
)

BENCHMARK = ["camera", "coins", "moon", "page", "clock", "astronaut", "coffee", "cell"]
BUDGET = 5000

passed, failed = 0, []


def check(label, condition, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"  PASS  {label}  {detail}")
    else:
        failed.append(label)
        print(f"  FAIL  {label}  {detail}")


def valid(t, k):
    return (t is not None and len(t) == k and all(LOW <= v <= HIGH for v in t)
            and all(a < b for a, b in zip(t, t[1:])))


print("=" * 72)
print("OPTIMISER CHECKS")
print("=" * 72)
hists = {name: histogram(load_grey(name)) for name in BENCHMARK}
rng = np.random.default_rng(123)

# ------------------------------------------------------------ objective
print("\n0. Fast objective == thresholding.py objective (500 random vectors/image)")
for name, p in hists.items():
    fo, fk = make_objective(p, "otsu"), make_objective(p, "kapur")
    ok_o = ok_k = True
    for _ in range(500):
        k = int(rng.integers(1, 8))
        t = tuple(sorted(int(v) for v in rng.choice(np.arange(1, 255), size=k, replace=False)))
        a, b = fo(t), otsu_score(p, t)
        if b == -1.0:                        # otsu_score's degenerate marker
            ok_o &= a == -math.inf
        else:
            ok_o &= abs(a - b) <= 1e-9 * max(1.0, abs(b))
        a, b = fk(t), kapur_score(p, t)
        ok_k &= (a == b == -math.inf) or abs(a - b) < 1e-8   # kapur_score has EPS
    check(f"{name}: Otsu objective matches", ok_o)
    check(f"{name}: Kapur objective matches", ok_k)

# ------------------------------------------------------------ repair
print("\n1. repair() always returns a valid vector")
ok = True
for _ in range(20000):
    k = int(rng.integers(1, 11))
    x = rng.choice([rng.uniform(-500, 800, k), np.full(k, rng.uniform(-10, 300)),
                    rng.uniform(250, 260, k)])
    ok &= valid(repair(x), k)
check("20,000 random/degenerate inputs, k = 1..10", ok)
check("all-equal at the top edge", repair([254, 254, 254]) == (252, 253, 254))
check("all-equal at the bottom edge", repair([0, 0, 0]) == (1, 2, 3))

# ------------------------------------------------------------ counter
print("\n2. CountedObjective enforces the budget")
c = CountedObjective(make_objective(hists["camera"], "otsu"), 3)
for t in [(10,), (100,), (200,)]:
    c(t)
try:
    c((50,))
    check("call budget+1 raises BudgetExhausted", False)
except BudgetExhausted:
    check("call budget+1 raises BudgetExhausted", True)
check("best-so-far history is tracked", len(c.history) == 3 and c.best_t == (100,))

# ------------------------------------------------------------ fairness/honesty
print(f"\n3. Every algorithm: exact budget, valid answer, never beats the DP")
for name in ("camera", "moon", "coffee"):
    p = hists[name]
    for obj in ("otsu", "kapur"):
        f = make_objective(p, obj)
        for k in (2, 4):
            _, opt = dp_multilevel(p, k, obj)
            for algo in ALGORITHMS:
                t, s, h = optimise(algo, f, k, BUDGET, seed=7)
                ok = (len(h) == BUDGET
                      and all(a <= b for a, b in zip(h, h[1:]))
                      and valid(t, k)
                      and s == f(t)
                      and s <= opt + 1e-9 * max(1.0, abs(opt)))
                check(f"{name} {obj} k={k} {algo}", ok,
                      f"gap={(opt - s) / abs(opt):.2e}")

# ------------------------------------------------------------ narrow histogram
print("\n3b. Narrow histogram ('clock' uses grey levels 99-247 only): every algorithm,")
print("    every seed 0-9, still returns a valid vector with a finite score")
f = make_objective(hists["clock"], "otsu")
for algo in ALGORITHMS:
    ok = True
    for seed in range(10):
        t, s, h = optimise(algo, f, 2, 1000, seed=seed)
        ok &= valid(t, 2) and math.isfinite(s)
    check(f"clock {algo}", ok)

# ------------------------------------------------------------ reproducibility
print("\n4. Same seed -> identical run")
f = make_objective(hists["coins"], "kapur")
for algo in ALGORITHMS:
    r1 = optimise(algo, f, 3, 2000, seed=42)
    r2 = optimise(algo, f, 3, 2000, seed=42)
    check(f"{algo} reproducible", r1[0] == r2[0] and r1[2] == r2[2])

# ------------------------------------------------------------ SA log
print("\n5. Simulated Annealing proposal log (for the Markov-chain analysis)")
log = []
optimise("sa", make_objective(hists["camera"], "otsu"), 3, BUDGET, seed=1, sa_log=log)
temps = [row[1] for row in log]
check("one log row per proposal after calibration", 0 < len(log) <= BUDGET, f"{len(log)} rows")
check("temperature strictly decreasing", all(a > b for a, b in zip(temps, temps[1:])))
check("improvements always accepted",
      all(acc for _, _, cur, new, acc in log if new >= cur))

# ------------------------------------------------------------ sanity
print("\n6. Easy case (k = 2, camera): optimum found in >= 5 of 10 seeds")
for obj in ("otsu", "kapur"):
    p = hists["camera"]
    f = make_objective(p, obj)
    _, opt = dp_multilevel(p, 2, obj)
    for algo in ALGORITHMS:
        hits = sum(abs(optimise(algo, f, 2, BUDGET, seed=s)[1] - opt) <= 1e-9 * max(1.0, abs(opt))
                   for s in range(10))
        if algo == "random_search":
            print(f"  info  {obj} random_search hit {hits}/10 (the floor, not required)")
        else:
            check(f"{obj} {algo}", hits >= 5, f"hit {hits}/10")

print("\n" + "=" * 72)
if failed:
    print(f"RESULT: {passed} passed, {len(failed)} FAILED")
    for f in failed:
        print(f"   - {f}")
else:
    print(f"RESULT: all {passed} checks passed. Optimisers are fair and honest.")
print("=" * 72)
