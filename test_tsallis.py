"""
Extension Part C - checks for the multilevel Tsallis criterion.
Applied Statistics & Stochastic Processes project.

What must be true before any Tsallis result can be trusted:

  1. Limit: as q -> 1, S_q -> Kapur's entropy (checked at q = 1 +- 1e-6).
  2. The fast objective used by the optimisers (make_objective "tsallis")
     equals the plain-loop reference tsallis_score on random vectors,
     including images with empty grey-level ranges (zero-mass classes).
  3. The vectorised exhaustive search returns the true optimum: it matches
     an independent full enumeration with the fast objective.
  4. It runs through CountedObjective: never more than the budget, one
     history entry per evaluation, returned score = score of returned t.
  5. The DP is NOT valid for this criterion: a DP-style maximisation of
     the separable part (sum of class entropies) returns different
     thresholds - and a lower S_q - than exhaustive search.
  6. The TRUE pseudo-additive Tsallis entropy (see thresholding.py) equals
     this criterion at k = 1, differs at k >= 2, and IS solved exactly by
     dp_tsallis_pseudoadditive (matches full enumeration for k = 1, 2, 3).

Run:  python test_tsallis.py
"""

import math
from itertools import combinations

import numpy as np

from optimisers import ALGORITHMS, make_objective, optimise
from thresholding import (BENCHMARK_ORIGINAL, bruteforce_tsallis, dp_tsallis_pseudoadditive,
                          histogram, kapur_score, load_grey, tsallis_pa_score, tsallis_score,
                          tsallis_table)

Q = 0.7
passed, failed = 0, []


def check(label, condition, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"  PASS  {label}  {detail}")
    else:
        failed.append(label)
        print(f"  FAIL  {label}  {detail}")


def close(a, b, rel=1e-10):
    if a == b:                       # also covers -inf == -inf
        return True
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


print("=" * 72)
print("PART C: MULTILEVEL TSALLIS CRITERION - CHECKS")
print("=" * 72)

hists = {n: histogram(load_grey(n)) for n in ["camera", "coins", "clock", "microaneurysms"]}

# ---------------------------------------------------------------- 1
print("\n1. q -> 1 limit: S_q approaches Kapur (q = 1 +- 1e-6, then 1 +- 1e-7)")
# The product term vanishes only LINEARLY, as (1-q) * prod_c S_c, and with
# 4-5 classes prod_c S_c is in the hundreds - so |S_q - Kapur| at q = 1 +- 1e-6
# is ~1e-4, not ~1e-6. The right statement of "S_q -> Kapur" is therefore
# that the difference shrinks in proportion to |q - 1|: cutting |q - 1| by
# 10x must cut the difference by ~10x (ratio checked to lie in [9, 11]).
for name in ("camera", "coins"):
    p = hists[name]
    for t in [(102,), (87, 176), (69, 134, 180), (40, 90, 140, 200)]:
        kap = kapur_score(p, t)
        for sign in (-1, 1):
            d6 = abs(tsallis_score(p, t, q=1 + sign * 1e-6) - kap)
            d7 = abs(tsallis_score(p, t, q=1 + sign * 1e-7) - kap)
            check(f"{name} t={t} q=1{'+' if sign > 0 else '-'}1e-6", 9 <= d6 / d7 <= 11,
                  f"|S_q - Kapur| = {d6:.2e} -> {d7:.2e} (ratio {d6 / d7:.2f})")

# ---------------------------------------------------------------- 2
print("\n2. Fast objective == plain-loop reference (500 random vectors per image)")
rng = np.random.default_rng(0)
for name, p in hists.items():
    f = make_objective(p, "tsallis", q=Q)
    worst, n_inf, ok = 0.0, 0, True
    for _ in range(500):
        k = int(rng.integers(1, 6))
        t = tuple(sorted(int(v) for v in rng.choice(np.arange(1, 255), size=k, replace=False)))
        a, b = f(t), tsallis_score(p, t, q=Q)
        n_inf += a == -math.inf
        if not close(a, b):
            ok = False
        if math.isfinite(a) and math.isfinite(b):
            worst = max(worst, abs(a - b))
    check(f"{name}", ok, f"max |diff| = {worst:.1e}, {n_inf} vectors with an empty class")

# ---------------------------------------------------------------- 3
print("\n3. Exhaustive search == independent full enumeration")
for name in ("camera", "clock"):
    p = hists[name]
    f = make_objective(p, "tsallis", q=Q)
    for k in (1, 2, 3):
        t_bf, s_bf = bruteforce_tsallis(p, k, q=Q)
        best_s, best_t = -math.inf, None
        for t in combinations(range(1, 255), k):
            s = f(t)
            if s > best_s:
                best_s, best_t = s, t
        check(f"{name} k={k}", t_bf == best_t and close(s_bf, best_s),
              f"exhaustive={t_bf} enumeration={best_t} S={s_bf:.6f}")

# ---------------------------------------------------------------- 4
print("\n4. Through CountedObjective: budget respected, score consistent")
p = hists["camera"]
f = make_objective(p, "tsallis", q=Q)
for algo in ALGORITHMS:
    t, s, hist = optimise(algo, f, 3, 500, seed=1)
    check(f"{algo}", len(hist) == 500 and close(s, tsallis_score(p, t, q=Q))
          and all(hist[i] <= hist[i + 1] for i in range(len(hist) - 1)),
          f"evaluations={len(hist)} t={t}")

# ---------------------------------------------------------------- 5
print("\n5. The DP is NOT valid: maximising the separable part gives another answer")


def dp_on_sum(p, k, q):
    """DP-style maximisation of sum_c S_c (the separable part only)."""
    tab = tsallis_table(p, q)
    L = len(p)
    G = np.full((L, L), -np.inf)                    # G[s, t]: class (s, t]
    G[:L - 1, :] = tab[1:, :]
    D = np.full((k + 1, L), -np.inf)
    B = np.zeros((k + 1, L), dtype=int)
    D[0, 1:L - 1] = tab[0, 1:L - 1]
    for j in range(1, k + 1):
        M = D[j - 1][:, None] + G
        cols = [L - 1] if j == k else list(range(j + 1, L - 1))
        D[j, cols] = M.max(axis=0)[cols]
        B[j, cols] = M.argmax(axis=0)[cols]
    out, t = [], L - 1
    for j in range(k, 0, -1):
        t = int(B[j, t])
        out.append(t)
    return tuple(reversed(out))


differ = []
for name in BENCHMARK_ORIGINAL:
    p = histogram(load_grey(name))
    for k in (2, 3):
        t_dp = dp_on_sum(p, k, Q)
        t_ex, s_ex = bruteforce_tsallis(p, k, q=Q)
        s_dp = tsallis_score(p, t_dp, q=Q)
        if t_dp != t_ex:
            differ.append((name, k, t_dp, t_ex, s_ex - s_dp))
print(f"     DP-on-sum differs from the exhaustive optimum in {len(differ)} of 16 cases"
      f" (8 images x k = 2, 3)")
for name, k, t_dp, t_ex, loss in differ[:4]:
    print(f"     e.g. {name} k={k}: DP-style {t_dp} vs exhaustive {t_ex}, S_q lower by {loss:.4g}")
check("at least one counterexample exists", len(differ) >= 1)
check("exhaustive is never beaten by DP-on-sum",
      all(loss >= -1e-12 for *_, loss in differ))

# ---------------------------------------------------------------- 6
print("\n6. True pseudo-additive Tsallis entropy: separable, exact DP valid")
p = hists["camera"]
for t in [(102,), (87, 176), (69, 134, 180)]:
    a, b = tsallis_pa_score(p, t, q=Q), tsallis_score(p, t, q=Q)
    if len(t) == 1:
        check(f"k=1 t={t}: pseudo-additive == study criterion", close(a, b), f"{a:.6f}")
    else:
        check(f"k={len(t)} t={t}: pseudo-additive != study criterion", not close(a, b),
              f"{a:.6f} vs {b:.6f}")

for name in ("camera", "clock"):
    p = hists[name]
    for k in (1, 2, 3):
        t_dp, s_dp = dp_tsallis_pseudoadditive(p, k, q=Q)
        best_s, best_t = -math.inf, None
        for t in combinations(range(1, 255), k):
            s = tsallis_pa_score(p, t, q=Q) if k < 3 else None
            if k == 3:
                break
            if s > best_s:
                best_s, best_t = s, t
        if k == 3:
            # 2.7 million vectors: vectorise over the last threshold with ln A
            P = np.concatenate([[0.0], np.cumsum(p)])
            pq = np.where(p > 0, p ** Q, 0.0)
            PQ = np.concatenate([[0.0], np.cumsum(pq)])

            def lnA(lo, hi):
                w = P[hi + 1] - P[lo + 1]
                a = PQ[hi + 1] - PQ[lo + 1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.where((w > 0) & (a > 0), np.log(a) - Q * np.log(w), -np.inf)
            best_v = -math.inf
            for t1, t2 in combinations(range(1, 254), 2):
                t3 = np.arange(t2 + 1, 255)
                v = lnA(-1, t1) + lnA(t1, t2) + lnA(t2, t3) + lnA(t3, 255)
                i = int(np.argmax(v))
                if v[i] > best_v:
                    best_v, best_t = v[i], (t1, t2, int(t3[i]))
            best_s = tsallis_pa_score(p, best_t, q=Q)
        check(f"{name} k={k}: DP == full enumeration", t_dp == best_t and close(s_dp, best_s),
              f"dp={t_dp} enumeration={best_t}")

print("=" * 72)
print(f"RESULT: {passed} checks passed, {len(failed)} failed.")
print("=" * 72)
if failed:
    raise SystemExit(1)
