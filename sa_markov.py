"""
Simulated Annealing as a time-inhomogeneous Markov chain - measured.
Applied Statistics & Stochastic Processes project.

Papers routinely SAY that Simulated Annealing is a Markov chain and cite
its convergence theory. This script MEASURES the chain our implementation
actually generates (contribution (iv) of the report).

Why it is a Markov chain: the next state depends only on the current state
and fresh randomness (propose a neighbour; accept with the Metropolis
probability min(1, exp(delta / T))). Why time-INhomogeneous: T falls every
step, so the transition probabilities themselves change over time.

Measured, over 30 seeds per cell (cells: 3 images x 2 objectives x k in {3, 5},
budget 5,000 evaluations - the same settings as the main experiment):

  A. Metropolis check     empirical acceptance rate of worsening moves vs the
                          rule's predicted probability exp(delta/T), binned.
                          Should lie on the diagonal: an empirical estimate of
                          the transition probabilities.
  B. Coarse-state chain   current state classed as
                            O = at the optimum,  N = rel. gap <= 1e-3,
                            F = rel. gap > 1e-3;
                          3x3 transition matrices estimated separately for the
                          first and second half of each run, and a chi-square
                          test of whether they are the same (homogeneity).
  C. Hitting time         evaluations until the chain first sits at the exact
                          optimum (empirical CDF; runs that never hit are
                          censored at the budget).
  D. Wasted evaluations   share of evaluations spent on a vector that had
                          already been evaluated earlier in the same run.
  E. Occupancy            share of the final 1,000 steps spent at the optimum
                          (low-temperature concentration, cf. the Boltzmann
                          distribution concentrating on optima as T -> 0).

Run:  python sa_markov.py          (about 1-2 minutes)
Writes results/sa_markov_*.csv and figures/sa_acceptance.png, figures/sa_hitting.png
"""

import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

from thresholding import load_grey, histogram, dp_multilevel
from optimisers import make_objective, optimise

RESULTS, FIGURES = Path("results"), Path("figures")
IMAGES = ["camera", "moon", "coins"]
OBJECTIVES = ["otsu", "kapur"]
KS = [3, 5]
SEEDS = range(30)
BUDGET = 5000
NEAR = 1e-3
STATES = ["O", "N", "F"]


def state_of(score, opt):
    rel = (opt - score) / abs(opt)
    if rel <= 1e-12:
        return 0
    return 1 if rel <= NEAR else 2


def run_cell(image, objective, k):
    p = histogram(load_grey(image))
    _, opt = dp_multilevel(p, k, objective)
    base = make_objective(p, objective)

    acc_pred, acc_obs = [], []                       # for A
    trans = {"early": np.zeros((3, 3), int), "late": np.zeros((3, 3), int)}   # B
    hits, wasted, occupancy = [], [], []             # C, D, E

    for seed in SEEDS:
        evaluated = []

        def f(t):
            evaluated.append(t)
            return base(t)

        log = []
        optimise("sa", f, k, BUDGET, seed, sa_log=log)

        # D: re-evaluations of a vector already seen in this run
        seen, dup = set(), 0
        for t in evaluated:
            dup += t in seen
            seen.add(t)
        wasted.append(dup / len(evaluated))

        # walk the chain
        hit_at = None
        prev_state = None
        half = BUDGET // 2
        at_opt_late, late_steps = 0, 0
        for n, T, fx, fy, accepted in log:
            if math.isfinite(fy) and fy < fx:        # A: a worsening proposal
                acc_pred.append(math.exp((fy - fx) / T))
                acc_obs.append(accepted)
            cur = fy if accepted else fx
            s = state_of(cur, opt)
            if prev_state is not None:
                trans["early" if n <= half else "late"][prev_state, s] += 1
            prev_state = s
            if s == 0 and hit_at is None:
                hit_at = n
            if n > BUDGET - 1000:
                late_steps += 1
                at_opt_late += s == 0
        hits.append(hit_at)
        occupancy.append(at_opt_late / late_steps if late_steps else float("nan"))

    return opt, np.array(acc_pred), np.array(acc_obs), trans, hits, wasted, occupancy


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    all_pred, all_obs = [], []
    summary, hit_rows = [], []
    pooled = {"early": np.zeros((3, 3), int), "late": np.zeros((3, 3), int)}

    for image in IMAGES:
        for objective in OBJECTIVES:
            for k in KS:
                opt, pred, obs, trans, hits, wasted, occ = run_cell(image, objective, k)
                all_pred.append(pred)
                all_obs.append(obs)
                for h in ("early", "late"):
                    pooled[h] += trans[h]
                hit_times = [h for h in hits if h is not None]
                summary.append({
                    "image": image, "objective": objective, "k": k,
                    "hit_rate": len(hit_times) / len(hits),
                    "median_hit_eval": float(np.median(hit_times)) if hit_times else "",
                    "worsening_accept_rate": float(obs.mean()) if obs.size else "",
                    "wasted_eval_share": float(np.mean(wasted)),
                    "late_occupancy_at_opt": float(np.nanmean(occ)),
                })
                for seed, h in zip(SEEDS, hits):
                    hit_rows.append({"image": image, "objective": objective, "k": k,
                                     "seed": seed, "hit_eval": "" if h is None else h})
                s = summary[-1]
                print(f"{image:<7} {objective:<5} k={k}: hit {s['hit_rate']:.0%}, "
                      f"median hit at eval {s['median_hit_eval']}, "
                      f"wasted evals {s['wasted_eval_share']:.1%}, "
                      f"late occupancy at optimum {s['late_occupancy_at_opt']:.1%}",
                      flush=True)

    pred = np.concatenate(all_pred)
    obs = np.concatenate(all_obs).astype(float)

    # A: calibration of the Metropolis rule, 10 equal-count bins
    order = np.argsort(pred)
    bins = np.array_split(order, 10)
    calib = [(float(pred[b].mean()), float(obs[b].mean()), len(b)) for b in bins]
    print("\nA. Metropolis check (worsening moves, 10 equal-count bins)")
    print("   predicted exp(delta/T)  ->  observed acceptance rate   (n)")
    for pm, om, n in calib:
        print(f"   {pm:20.4f}  ->  {om:8.4f}   ({n:,})")
    max_dev = max(abs(a - b) for a, b, _ in calib)
    print(f"   largest |observed - predicted| = {max_dev:.4f}")

    # B: homogeneity of the coarse chain, early vs late
    print("\nB. Coarse-state transition matrices (rows: from, cols: to; O/N/F)")
    for h in ("early", "late"):
        M = pooled[h]
        P = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        print(f"   {h} half ({M.sum():,} transitions):")
        for i, name in enumerate(STATES):
            print(f"     {name}: " + "  ".join(f"{P[i, j]:.3f}" for j in range(3))
                  + f"   (n={M[i].sum():,})")
    homog = []
    for i, name in enumerate(STATES):
        table = np.vstack([pooled["early"][i], pooled["late"][i]])
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] >= 2 and (table.sum(axis=1) > 0).all():
            chi2, pval, dof, _ = chi2_contingency(table)
            homog.append((name, chi2, dof, pval))
            p_txt = f"{pval:.2e}" if pval > 0 else "< 1e-300 (underflows to 0)"
            print(f"   from {name}: early vs late chi2 = {chi2:,.1f}, dof = {dof}, p = {p_txt}")

    with open(RESULTS / "sa_markov_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(RESULTS / "sa_markov_hits.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hit_rows[0].keys()))
        w.writeheader()
        w.writerows(hit_rows)
    with open(RESULTS / "sa_markov_chain.csv", "w", newline="") as fh:
        fh.write("half,from,to_O,to_N,to_F\n")
        for h in ("early", "late"):
            for i, name in enumerate(STATES):
                fh.write(f"{h},{name}," + ",".join(map(str, pooled[h][i])) + "\n")
        fh.write("\nfrom,chi2,dof,p\n")
        for name, chi2, dof, pval in homog:
            fh.write(f"{name},{chi2},{dof},{pval}\n")
        fh.write("\nbin,predicted,observed,n\n")
        for i, (pm, om, n) in enumerate(calib):
            fh.write(f"{i},{pm},{om},{n}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ink, muted, grid = "#0b0b0b", "#52514e", "#e5e5e2"

        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot([0, 1], [0, 1], color=muted, lw=1, ls="--", label="Metropolis rule")
        ax.plot([c[0] for c in calib], [c[1] for c in calib], "o-", color="#2a78d6",
                lw=2, ms=6, label="observed")
        ax.set_xlabel("predicted acceptance  exp(Δ/T)")
        ax.set_ylabel("observed acceptance rate")
        ax.set_title("Worsening moves: rule vs measured chain", fontsize=10, color=ink)
        ax.grid(True, color=grid, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "sa_acceptance.png", dpi=160)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        palette = {"camera": "#2a78d6", "moon": "#eb6834", "coins": "#1baf7a"}
        styles = {3: "-", 5: "--"}
        for ax, objective in zip(axes, OBJECTIVES):
            for image in IMAGES:
                for k in KS:
                    hs = [r["hit_eval"] for r in hit_rows if r["image"] == image
                          and r["objective"] == objective and r["k"] == k]
                    x = np.arange(0, BUDGET + 1, 50)
                    y = [np.mean([h != "" and h <= xi for h in hs]) for xi in x]
                    ax.plot(x, y, color=palette[image], ls=styles[k], lw=2,
                            label=f"{image}, k={k}")
            ax.set_title(f"{objective.capitalize()}", fontsize=10, color=ink)
            ax.set_xlabel("evaluations")
            ax.grid(True, color=grid, lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("share of runs that have hit the optimum")
        axes[1].legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle("Simulated Annealing: empirical hitting-time distribution", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGURES / "sa_hitting.png", dpi=160)
        print(f"\nSaved figures/sa_acceptance.png, figures/sa_hitting.png")
    except ImportError:
        pass
    print("Saved results/sa_markov_summary.csv, sa_markov_hits.csv, sa_markov_chain.csv")


if __name__ == "__main__":
    main()
