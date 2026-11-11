"""
Interactive demo: exact DP versus the seven metaheuristics, live.
Applied Statistics & Stochastic Processes project.

Why this exists
---------------
The paper's whole argument (Q1 and Q2) is easiest to see, not read:
pick an image, pick k, and watch the exact dynamic programme return the
global optimum in milliseconds while each metaheuristic spends its full
evaluation budget and sometimes still misses. This app is a thin window
onto code that already exists and is already verified:

  * thresholding.py  - dp_multilevel (exact optimum), the 24-image benchmark,
                       the Tsallis criterion and its exhaustive search,
                       segment, quality
  * optimisers.py    - make_objective, optimise (same counter and budget
                       as the 13,440-run experiment)
  * run_experiments  - HIT_TOL, so "hit" means exactly what it means in
                       the paper

Tsallis (extension Part C) is the case where the DP does NOT apply: its
product term couples all classes. There the reference is an exhaustive
search for k <= 3 (exact) and, for k >= 4, the best score found by the
selected metaheuristics ("best known", NOT proven optimal) - and the page
says which one it is showing.

Nothing here computes a result its own way. Every number on screen is
produced live by those functions, so the app cannot disagree with the
report (rule: never invent a number).

Run it with:
    .venv\\Scripts\\streamlit run app.py
"""

import io
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from optimisers import ALGORITHMS, make_objective, optimise
from run_experiments import HIT_TOL
from thresholding import (BENCHMARK_24, BENCHMARK_ORIGINAL, bruteforce_tsallis, dp_multilevel,
                          histogram, load_grey, quality, segment)

TSALLIS_Q = 0.7            # fixed in advance (EXTENSIONS_DECISIONS.md, C2)
OBJECTIVE_LABELS = {"otsu": "Otsu", "kapur": "Kapur", "tsallis": "Tsallis (q = 0.7)"}

ALGO_LABELS = {
    "random_search": "Random search",
    "hill_climber": "Hill climber",
    "sa": "Simulated Annealing",
    "pso": "PSO",
    "ga": "Genetic Algorithm",
    "de": "Differential Evolution",
    "gwo": "Grey Wolf Optimiser",
}

st.set_page_config(page_title="Multilevel Thresholding: Exact DP vs Metaheuristics",
                   layout="wide")


# ----------------------------------------------------------------------
# Cached computations. Streamlit re-runs the whole script on every widget
# change, so anything expensive is cached on its inputs.
# ----------------------------------------------------------------------

@st.cache_data
def load_benchmark(name):
    return load_grey(name)


def load_upload(file):
    """Any uploaded picture -> uint8 greyscale, the same type load_grey gives."""
    return np.asarray(Image.open(io.BytesIO(file.getvalue())).convert("L"),
                      dtype=np.uint8)


@st.cache_data
def run_dp(p, k, objective):
    """Exact optimum plus its wall-clock time (median of a few repeats,
    because a single millisecond-scale timing is noisy)."""
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        thresholds, score = dp_multilevel(p, k, objective)
        times.append(time.perf_counter() - t0)
    return thresholds, score, float(np.median(times))


@st.cache_data
def run_exhaustive(p, k):
    """Exact Tsallis optimum by exhaustive search (k <= 3) and its wall time."""
    t0 = time.perf_counter()
    thresholds, score = bruteforce_tsallis(p, k, q=TSALLIS_Q)
    return thresholds, score, time.perf_counter() - t0


@st.cache_data
def run_meta(p, k, objective, algorithm, budget, seed):
    """One metaheuristic run, through the same counter as the experiment."""
    f = (make_objective(p, objective, q=TSALLIS_Q) if objective == "tsallis"
         else make_objective(p, objective))
    t0 = time.perf_counter()
    thresholds, score, history = optimise(algorithm, f, k, budget, seed)
    return thresholds, score, np.asarray(history), time.perf_counter() - t0


def is_hit(score, opt):
    """Same definition as run_experiments.py."""
    return (opt - score) <= HIT_TOL * max(1.0, abs(opt))


# ----------------------------------------------------------------------
# Sidebar: the experiment's factors, one run at a time
# ----------------------------------------------------------------------

st.sidebar.header("Setup")
source = st.sidebar.radio("Image", ["Benchmark", "Upload"], horizontal=True)
if source == "Benchmark":
    image_name = st.sidebar.selectbox(
        "Benchmark image", BENCHMARK_24,
        format_func=lambda n: n if n in BENCHMARK_ORIGINAL else f"{n}  (extension)")
    img = load_benchmark(image_name)
else:
    upload = st.sidebar.file_uploader("Greyscale or colour image",
                                      type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if upload is None:
        st.info("Upload an image in the sidebar to begin.")
        st.stop()
    image_name = upload.name
    img = load_upload(upload)

objective = st.sidebar.radio("Objective", ["otsu", "kapur", "tsallis"],
                             format_func=OBJECTIVE_LABELS.get,
                             help="Otsu and Kapur are separable, so the exact DP applies. "
                                  "Tsallis is not: no DP, exhaustive search only up to k = 3.")
k = st.sidebar.slider("Number of thresholds k", 1, 10, 3)
algorithms = st.sidebar.multiselect(
    "Metaheuristics", ALGORITHMS, default=["pso", "sa", "ga"],
    format_func=ALGO_LABELS.get)
budget = st.sidebar.number_input("Evaluation budget", 100, 50000, 5000, step=500,
                                 help="The paper uses 5,000 for every algorithm.")
seed = st.sidebar.number_input("Seed", 0, 10_000, 0,
                               help="Same seed, same run: optimisers are reproducible.")

p = histogram(img)

# ----------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------

def meta_row(a):
    t, s, hist, secs = run_meta(p, k, objective, a, int(budget), int(seed))
    meta[a] = (t, s, hist)
    return secs


meta, meta_secs = {}, {}
if objective == "tsallis":
    # No DP: run the metaheuristics first, because for k >= 4 the reference
    # is the best of THEM.
    with st.spinner("Running metaheuristics..."):
        for a in algorithms:
            meta_secs[a] = meta_row(a)
    if k <= 3:
        opt_t, opt_s, ref_time = run_exhaustive(p, k)
        ref_name, ref_kind, time_label = "Exhaustive search", "exact", "Exhaustive search time"
    else:
        scored = [(meta[a][1], a) for a in algorithms if meta[a][0]]
        if not scored:
            st.warning(f"For Tsallis with k = {k} there is no exact solver; the reference is the "
                       "best score found by the selected metaheuristics. Select at least one.")
            st.stop()
        opt_s, best_a = max(scored)
        opt_t, ref_time = meta[best_a][0], None
        ref_name, ref_kind, time_label = "Best known", "best known", "Reference time"
else:
    opt_t, opt_s, ref_time = run_dp(p, k, objective)
    ref_name, ref_kind, time_label = "Exact DP", "exact", "DP time"
    if opt_t is None:
        st.error(f"No valid split into {k + 1} non-empty classes exists for this image. "
                 "Try a smaller k.")
        st.stop()
    with st.spinner("Running metaheuristics..."):
        for a in algorithms:
            meta_secs[a] = meta_row(a)

dp_seg = segment(img, opt_t)
dp_psnr, dp_ssim = quality(img, dp_seg)
rows = [{
    "Method": ref_name, "Thresholds": ", ".join(map(str, opt_t)),
    "Score": opt_s, "Gap": 0.0, "Relative gap": 0.0, "Hit": True,
    "PSNR (dB)": dp_psnr, "SSIM": dp_ssim,
    "Time (ms)": ref_time * 1e3 if ref_time is not None else np.nan,
}]
for a in algorithms:
    t, s, _ = meta[a]
    seg = segment(img, t) if t else None
    ps, ss = quality(img, seg) if seg is not None else (np.nan, np.nan)
    rows.append({
        "Method": ALGO_LABELS[a], "Thresholds": ", ".join(map(str, t)) if t else "-",
        "Score": s, "Gap": opt_s - s, "Relative gap": (opt_s - s) / abs(opt_s),
        "Hit": is_hit(s, opt_s), "PSNR (dB)": ps, "SSIM": ss,
        "Time (ms)": meta_secs[a] * 1e3,
    })

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------

st.title("Multilevel Thresholding: Exact DP vs Metaheuristics")
st.caption(f"Image **{image_name}** · {OBJECTIVE_LABELS[objective]} · k = {k} · "
           f"budget {int(budget):,} evaluations · seed {int(seed)}")

c1, c2, c3 = st.columns(3)
c1.metric("Exact optimum" if ref_kind == "exact" else "Best known (not proven optimal)",
          f"{opt_s:.6g}")
c2.metric(time_label, f"{ref_time * 1e3:.2f} ms" if ref_time is not None else "n/a")
hits = sum(r["Hit"] for r in rows[1:])
c3.metric("Metaheuristics that hit the optimum" if ref_kind == "exact"
          else "Metaheuristics that matched the best known", f"{hits} / {len(algorithms)}")
if objective == "tsallis":
    st.info("**Tsallis is not separable**: its product term couples every class, so the exact "
            "DP does not apply (see test_tsallis.py). "
            + ("The reference is an exhaustive search over every threshold vector."
               if ref_kind == "exact" else
               f"At k = {k} an exhaustive search is infeasible, so the reference is the best "
               "score found by the selected metaheuristics - it is not proven optimal, and the "
               "best method trivially matches it."))

st.subheader("Results")
st.dataframe(
    pd.DataFrame(rows), hide_index=True, width="stretch",
    column_config={
        "Score": st.column_config.NumberColumn(format="%.6f"),
        "Gap": st.column_config.NumberColumn(format="%.3e"),
        "Relative gap": st.column_config.NumberColumn(format="%.3e"),
        "PSNR (dB)": st.column_config.NumberColumn(format="%.2f"),
        "SSIM": st.column_config.NumberColumn(format="%.4f"),
        "Time (ms)": st.column_config.NumberColumn(format="%.2f"),
    })

left, right = st.columns(2)

with left:
    st.subheader("Histogram and thresholds")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(np.arange(len(p)), p, width=1.0, color="0.6")
    for i, t in enumerate(opt_t):
        ax.axvline(t, color="black", lw=2, label=ref_name if i == 0 else None)
    colours = plt.cm.tab10.colors
    for j, a in enumerate(algorithms):
        t = meta[a][0]
        for i, x in enumerate(t or ()):
            ax.axvline(x, color=colours[j % 10], ls="--", lw=1.3,
                       label=ALGO_LABELS[a] if i == 0 else None)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Grey level")
    ax.set_ylabel("p(i)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Convergence (gap to the optimum)")
    if algorithms:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for j, a in enumerate(algorithms):
            gap = opt_s - meta[a][2]
            # log scale cannot show 0; floor at 1e-12 so a hit drops to the bottom
            ax.plot(np.arange(1, len(gap) + 1), np.maximum(gap, 1e-12),
                    color=colours[j % 10], label=ALGO_LABELS[a])
        ax.set_yscale("log")
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Best-so-far gap (log)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("A curve that reaches the bottom (1e-12) has found the "
                   + ("exact optimum." if ref_kind == "exact" else "best known value."))
    else:
        st.info("Select at least one metaheuristic in the sidebar.")

st.subheader("Segmentations (class-mean reconstruction)")
shown = [("Original", img), (ref_name, dp_seg)]
shown += [(ALGO_LABELS[a], segment(img, meta[a][0])) for a in algorithms if meta[a][0]]
cols = st.columns(min(len(shown), 4))
for i, (label, im) in enumerate(shown):
    cols[i % len(cols)].image(im, caption=label, width="stretch", clamp=True)
