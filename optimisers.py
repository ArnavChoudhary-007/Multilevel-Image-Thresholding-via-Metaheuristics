"""
Week 2 - the seven stochastic optimisers, on one fair footing.
Applied Statistics & Stochastic Processes project.

Every optimiser here MAXIMISES an objective J(t) over threshold vectors
t = (t1 < ... < tk), each in [1, 254], and every one of them is run
through the same CountedObjective wrapper. That wrapper is the single
most important thing in this file:

  * it counts objective evaluations and HARD-STOPS at the budget
    (raises BudgetExhausted), so no algorithm can sneak extra work;
  * it records the best-so-far score after every evaluation, giving a
    convergence curve on a common x-axis (evaluations, not iterations).

Why evaluations and not iterations? A swarm of 30 particles running 100
iterations spends 3,000 evaluations; a single-solution method running 100
iterations spends 100. Comparing "per iteration" silently hands the swarm
a 30x advantage. Equal EVALUATION budgets remove that.

The seven methods (paper, section 4.2):
    random_search   uniform random sampling - the "no intelligence" floor
    hill_climber    first-improvement local search with random restarts
    sa              Simulated Annealing (Metropolis rule, geometric cooling)
    pso             Particle Swarm Optimisation (constriction coefficients)
    ga              real-coded Genetic Algorithm
    de              Differential Evolution, DE/rand/1/bin
    gwo             Grey Wolf Optimiser (the "modern swarm method")

All seven are written by hand rather than taken from mealpy, so that
  (a) every one of them provably goes through the same counter,
  (b) a seed reproduces a run exactly, and
  (c) you can explain every line in a viva.

All parameters are in PARAMS below, fixed BEFORE any experiment was run.
They must not be tuned by looking at results.

Extensions (Part B): every optimiser also works for histograms with L > 256
grey levels, searching thresholds in [1, L-2]. Distance-based settings scale
with the range: PSO's vmax and GA's mutation sd were already fractions of
the range, DE and GWO move by differences of population members, and SA's
step sd becomes 8 * L / 256. The hill climber keeps its +-1 neighbourhood,
because that IS the method; at large L it is expected to stall.

Usage:
    from thresholding import load_grey, histogram
    from optimisers import make_objective, optimise
    f = make_objective(histogram(load_grey("camera")), "otsu")
    thresholds, score, history = optimise("pso", f, k=3, budget=5000, seed=0)
"""

import math

import numpy as np

from thresholding import class_tables

LOW, HIGH = 1, 254          # admissible threshold range (paper section 1.2)
ALGORITHMS = ["random_search", "hill_climber", "sa", "pso", "ga", "de", "gwo"]

# ----------------------------------------------------------------------
# Parameters - fixed in advance. Sources given where a standard exists;
# the rest are a-priori design choices, stated so they can be reported.
# ----------------------------------------------------------------------
PARAMS = {
    "sa": {
        "p0": 0.8,          # initial acceptance prob. of an average worsening move
                            #   (high initial acceptance, as in Kirkpatrick et al. 1983)
        "calib_moves": 50,  # proposals used to estimate the typical worsening move
        "t_final_ratio": 1e-3,  # T_end = T0 * 1e-3, reached exactly at the budget
        "step_sd": 8.0,     # sd (grey levels) of the move on one threshold
    },
    "pso": {
        "swarm": 30,
        "w": 0.7298,        # Clerc & Kennedy (2002) constriction coefficients
        "c1": 1.49618,
        "c2": 1.49618,
        "vmax_frac": 0.2,   # |velocity| <= 20% of the search range
    },
    "ga": {
        "pop": 30,
        "tournament": 2,
        "p_crossover": 0.9,
        "blx_alpha": 0.5,   # BLX-alpha crossover (Eshelman & Schaffer 1993)
        "mut_sd_frac": 0.1, # Gaussian mutation sd = 10% of the range
        "elites": 1,
    },
    "de": {
        "pop": 30,
        "F": 0.5,           # Storn & Price (1997) common settings
        "CR": 0.9,
    },
    "gwo": {
        "pop": 30,          # Mirjalili et al. (2014); a decreases 2 -> 0
    },
}


# ----------------------------------------------------------------------
# Objective + evaluation counter
# ----------------------------------------------------------------------

class BudgetExhausted(Exception):
    """Raised by CountedObjective when the evaluation budget is used up."""


def make_objective(p, objective="otsu", **kwargs):
    """
    Fast scalar objective f(t) for one histogram, built on class_tables.

    Same maths as otsu_score / kapur_score (test_optimisers.py checks it),
    but with the tables computed ONCE instead of on every call - this is
    what makes 13,440 runs x 5,000 evaluations affordable.

    Matches the DP's conventions exactly (Kapur: 0 ln 0 = 0, no EPS), so a
    metaheuristic that finds the optimum gets a gap of exactly 0.
    A split with an empty class returns -inf (rejected, not penalised).
    """
    P, S, T = (a.tolist() for a in class_tables(p))
    mu_T2 = float((np.arange(len(p)) * p).sum()) ** 2
    top = len(p) - 1

    if objective == "otsu":
        def f(t):
            total, lo = 0.0, -1
            for hi in tuple(t) + (top,):
                w = P[hi + 1] - P[lo + 1]
                if w <= 0:
                    return -math.inf
                m = S[hi + 1] - S[lo + 1]
                total += m * m / w
                lo = hi
            return total - mu_T2
    elif objective == "kapur":
        def f(t):
            total, lo = 0.0, -1
            for hi in tuple(t) + (top,):
                w = P[hi + 1] - P[lo + 1]
                if w <= 0:
                    return -math.inf
                total += math.log(w) - (T[hi + 1] - T[lo + 1]) / w
                lo = hi
            return total
    elif objective == "tsallis":
        # Non-separable criterion (Part C). Every class entropy S_c is
        # precomputed once into an L x L table (L = 256 here), so one
        # evaluation is k+1 lookups plus a sum and a product - the same
        # O(k) cost per evaluation as Otsu and Kapur.
        from thresholding import tsallis_table
        q_val = kwargs.get("q", 0.7)
        table = tsallis_table(p, q=q_val).tolist()   # plain floats: fast lookups
        one_minus_q = 1.0 - q_val

        def f(t):
            lo = 0
            total, prod = 0.0, 1.0
            for hi in tuple(t) + (top,):
                s_c = table[lo][hi]
                if s_c == -math.inf:
                    return -math.inf
                total += s_c
                prod *= s_c
                lo = hi + 1
            return total + one_minus_q * prod
        f.q = q_val
    else:
        raise ValueError("objective must be 'otsu', 'kapur' or 'tsallis'")
    f.objective = objective
    f.L = len(p)          # optimise() reads the search range [1, L-2] from this
    return f


class CountedObjective:
    """
    Wraps f(t). Every call costs one evaluation; call number budget+1
    raises BudgetExhausted. Tracks the best vector seen and the
    best-so-far score after each evaluation (the convergence curve).
    """

    def __init__(self, f, budget):
        self.f = f
        self.budget = int(budget)
        self.n = 0
        self.best_score = -math.inf
        self.best_t = None
        self.history = []

    def __call__(self, t):
        if self.n >= self.budget:
            raise BudgetExhausted
        self.n += 1
        t = tuple(int(v) for v in t)
        s = self.f(t)
        if s > self.best_score:
            self.best_score, self.best_t = s, t
        self.history.append(self.best_score)
        return s


# ----------------------------------------------------------------------
# Representation helpers
# ----------------------------------------------------------------------

def repair(x, low=LOW, high=HIGH):
    """
    Map any real vector to a valid threshold vector:
    round -> clip to [low, high] -> sort -> push apart so strictly increasing.
    Continuous methods (PSO, GA, DE, GWO) search in R^k and are evaluated
    at repair(x); the search itself never leaves the box [low, high]^k.
    """
    t = np.clip(np.rint(np.asarray(x, dtype=float)), low, high).astype(int)
    t.sort()
    k = t.size
    for i in range(1, k):                     # push up to remove ties
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1
    if t[-1] > high:                          # ...then down if we overflowed
        t[-1] = high
        for i in range(k - 2, -1, -1):
            if t[i] >= t[i + 1]:
                t[i] = t[i + 1] - 1
    return tuple(int(v) for v in t)


def random_vector(rng, k, low=LOW, high=HIGH):
    """Uniform over all C(high - low + 1, k) admissible vectors."""
    return tuple(sorted(int(v) for v in rng.choice(np.arange(low, high + 1), size=k,
                                                     replace=False)))


def random_population(rng, n, k, low=LOW, high=HIGH):
    return rng.uniform(low, high, size=(n, k))


# ----------------------------------------------------------------------
# 1. Random search
# ----------------------------------------------------------------------

def random_search(f, k, rng, budget, low=LOW, high=HIGH):
    while True:
        f(random_vector(rng, k, low=low, high=high))


# ----------------------------------------------------------------------
# 2. First-improvement hill climber with random restarts
# ----------------------------------------------------------------------

def hill_climber(f, k, rng, budget, low=LOW, high=HIGH):
    """
    Neighbourhood: move ONE threshold by +-1 (2k neighbours). Neighbours
    are tried in random order and the FIRST better one is taken. At a local
    optimum (no better neighbour) restart from a fresh random vector, so
    the remaining budget is not wasted.
    """
    x = random_vector(rng, k, low=low, high=high)
    fx = f(x)
    while True:
        moves = [(i, d) for i in range(k) for d in (-1, 1)]
        order = rng.permutation(len(moves))
        improved = False
        for m in order:
            i, d = moves[m]
            y = list(x)
            y[i] += d
            lo = y[i - 1] if i > 0 else low - 1
            hi = y[i + 1] if i < k - 1 else high + 1
            if not lo < y[i] < hi:
                continue                      # invalid: no evaluation spent
            fy = f(y)
            if fy > fx:
                x, fx, improved = tuple(y), fy, True
                break
        if not improved:
            x = random_vector(rng, k, low=low, high=high)
            fx = f(x)


# ----------------------------------------------------------------------
# 3. Simulated Annealing
# ----------------------------------------------------------------------

def sa(f, k, rng, budget, log=None, low=LOW, high=HIGH, L=256):
    """
    Metropolis rule: a move that changes the score by delta is accepted
    with probability  min(1, exp(delta / T)). Improvements are always
    accepted; worsening moves are accepted with a probability that shrinks
    as T falls. The sequence of states is therefore a Markov chain whose
    transition probabilities change over time (time-inhomogeneous),
    because T changes - the object studied in sa_markov.py.

    Temperature: T0 is calibrated so an average worsening move is accepted
    with probability p0 at the start; T then decays geometrically per
    evaluation so it reaches T0 * t_final_ratio exactly at the budget.
    Geometric cooling does NOT satisfy Hajek's (1988) condition for
    guaranteed convergence - the paper says so explicitly.

    Proposal: pick one threshold at random, move it by a rounded Gaussian
    step (never 0), then repair. Step SD scales with L (Part B).

    If `log` is a list, one tuple per proposal is appended:
        (evaluation_number, T, current_score, proposed_score, accepted)
    """
    prm = PARAMS["sa"]
    step_sd = prm["step_sd"] * (L / 256.0)

    def propose(x):
        y = list(x)
        i = rng.integers(k)
        step = 0
        while step == 0:
            step = int(np.rint(rng.normal(0.0, step_sd)))
        y[i] += step
        return repair(y, low=low, high=high)

    # start from a FEASIBLE vector (no empty class). On images with a narrow
    # grey-level range (e.g. 'clock' uses only levels 99-247) a random start
    # can have an empty class; every neighbour of it is then empty too, so the
    # Metropolis rule could never leave. Each retry costs one evaluation.
    x = random_vector(rng, k, low=low, high=high)
    fx = f(x)
    while not math.isfinite(fx):
        x = random_vector(rng, k, low=low, high=high)
        fx = f(x)

    # calibration: look at a few proposals from the start point (they cost
    # evaluations like everything else) to learn the typical worsening size
    worse = []
    for _ in range(prm["calib_moves"]):
        fy = f(propose(x))
        if math.isfinite(fy) and fy < fx:
            worse.append(fx - fy)
    mean_worse = float(np.mean(worse)) if worse else 1.0
    T0 = -mean_worse / math.log(prm["p0"])
    remaining = max(1, budget - f.n)
    alpha = prm["t_final_ratio"] ** (1.0 / remaining)

    T = T0
    while True:
        y = propose(x)
        fy = f(y)
        delta = fy - fx
        if delta >= 0:
            accept = True
        elif not math.isfinite(fy):
            accept = False
        else:
            accept = rng.random() < math.exp(delta / T)
        if log is not None:
            log.append((f.n, T, fx, fy, accept))
        if accept:
            x, fx = y, fy
        T *= alpha


# ----------------------------------------------------------------------
# 4. Particle Swarm Optimisation
# ----------------------------------------------------------------------

def pso(f, k, rng, budget, low=LOW, high=HIGH):
    """
    Each particle i has position x_i and velocity v_i. Per iteration:
        v_i <- w v_i + c1 r1 (pbest_i - x_i) + c2 r2 (gbest - x_i)
        x_i <- x_i + v_i
    inertia (w) + pull to own best (cognitive) + pull to swarm best (social).
    Positions stay inside [low, high]; a particle hitting a wall stops there.
    """
    prm = PARAMS["pso"]
    n = prm["swarm"]
    vmax = prm["vmax_frac"] * (high - low)
    x = random_population(rng, n, k, low=low, high=high)
    v = rng.uniform(-vmax, vmax, size=(n, k))
    pbest = x.copy()
    pbest_f = np.array([f(repair(xi, low=low, high=high)) for xi in x])
    g = int(np.argmax(pbest_f))
    while True:
        r1 = rng.random((n, k))
        r2 = rng.random((n, k))
        v = (prm["w"] * v
             + prm["c1"] * r1 * (pbest - x)
             + prm["c2"] * r2 * (pbest[g] - x))
        v = np.clip(v, -vmax, vmax)
        x = x + v
        hit = (x < low) | (x > high)
        x = np.clip(x, low, high)
        v[hit] = 0.0
        for i in range(n):
            fi = f(repair(x[i], low=low, high=high))
            if fi > pbest_f[i]:
                pbest[i], pbest_f[i] = x[i], fi
                if fi > pbest_f[g]:
                    g = i


# ----------------------------------------------------------------------
# 5. Genetic Algorithm (real-coded)
# ----------------------------------------------------------------------

def ga(f, k, rng, budget, low=LOW, high=HIGH):
    """
    Generational GA: tournament selection -> BLX-alpha crossover ->
    Gaussian mutation (each gene with prob 1/k) -> keep the best `elites`
    unchanged into the next generation.
    """
    prm = PARAMS["ga"]
    n = prm["pop"]
    span = high - low
    pop = random_population(rng, n, k, low=low, high=high)
    fit = np.array([f(repair(ind, low=low, high=high)) for ind in pop])

    def select():
        idx = rng.integers(n, size=prm["tournament"])
        return pop[idx[np.argmax(fit[idx])]]

    while True:
        order = np.argsort(-fit)
        children = [pop[i].copy() for i in order[:prm["elites"]]]
        child_fit = [fit[i] for i in order[:prm["elites"]]]
        while len(children) < n:
            a, b = select(), select()
            if rng.random() < prm["p_crossover"]:
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                d = hi - lo
                c = rng.uniform(lo - prm["blx_alpha"] * d, hi + prm["blx_alpha"] * d)
            else:
                c = a.copy()
            mutate = rng.random(k) < 1.0 / k
            c = c + mutate * rng.normal(0.0, prm["mut_sd_frac"] * span, size=k)
            c = np.clip(c, low, high)
            children.append(c)
            child_fit.append(f(repair(c, low=low, high=high)))
        pop, fit = np.array(children), np.array(child_fit)


# ----------------------------------------------------------------------
# 6. Differential Evolution, DE/rand/1/bin
# ----------------------------------------------------------------------

def de(f, k, rng, budget, low=LOW, high=HIGH):
    """
    For each target x_i: mutant v = x_r1 + F (x_r2 - x_r3) with r1, r2, r3
    distinct and != i; binomial crossover with rate CR (at least one gene
    from the mutant); keep the trial if it is at least as good (greedy).
    """
    prm = PARAMS["de"]
    n = prm["pop"]
    pop = random_population(rng, n, k, low=low, high=high)
    fit = np.array([f(repair(ind, low=low, high=high)) for ind in pop])
    while True:
        for i in range(n):
            choices = [j for j in range(n) if j != i]
            r1, r2, r3 = rng.choice(choices, size=3, replace=False)
            mutant = pop[r1] + prm["F"] * (pop[r2] - pop[r3])
            cross = rng.random(k) < prm["CR"]
            cross[rng.integers(k)] = True
            trial = np.clip(np.where(cross, mutant, pop[i]), low, high)
            ft = f(repair(trial, low=low, high=high))
            if ft >= fit[i]:
                pop[i], fit[i] = trial, ft


# ----------------------------------------------------------------------
# 7. Grey Wolf Optimiser
# ----------------------------------------------------------------------

def gwo(f, k, rng, budget, low=LOW, high=HIGH):
    """
    The three best wolves (alpha, beta, delta) lead. Every wolf moves to
    the average of three points, one pulled toward each leader:
        X_l = leader - A |C leader - x|,  A = 2a r1 - a,  C = 2 r2
    `a` falls linearly from 2 to 0 over the run (here: over the budget),
    shifting the pack from exploring (|A| > 1) to closing in (|A| < 1).
    """
    n = PARAMS["gwo"]["pop"]
    pop = random_population(rng, n, k, low=low, high=high)
    fit = np.array([f(repair(ind, low=low, high=high)) for ind in pop])
    while True:
        order = np.argsort(-fit)
        leaders = [pop[order[0]].copy(), pop[order[1]].copy(), pop[order[2]].copy()]
        a = 2.0 * max(0.0, 1.0 - f.n / budget)
        for i in range(n):
            new = np.zeros(k)
            for leader in leaders:
                A = 2.0 * a * rng.random(k) - a
                C = 2.0 * rng.random(k)
                new += leader - A * np.abs(C * leader - pop[i])
            pop[i] = np.clip(new / 3.0, low, high)
            fit[i] = f(repair(pop[i], low=low, high=high))


# ----------------------------------------------------------------------
# Single entry point
# ----------------------------------------------------------------------

_REGISTRY = {
    "random_search": random_search,
    "hill_climber": hill_climber,
    "sa": sa,
    "pso": pso,
    "ga": ga,
    "de": de,
    "gwo": gwo,
}


def optimise(algorithm, objective, k, budget, seed, sa_log=None, L=None):
    """
    Run one optimiser for exactly `budget` evaluations.

    algorithm : one of ALGORITHMS
    objective : f(t) -> score, e.g. from make_objective(p, "otsu")
    seed      : integer; the same seed always reproduces the same run
    sa_log    : optional list; if given and algorithm == "sa", every
                proposal is recorded into it (see sa()).
    L         : number of grey levels; thresholds are searched in [1, L-2].
                Default: taken from the objective (make_objective sets f.L),
                else 256. At L = 256 every run is bit-identical to the
                original 8-bit code (test_extensions.py re-runs rows of
                results/runs.csv to prove it).

    Returns (thresholds, score, history), where history[i] is the best
    score after i+1 evaluations (len(history) == budget).
    """
    if algorithm not in _REGISTRY:
        raise ValueError(f"unknown algorithm {algorithm!r}; choose from {ALGORITHMS}")
    rng = np.random.default_rng(seed)
    if L is None:
        L = getattr(objective, "L", 256)
    f = CountedObjective(objective, budget)
    low, high = 1, L - 2
    try:
        if algorithm == "sa":
            sa(f, k, rng, budget, log=sa_log, low=low, high=high, L=L)
        else:
            _REGISTRY[algorithm](f, k, rng, budget, low=low, high=high)
    except BudgetExhausted:
        pass
    return f.best_t, f.best_score, f.history
