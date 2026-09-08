"""
Reproducible analysis pipeline (v4.0.0)

"Capacity Allocation Under Diminishing Marginal Returns in Large Language Model
Training: A Production-Economic Decision Rule Coupling Scaling Laws with Cost
and Carbon Constraints"

One command regenerates every number, table and figure reported in the paper.

    PYTHONPATH=. python src/analysis.py

Design rules enforced here, in order:

  1. REPLICATE, THEN FREEZE. The production-function parameters are obtained by
     replicating the estimation protocol of the source reconstruction exactly
     (Huber loss on log-residuals in log-sum-exp form, delta=1e-3, L-BFGS-B from
     a grid of starts, on the 240 points remaining after the five highest-loss
     outliers are excluded). They are then frozen. No downstream quantity is
     inherited from an earlier specification; everything is recomputed from the
     frozen values.

  2. CLOSED FORM, NOT GRIDS. The expansion path follows from the first-order
     condition of  min L(N,D) s.t. 6ND = C, so every frontier quantity is
     evaluated analytically. This removes interpolation error and the clamping
     failure mode that affected v1/v2.

  3. THE DECISION VARIABLE IS COMPUTE. Marginal gain is defined per doubling of
     COMPUTE, not of model size. On the path N grows as C^a with a ~ 0.514, so
     doubling N corresponds to 3.85x the compute; a metric on N -> 2N would
     report the gain from almost four times the resources.

  4. COMPARE LIKE WITH LIKE. The empirical counterpart of the path is the
     compute-matched envelope: lowest observed loss per band of training
     compute. An envelope over bands of model size is NOT comparable to a locus
     derived under a budget constraint, and using it produces materially
     different (and misleading) conclusions -- see CHANGELOG v4.0.0.

Data: Besiroglu et al. (2024) public reconstruction of Hoffmann et al. (2022).
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import minimize, curve_fit, brentq

DATA = "data/llm_scaling_dataset.csv"
OUT_FIG, OUT_RES = "figures/", "results/"
DELTA = 1e-3                      # Huber delta of the source protocol
OPT = dict(ftol=1e-15, gtol=1e-12, maxiter=5000)
N_OUTLIERS = 5                    # highest-loss points excluded, as in source
RNG_MC, RNG_BOOT = 7, 42
INK, SIG, OK, ALT, MUT = "#1f3b73", "#c0392b", "#1e6f45", "#8a5a1e", "#8a8a8a"

# ----------------------------------------------------------------------
# 1. Estimation: replicate the source protocol, then freeze
# ----------------------------------------------------------------------
def load():
    d = pd.read_csv(DATA)
    return d.N.values, d.D.values, d.loss.values, d.C_flop.values

def _objective(p, lnN, lnD, lnL):
    """Huber loss on log-residuals, log-sum-exp form, with analytic gradient."""
    a, b, e, al, be = p
    x1, x2 = a - al * lnN, b - be * lnD
    x3 = np.full_like(lnN, e)
    m = np.maximum(np.maximum(x1, x2), x3)
    e1, e2, e3 = np.exp(x1 - m), np.exp(x2 - m), np.exp(x3 - m)
    Z = e1 + e2 + e3
    r = m + np.log(Z) - lnL
    ar = np.abs(r)
    h = np.where(ar <= DELTA, 0.5 * r ** 2, DELTA * (ar - 0.5 * DELTA))
    dh = np.where(ar <= DELTA, r, DELTA * np.sign(r))
    s1, s2, s3 = e1 / Z, e2 / Z, e3 / Z
    g = np.array([np.sum(dh * s1), np.sum(dh * s2), np.sum(dh * s3),
                  np.sum(dh * -lnN * s1), np.sum(dh * -lnD * s2)])
    return np.sum(h), g

def fit_source_protocol(N, D, L, grid_search=True, p0=None):
    lnN, lnD, lnL = np.log(N), np.log(D), np.log(L)
    starts = (list(product(np.arange(0, 30, 5), np.arange(0, 30, 5),
                           np.arange(-1, 1.5, .5), np.arange(0, 2.5, .5),
                           np.arange(0, 2.5, .5)))
              if grid_search else [p0])
    best = (np.inf, None)
    for s in starts:
        r = minimize(_objective, list(s), args=(lnN, lnD, lnL), jac=True,
                     method="L-BFGS-B", options=OPT)
        if r.fun < best[0]:
            best = (r.fun, r.x)
    a, b, e, al, be = best[1]
    return dict(A=float(np.exp(a)), B=float(np.exp(b)), E=float(np.exp(e)),
                alpha=float(al), beta=float(be)), best[1]

def drop_outliers(N, D, L, C, k=N_OUTLIERS):
    keep = L < np.sort(L)[-k]
    return N[keep], D[keep], L[keep], C[keep]

# ----------------------------------------------------------------------
# 2. Expansion path in closed form
# ----------------------------------------------------------------------
class Path:
    """Expansion path of  min L(N,D) s.t. 6ND = C, from frozen parameters."""
    def __init__(self, p):
        self.A, self.B, self.E = p["A"], p["B"], p["E"]
        self.al, self.be = p["alpha"], p["beta"]
        self.a = self.be / (self.al + self.be)          # N ~ C^a
        self.K = ((self.al * self.A) / (self.be * self.B)) ** (1 / (self.al + self.be))
    def N(self, C):  return self.K * (C / 6) ** self.a
    def D(self, C):  return C / (6 * self.N(C))
    def L(self, C):
        n = self.N(C)
        return self.E + self.A * n ** -self.al + self.B * self.D(C) ** -self.be
    def C_of_N(self, n): return 6 * (n / self.K) ** (1 / self.a)
    def gain(self, C):                                   # per COMPUTE doubling
        return (self.L(C) - self.L(2 * C)) / self.L(C)
    def threshold(self, hurdle, lo=1e15, hi=1e32):
        try:
            return self.N(brentq(lambda C: self.gain(C) - hurdle, lo, hi))
        except ValueError:
            return float("nan")
    def dC_per_1pct(self, n):
        C0 = self.C_of_N(n)
        target = .99 * self.L(C0)
        return brentq(lambda C: self.L(C) - target, C0, C0 * 1e6) - C0

# ----------------------------------------------------------------------
# 3. Empirical envelopes  (compute-matched is the decision-relevant one)
# ----------------------------------------------------------------------
def envelope(x, L, nbands=15, min_per_band=2):
    lx = np.log10(x)
    edges = np.linspace(lx.min(), lx.max(), nbands + 1)
    idx = np.clip(np.digitize(lx, edges) - 1, 0, nbands - 1)
    rows = []
    for b in range(nbands):
        m = idx == b
        if m.sum() >= min_per_band:
            j = np.argmin(L[m])
            rows.append((x[m][j], L[m][j]))
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])

f_pow = lambda x, A, b, E: A * x ** -b + E
f_log = lambda x, k, a, b, E: k / np.log(a * x + b) + E
P0_POW_C, P0_LOG_C = [50, .05, 1.8], [20, 1e-19, 1, 1.5]
BND_LOG_C = ([0, 1e-25, 0, .5], [500, 1, 1e6, 2.4])
P0_POW_N, P0_LOG_N = [400, .34, 1.7], [20, 1e-6, 10, 1.5]
BND_LOG_N = ([0, 1e-12, 0, .5], [100, 1, 1e6, 2.4])

def gof(f, p, x, y, k):
    pred = f(x, *p); rss = np.sum((y - pred) ** 2); n = len(y)
    return dict(R2=float(1 - rss / np.sum((y - y.mean()) ** 2)),
                MAPE=float(np.mean(np.abs((y - pred) / y)) * 100),
                AIC=float(n * np.log(rss / n) + 2 * k))

def loocv(f, p0, bounds, x, y, log_starts=None):
    """Leave-one-out RMSE. For the log form the refit is multistart, so the
    cross-validation is not contaminated by the optimizer landing in different
    local optima on different folds."""
    err = []
    for i in range(len(x)):
        m = np.ones(len(x), bool); m[i] = False
        if log_starts is not None:
            pi = fit_log_multistart(x[m], y[m], bounds, log_starts)
        else:
            kw = dict(bounds=bounds) if bounds else {}
            pi, _ = curve_fit(f, x[m], y[m], p0=p0, maxfev=800000, **kw)
        err.append(y[i] - f(x[i], *pi))
    return float(np.sqrt(np.mean(np.array(err) ** 2)))

# The reciprocal-logarithmic form has many local optima: on the compute-matched
# envelope, single-start fits from different seeds land at residual sums of
# squares between 0.09 and 35. A single start therefore makes the reported fit
# depend on the optimizer version rather than on the data, and can also report a
# worse fit than the form actually admits. Every log fit below is a multistart
# over a fixed grid, keeping the lowest residual sum of squares, which is both
# deterministic across environments and fair to the specification.
LOG_STARTS_C = [[20, 1e-19, 1, 1.5], [50, 1e-20, 1, 1.0], [10, 1e-18, .5, 1.8],
                [100, 1e-21, 10, .8], [45, 4e-11, 1, .6], [5, 1e-17, 1, 2.0],
                [30, 1e-13, .25, .7], [15, 1e-16, 2, 1.2]]
LOG_STARTS_N = [[20, 1e-6, 10, 1.5], [50, 1e-7, 1, 1.0], [10, 1e-5, .5, 1.8],
                [80, 1e-8, 100, .8], [30, 1e-6, .25, .7], [15, 1e-4, 2, 1.2]]

def fit_log_multistart(x, y, bounds, starts):
    best = (np.inf, None)
    for p0 in starts:
        try:
            p, _ = curve_fit(f_log, x, y, p0=p0, bounds=bounds, maxfev=800000)
        except Exception:
            continue
        rss = float(np.sum((y - f_log(x, *p)) ** 2))
        if rss < best[0]:
            best = (rss, p)
    if best[1] is None:
        raise RuntimeError("log fit failed from every start")
    return best[1]

def compare_forms(x, y, p0p, p0l, bndl, log_starts):
    pp, _ = curve_fit(f_pow, x, y, p0=p0p, maxfev=800000)
    pl = fit_log_multistart(x, y, bndl, log_starts)
    return dict(
        n=int(len(x)),
        power=dict(**gof(f_pow, pp, x, y, 3), LOOCV=loocv(f_pow, p0p, None, x, y),
                   params=[float(v) for v in pp]),
        log=dict(**gof(f_log, pl, x, y, 4), LOOCV=loocv(f_log, p0l, bndl, x, y, log_starts),
                 params=[float(v) for v in pl]))

# ----------------------------------------------------------------------
# 4. Three-way decomposition of the resource gap
# ----------------------------------------------------------------------
# Illustrative economic and energy parameters. These are scenario inputs, not
# measurements: the compute price is an order-of-magnitude figure from public
# cloud pricing, the energy per FLOP is representative of current accelerators,
# and the grid intensity is a national average (Ember, 2025).
C_COMPUTE = 3e-18      # $/FLOP. Constructed scenario, not a market survey.
                       # 2 $/h at a peak 1e15 FLOP/s is 5.6e-19 $/FLOP if every
                       # peak operation were useful. Model FLOPs utilisation of
                       # order 40% multiplies this by ~2.5; storage, interconnect,
                       # checkpointing and abandoned runs contribute a further ~2x.
                       # 3e-18 is therefore ~5.4x the peak-rate figure. Each factor
                       # is an assumption; vary this constant to test others.
P_CO2     = 80.0       # $/tCO2e
E_FLOP    = 2e-12      # J/FLOP
PUE       = 1.2        # dimensionless
CI_GRID   = 400.0      # g CO2e/kWh

LANDAUER_J_PER_BIT = 2.87e-21     # kT ln2 at T = 300 K (reported, not used in the paper)

def _excess_ratios(path, L, C, C_lo, C_hi):
    """Realised compute over the compute the path needs for the same loss.
    Points whose implied optimum falls outside the observed budget range are
    dropped: inverting the law there would extrapolate."""
    keep, dropped = [], 0
    for c, l in zip(C, L):
        if l <= path.E:
            dropped += 1; continue
        try:
            c_opt = brentq(lambda x: path.L(x) - l, 1e14, 1e30)
        except ValueError:
            dropped += 1; continue
        if not (C_lo / 10 <= c_opt <= C_hi * 10):
            dropped += 1; continue
        keep.append(c / c_opt)
    return np.array(keep), dropped

def _summary(r, dropped):
    return dict(n=int(len(r)), dropped=int(dropped), median=float(np.median(r)),
                p90=float(np.percentile(r, 90)), max=float(r.max()),
                share_above_2x=float(np.mean(r > 2)))

def sample_composition(path, L, C, k=N_OUTLIERS):
    """Which observations each sample drops, and how the two rules overlap."""
    C_lo, C_hi = C.min(), C.max()
    domain_excluded = []
    for i, (c, l) in enumerate(zip(C, L)):
        if l <= path.E:
            domain_excluded.append(i); continue
        try:
            c_opt = brentq(lambda x: path.L(x) - l, 1e14, 1e30)
        except ValueError:
            domain_excluded.append(i); continue
        if not (C_lo / 10 <= c_opt <= C_hi * 10):
            domain_excluded.append(i)
    high_loss = sorted(int(i) for i in np.argsort(L)[-k:])
    overlap = sorted(set(domain_excluded) & set(high_loss))
    return dict(
        n_total=int(len(L)),
        domain_excluded=dict(indices=[int(i) for i in domain_excluded],
                             losses=[float(L[i]) for i in domain_excluded]),
        high_loss_excluded=dict(indices=high_loss, losses=[float(L[i]) for i in high_loss]),
        overlap_indices=overlap,
        n_row1=int(len(L) - len(domain_excluded)),
        n_row2=int(len(L) - len(set(domain_excluded) | set(high_loss))),
        note=("Row 1 drops only the domain exclusions. Row 2 drops the union of the "
              "domain exclusions and the five highest-loss points; the overlap is why "
              "removing five points reduces the count by three."))

def decomposition(path, N, L, C, L_fit, C_fit, e_flop=2e-12, bits_per_flop=16):
    C_lo, C_hi = C.min(), C.max()
    r_all, d_all = _excess_ratios(path, L, C, C_lo, C_hi)
    r_fit, d_fit = _excess_ratios(path, L_fit, C_fit, C_lo, C_hi)
    return dict(
        geometry_exponent=float(1 / path.a),
        allocation=dict(**_summary(r_all, d_all),
                        sensitivity=dict(
                            full_sample=_summary(r_all, d_all),
                            excluding_5_high_loss=_summary(r_fit, d_fit))),
        hardware_gap_not_reported=("Converting a floor on irreversible bit erasure "
            "into a bound per floating-point operation requires assumptions about "
            "erasures per operation that are not justified here; no result depends "
            "on this quantity."),
        note=("Geometry is a property of the production function and is not "
              "addressable by hardware. Allocation is the distance from the "
              "expansion path and is closed by budget balancing. The hardware "
              "gap to the Landauer bound closes only with a change of substrate."))

# ----------------------------------------------------------------------
# 5. Robustness
# ----------------------------------------------------------------------
def monte_carlo(N, D, L, p_start, hurdle=0.04, noises=(0.01, 0.02, 0.05), reps=800):
    rng = np.random.default_rng(RNG_MC)
    out = {}
    for s in noises:
        vals = []
        for _ in range(reps):
            Lp = L * (1 + rng.normal(0, s, len(L)))
            p, _ = fit_source_protocol(N, D, Lp, grid_search=False, p0=p_start)
            t = Path(p).threshold(hurdle)
            if np.isfinite(t):
                vals.append(np.log10(t))
        v = np.array(vals)
        out[f"{int(s*100)}%"] = dict(median=float(10 ** np.median(v)),
            ci=[float(10 ** np.percentile(v, 2.5)), float(10 ** np.percentile(v, 97.5))],
            n=int(len(v)))
    return out

def bootstrap(N, D, L, p_start, reps=1000):
    rng = np.random.default_rng(RNG_BOOT)
    keys = ["A", "B", "E", "alpha", "beta"]
    acc = {k: [] for k in keys}; acc["a"] = []
    for _ in range(reps):
        i = rng.choice(len(N), len(N), replace=True)
        p, _ = fit_source_protocol(N[i], D[i], L[i], grid_search=False, p0=p_start)
        for k in keys:
            acc[k].append(p[k])
        acc["a"].append(p["beta"] / (p["alpha"] + p["beta"]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in acc.items()}

# ----------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------
def figures(path, N, L, C, envC, envL, cmp_C, hurdles, res):
    os.makedirs(OUT_FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 300})
    Nmin, Nmax = N.min(), N.max()
    ns = np.logspace(np.log10(Nmin), np.log10(Nmax), 300)

    # Fig 1: cloud + path
    plt.figure(figsize=(6.4, 4.4))
    plt.scatter(N, L, s=9, alpha=.22, color=MUT, label=f"Observations (n={len(N)})")
    plt.plot(ns, [path.L(path.C_of_N(x)) for x in ns], lw=2.4, color=INK,
             label="Expansion path $L^*(N)$")
    plt.xscale("log"); plt.xlabel("Model parameters $N$"); plt.ylabel("Validation loss")
    plt.title("Validation loss versus model size"); plt.legend(fontsize=7.6)
    plt.tight_layout(); plt.savefig(OUT_FIG + "fig1_path.png"); plt.close()

    # Fig 2: marginal gain per compute doubling
    plt.figure(figsize=(6.4, 4.4))
    nn = np.logspace(7.3, 10.6, 400)
    plt.plot(nn, [path.gain(path.C_of_N(x)) * 100 for x in nn], lw=2.4, color=INK)
    for h in hurdles:
        plt.axhline(h * 100, ls="--" if h == .04 else ":", lw=1.1,
                    color=SIG if h == .04 else "#b9b9b9")
    plt.axvspan(nn[0], Nmin, color="#f3f0e6", zorder=0)
    plt.axvspan(Nmax, nn[-1], color="#f3f0e6", zorder=0)
    plt.xscale("log"); plt.ylim(0, 9)
    plt.xlabel("Model parameters $N$")
    plt.ylabel("Marginal loss reduction per compute doubling (%)")
    plt.title("Diminishing marginal returns along the expansion path")
    plt.tight_layout(); plt.savefig(OUT_FIG + "fig2_gain.png"); plt.close()

    # Fig 3: threshold as a function of the hurdle
    hs = np.linspace(.012, .08, 40)
    ts = [path.threshold(h) for h in hs]
    plt.figure(figsize=(6.4, 4.4))
    plt.plot(hs * 100, ts, "-o", ms=3.4, color=INK)
    plt.axhspan(Nmin, Nmax, color="#eef3ea", zorder=0, label="Observed range")
    plt.yscale("log"); plt.xlabel("Marginal performance hurdle $r$ (%)")
    plt.ylabel("Efficiency threshold $N^*(r)$")
    plt.title("The threshold is a function of the organizational hurdle")
    plt.legend(fontsize=7.6)
    plt.tight_layout(); plt.savefig(OUT_FIG + "fig3_threshold.png"); plt.close()

    # Fig 4: dC per 1%
    plt.figure(figsize=(6.4, 4.4))
    n2 = np.logspace(8, 10.2, 200)
    plt.plot(n2, [path.dC_per_1pct(x) for x in n2], lw=2.4, color=INK)
    plt.axvspan(n2[0], Nmin, color="#f3f0e6", zorder=0)
    plt.axvspan(Nmax, n2[-1], color="#f3f0e6", zorder=0)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Model parameters $N$"); plt.ylabel(r"$\Delta C$ per 1% loss reduction (FLOP)")
    plt.title("Marginal compute per unit of performance")
    plt.tight_layout(); plt.savefig(OUT_FIG + "fig4_cost.png"); plt.close()

    # Fig 4b: Monte Carlo robustness (kept with the others so every figure in the
    # paper comes from this pipeline)
    mc = res["monte_carlo_4pct"]; det = res["thresholds"]["4%"]["N"]
    plt.figure(figsize=(6.4, 3.6))
    lv, pos = ["1%", "2%", "5%"], [1, 2, 3]
    for i, k in enumerate(lv):
        m = mc[k]
        plt.plot([m["ci"][0], m["ci"][1]], [pos[i]] * 2, lw=3, color=INK,
                 solid_capstyle="round")
        plt.plot([m["median"]], [pos[i]], "o", ms=8, color=SIG, zorder=5)
    plt.axvline(det, color=OK, lw=1.6, ls="--",
                label=f"deterministic $N^*$ = {det:.2e}")
    plt.yticks(pos, [f"±{k} loss noise" for k in lv]); plt.xscale("log")
    plt.xlabel("Threshold $N^*$ at $r=4\\%$")
    plt.title("Robustness: perturbation magnitude as a sensitivity parameter")
    plt.legend(fontsize=7.5); plt.tight_layout()
    plt.savefig(OUT_FIG + "fig_mc.png"); plt.close()

    # Fig 5: VALIDATION + functional form on the decision-relevant object
    pp = cmp_C["power"]["params"]; pl = cmp_C["log"]["params"]
    cs = np.logspace(np.log10(envC.min()), np.log10(envC.max()), 200)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax[0].scatter(C, L, s=8, alpha=.18, color=MUT, label=f"Observations (n={len(C)})")
    ax[0].plot(cs, [path.L(c) for c in cs], lw=2.4, color=INK, label="Fitted path $L^*(C)$")
    ax[0].scatter(envC, envL, s=34, color=OK, marker="D", zorder=5,
                  label=f"Compute-matched envelope (n={len(envC)})")
    ax[0].set_xscale("log"); ax[0].set_xlabel("Training compute $C$ (FLOP)")
    ax[0].set_ylabel("Validation loss")
    ax[0].set_title("Model path against the empirical envelope"); ax[0].legend(fontsize=7.2)
    ax[1].scatter(envC, envL, s=34, color=OK, marker="D", zorder=5, label="Envelope")
    ax[1].plot(cs, f_pow(cs, *pp), lw=2.2, color=INK,
               label=f"Power law ($R^2$={cmp_C['power']['R2']:.3f}, LOOCV {cmp_C['power']['LOOCV']:.3f})")
    ax[1].plot(cs, f_log(cs, *pl), "--", lw=2.2, color=ALT,
               label=f"Reciprocal-log ($R^2$={cmp_C['log']['R2']:.3f}, LOOCV {cmp_C['log']['LOOCV']:.3f})")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Training compute $C$ (FLOP)")
    ax[1].set_ylabel("Validation loss")
    ax[1].set_title("Functional form on the decision-relevant object"); ax[1].legend(fontsize=7.2)
    plt.tight_layout(); plt.savefig(OUT_FIG + "fig5_validation.png"); plt.close()

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT_RES, exist_ok=True)
    N_all, D_all, L_all, C_all = load()
    N, D, L, C = drop_outliers(N_all, D_all, L_all, C_all)

    # --- estimate and FREEZE -------------------------------------------------
    frozen, raw = fit_source_protocol(N, D, L, grid_search=True)
    frozen, raw = fit_source_protocol(N, D, L, grid_search=False, p0=raw)   # refine
    path = Path(frozen)
    published = dict(A=482.01, B=2085.43, E=1.8172, alpha=0.3478, beta=0.3658)

    # --- decision quantities -------------------------------------------------
    hurdles = [0.04, 0.03, 0.02, 0.01]
    thresholds = {}
    for h in hurdles:
        t = path.threshold(h)
        thresholds[f"{int(h*100)}%"] = dict(
            N=float(t), log10N=float(np.log10(t)) if np.isfinite(t) else None,
            status=("interpolated" if N_all.min() <= t <= N_all.max()
                    else "extrapolated_below" if t < N_all.min() else "extrapolated_above"))
    gains = {f"{n:.0e}": float(path.gain(path.C_of_N(n)))
             for n in [1e8, 3e8, 1e9, 3e9, 1e10]}
    dC = {f"{n:.0e}": float(path.dC_per_1pct(n)) for n in [1e8, 1e9, 1e10]}

    # --- empirical validation: compute-matched envelope ----------------------
    envC, envL = envelope(C_all, L_all, nbands=15)   # full sample: outliers are high-loss
    cmp_C = compare_forms(envC, envL, P0_POW_C, P0_LOG_C, BND_LOG_C, LOG_STARTS_C)
    ppC = cmp_C["power"]["params"]
    gain_env = lambda c: (f_pow(c, *ppC) - f_pow(2 * c, *ppC)) / f_pow(c, *ppC)
    validation = {}
    for h in [0.04, 0.03, 0.02]:
        c_emp = brentq(lambda c: gain_env(c) - h, 1e15, 1e32)
        c_mod = path.C_of_N(path.threshold(h))
        # The comparison in COMPUTE is the independent one. Translating to model
        # size uses the path's own budget-to-size map, which compresses the
        # discrepancy by the frontier elasticity and overstates the agreement.
        validation[f"{int(h*100)}%"] = dict(
            compute_model=float(c_mod), compute_empirical=float(c_emp),
            compute_ratio=float(c_mod / c_emp),
            size_model=float(path.N(c_mod)), size_empirical=float(path.N(c_emp)),
            size_ratio=float(path.N(c_mod) / path.N(c_emp)))

    # --- methodological finding: the wrong object hides the difference -------
    envN, envLN = envelope(N_all, L_all, nbands=15)
    cmp_N = compare_forms(envN, envLN, P0_POW_N, P0_LOG_N, BND_LOG_N, LOG_STARTS_N)

    # --- correspondence between the elasticity hurdle and the monetary rule ---
    # For a fixed price and loss function, each hurdle selects a scale; evaluating
    # V1* = c_eff * dC at that scale gives the value of a 1% improvement that an
    # organization adopting the hurdle is implicitly assuming. This reproduces the
    # same stopping point; it does not identify the organization's actual
    # preference.
    c_eff = C_COMPUTE + P_CO2 / 1e6 * E_FLOP / 3.6e6 * PUE * CI_GRID
    correspondence = {}
    for h in [0.04, 0.03, 0.02]:
        n = path.threshold(h)
        dc = path.dC_per_1pct(n)
        correspondence[f"{int(h*100)}%"] = dict(
            N_star=float(n), dC=float(dc), V1_star=float(c_eff * dc))

    # --- sensitivity of the envelope to the number of bands ------------------
    band_sensitivity = {}
    for nb in [10, 12, 15, 20, 25]:
        try:
            eC, eL = envelope(C_all, L_all, nbands=nb)
            cmp_nb = compare_forms(eC, eL, P0_POW_C, P0_LOG_C, BND_LOG_C, LOG_STARTS_C)
            pnb = cmp_nb["power"]["params"]
            g_nb = lambda c: ((f_pow(c, *pnb) - f_pow(2 * c, *pnb)) / f_pow(c, *pnb))
            c_emp4 = brentq(lambda c: g_nb(c) - 0.04, 1e15, 1e32)
            band_sensitivity[str(nb)] = dict(
                n_points=int(len(eC)),
                power_LOOCV=cmp_nb["power"]["LOOCV"], log_LOOCV=cmp_nb["log"]["LOOCV"],
                power_preferred=bool(cmp_nb["power"]["LOOCV"] < cmp_nb["log"]["LOOCV"]),
                compute_ratio_4pct=float(path.C_of_N(path.threshold(0.04)) / c_emp4))
        except Exception as exc:
            band_sensitivity[str(nb)] = dict(error=str(exc))

    # --- decomposition, robustness ------------------------------------------
    decomp = decomposition(path, N_all, L_all, C_all, L, C)
    mc = monte_carlo(N, D, L, raw)
    boot = bootstrap(N, D, L, raw)

    results = dict(
        frozen_parameters=frozen,
        published_reference=published,
        parameter_deviation_pct={k: float(100 * (frozen[k] - published[k]) / published[k])
                                 for k in published},
        frontier_elasticity_a=float(path.a),
        compute_exponent=float(1 / path.a),
        N_doubling_compute_factor=float(2 ** (1 / path.a)),
        observed_N_range=[float(N_all.min()), float(N_all.max())],
        gains_per_compute_doubling=gains,
        thresholds=thresholds,
        dC_per_1pct=dC,
        empirical_validation=dict(
            envelope_compute_matched=cmp_C,
            thresholds_model_vs_empirical=validation,
            envelope_by_model_size=cmp_N,
            note=("The compute-matched envelope is the empirical counterpart of a "
                  "locus derived under a budget constraint and is the object the "
                  "decision is made against, and agreement is reported in compute because the "
                  "translation to model size borrows the map being checked. On it the power law is decisively "
                  "preferred. On an envelope defined over model size the two forms "
                  "appear statistically indistinguishable; that comparison is not "
                  "decision-relevant and should not be used to license extrapolation.")),
        economic_parameters=dict(c_compute=C_COMPUTE, p_CO2=P_CO2, e_flop=E_FLOP,
                                 PUE=PUE, CI_grid=CI_GRID, c_eff=float(c_eff),
                                 note="Scenario inputs, not measurements; see source comments."),
        hurdle_value_correspondence=correspondence,
        envelope_band_sensitivity=band_sensitivity,
        sample_composition=sample_composition(path, L_all, C_all),
        decomposition=decomp,
        monte_carlo_4pct=mc,
        bootstrap_CI95=boot)

    with open(OUT_RES + "results.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame([dict(N=n, C=path.C_of_N(n), L=path.L(path.C_of_N(n)),
                       gain=path.gain(path.C_of_N(n)))
                  for n in [1e8, 3e8, 1e9, 3e9, 1e10]]).to_csv(
        OUT_RES + "table_marginal_returns.csv", index=False)
    pd.DataFrame(dict(C=envC, loss=envL)).to_csv(
        OUT_RES + "envelope_compute_matched.csv", index=False)
    _r, _d = _excess_ratios(path, L_all, C_all, C_all.min(), C_all.max())
    pd.DataFrame(dict(excess_compute_ratio=_r)).to_csv(
        OUT_RES + "excess_compute_ratios.csv", index=False)

    figures(path, N_all, L_all, C_all, envC, envL, cmp_C, hurdles, results)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
