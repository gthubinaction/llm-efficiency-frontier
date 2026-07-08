"""
Reproducible analysis pipeline (v2) for:
"Sustainable Resource Allocation in LLM Training: A Production Engineering
Framework with Efficiency Frontier Analysis"

Single entry point: produces all fitted parameters, the compute-optimal
frontier, marginal-gain thresholds, a Monte Carlo robustness test, and every
figure / table from one dataset. No manual processing.

Data: reconstructed Hoffmann et al. (2022) points from Besiroglu et al. (2024),
Epoch AI repo `epoch-research/analyzing-chinchilla`.

Changelog v2 (vs. v1):
  [FIX-1] Compute-budget grid extended to 1e23 FLOP so that the marginal-gain
          curve g(N) = [L*(N) - L*(2N)] / L*(N) never evaluates L* beyond the
          frontier grid. In v1, np.interp silently clamped L*(2N) to the last
          grid value for N > Nf.max/2, inflating apparent gains decay: the
          reported 0.9% at N=1e10 is actually ~2.7%, and the 2% threshold is
          ~2.5e10 (outside the observed N range -> extrapolation), not 7.8e9.
  [FIX-2] Thresholds are now flagged as interpolated vs. extrapolated relative
          to the observed N range of the data.
  [FIX-3] The logarithmic form is implemented exactly as written in the
          manuscript, L(N) = c / ln(a*N + b) + E, and its parameter standard
          errors are reported. Over this range b is unidentifiable (SE >> |b|),
          which is documented in results.json rather than hidden by bounds.
  [NEW-1] Non-circular model comparison: both single-variable forms are also
          fitted to the 15-point EMPIRICAL frontier (real envelope points),
          with leave-one-out cross-validation. The fit to the model-derived
          frontier is kept but explicitly labelled as such (a power law fitted
          to the frontier of a power law is near-exact by construction).
  [NEW-2] The deterministic-vs-Monte-Carlo threshold gap is quantified and
          explained: refitting under noise biases the threshold slightly
          downward (Jensen-type effect of the nonlinear refit), which is why
          the MC median (~2.4e9) sits below the deterministic value (~2.7e9).
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares, minimize_scalar, brentq

RNG = np.random.default_rng(7)
DATA = "data/llm_scaling_dataset.csv"
DATA_EMP_FRONTIER = "data/llm_scaling_frontier.csv"
OUT_FIG = "figures/"
OUT_RES = "results/"

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def load():
    d = pd.read_csv(DATA)
    return d.N.values, d.D.values, d.loss.values

def load_empirical_frontier():
    d = pd.read_csv(DATA_EMP_FRONTIER)
    return d.N.values, d.loss.values

# ----------------------------------------------------------------------
# (1) Full Chinchilla law L(N,D) = E + A/N^a + B/D^b  (Huber, log-domain)
#     Uses real (N, D, C) jointly.
# ----------------------------------------------------------------------
def fit_ND(N, D, L):
    def resid(t):
        E, lA, a, lB, b = t
        p = E + np.exp(lA) * N ** -a + np.exp(lB) * D ** -b
        return np.log(p) - np.log(L)
    s = least_squares(resid, [np.log(1.7), np.log(400), 0.34, np.log(400), 0.28],
                      loss="huber", f_scale=0.01, max_nfev=20000)
    E, lA, a, lB, b = s.x
    return dict(E=E, A=np.exp(lA), alpha=a, B=np.exp(lB), beta=b)

def L_ND(n, dd, p):
    return p["E"] + p["A"] * n ** -p["alpha"] + p["B"] * dd ** -p["beta"]

# ----------------------------------------------------------------------
# (2) Compute-optimal frontier: for each budget C, N* = argmin_N L(N, C/6N)
#     [FIX-1] Budgets extend to 1e23 so 2N stays inside the grid for all
#     N <= 1e10 evaluated downstream (Nf.max ~ 4.6e10 > 2e10).
# ----------------------------------------------------------------------
def frontier(p, Cs=None):
    if Cs is None:
        Cs = np.logspace(18, 23.5, 90)
    Nf, Lf = [], []
    for C in Cs:
        f = lambda ln: L_ND(10 ** ln, C / (6 * 10 ** ln), p)
        r = minimize_scalar(f, bounds=(6, 12), method="bounded")
        Nf.append(10 ** r.x); Lf.append(f(r.x))
    o = np.argsort(Nf)
    return np.array(Nf)[o], np.array(Lf)[o]

# ----------------------------------------------------------------------
# Candidate single-variable forms
#   [FIX-3] log_model is now EXACTLY the manuscript's form.
# ----------------------------------------------------------------------
def log_model(N, c, a, b, E):     # L(N) = c / ln(aN + b) + E  (manuscript form)
    return c / np.log(a * N + b) + E

def power_law(N, A, beta, E):
    return A * np.power(N, -beta) + E

LOG_P0 = [20.0, 1e-6, 10.0, 1.5]
LOG_BOUNDS = ([0, 1e-12, 0, 0.5], [100, 1.0, 1e6, 2.4])
POW_P0 = [400, 0.34, 1.7]

def gof(f, p, N, L, k):
    pred = f(N, *p); n = len(L); rss = np.sum((L - pred) ** 2)
    r2 = 1 - rss / np.sum((L - L.mean()) ** 2)
    mape = np.mean(np.abs((L - pred) / L)) * 100
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return dict(R2=float(r2), MAPE=float(mape), AIC=float(aic), BIC=float(bic))

def loocv_rmse(f, p0, bounds, N, L):
    """[NEW-1] Leave-one-out CV on real points. Returns RMSE of held-out
    predictions; np.nan if any refit fails."""
    errs = []
    n = len(N)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        try:
            if bounds is None:
                pi, _ = curve_fit(f, N[m], L[m], p0=p0, maxfev=400000)
            else:
                pi, _ = curve_fit(f, N[m], L[m], p0=p0, bounds=bounds, maxfev=400000)
            errs.append(L[i] - f(N[i], *pi))
        except Exception:
            return float("nan")
    return float(np.sqrt(np.mean(np.array(errs) ** 2)))

# ----------------------------------------------------------------------
# (3) Marginal gain per doubling along the frontier & threshold by hurdle
#     Interpolation in log10(N) domain; guarded against extrapolation.
# ----------------------------------------------------------------------
def gain_curve(Nf, Lf, ns):
    lNf = np.log10(Nf)
    Ls = lambda x: np.interp(np.log10(x), lNf, Lf)
    g = np.full(len(ns), np.nan)
    ok = (ns >= Nf.min()) & (2 * ns <= Nf.max())   # [FIX-1] no silent clamping
    g[ok] = (Ls(ns[ok]) - Ls(2 * ns[ok])) / Ls(ns[ok])
    return g

def threshold(ns, g, hurdle):
    valid = ~np.isnan(g)
    ix = np.where(valid & (g < hurdle))[0]
    return float(ns[ix[0]]) if len(ix) else float("nan")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    for d in (OUT_FIG, OUT_RES):
        os.makedirs(d, exist_ok=True)
    N, D, L = load()
    N_obs_min, N_obs_max = float(N.min()), float(N.max())
    p = fit_ND(N, D, L)
    predND = L_ND(N, D, p)
    p["R2"] = float(1 - np.sum((L - predND) ** 2) / np.sum((L - L.mean()) ** 2))

    Nf, Lf = frontier(p)
    ns = np.logspace(8, 10.7, 500)
    g = gain_curve(Nf, Lf, ns)

    hurdles = [0.02, 0.04, 0.06, 0.08]
    th = {}
    for h in hurdles:
        t = threshold(ns, g, h)
        th[f"{int(h*100)}%"] = dict(
            N=t,
            log10N=float(np.log10(t)) if not np.isnan(t) else float("nan"),
            within_observed_range=bool(N_obs_min <= t <= N_obs_max)  # [FIX-2]
        )

    # --- single-variable fits to the MODEL-DERIVED frontier (labelled) ------
    pl, cov_l = curve_fit(log_model, Nf, Lf, p0=LOG_P0, bounds=LOG_BOUNDS,
                          maxfev=400000)
    pp, cov_p = curve_fit(power_law, Nf, Lf, p0=POW_P0, maxfev=400000)
    se_l = np.sqrt(np.diag(cov_l))
    g_log = gof(log_model, pl, Nf, Lf, 4)
    g_pow = gof(power_law, pp, Nf, Lf, 3)

    # --- [NEW-1] non-circular comparison on the 15 EMPIRICAL frontier points -
    Ne, Le = load_empirical_frontier()
    ple, _ = curve_fit(log_model, Ne, Le, p0=LOG_P0, bounds=LOG_BOUNDS,
                       maxfev=400000)
    ppe, _ = curve_fit(power_law, Ne, Le, p0=POW_P0, maxfev=400000)
    emp = dict(
        n_points=int(len(Ne)),
        log_fit=dict(params=dict(c=float(ple[0]), a=float(ple[1]),
                                 b=float(ple[2]), E=float(ple[3])),
                     **gof(log_model, ple, Ne, Le, 4),
                     LOOCV_RMSE=loocv_rmse(log_model, LOG_P0, LOG_BOUNDS, Ne, Le)),
        power_fit=dict(params=dict(A=float(ppe[0]), beta=float(ppe[1]),
                                   E=float(ppe[2])),
                       **gof(power_law, ppe, Ne, Le, 3),
                       LOOCV_RMSE=loocv_rmse(power_law, POW_P0, None, Ne, Le)),
        note=("Fits and LOOCV on the 15 real envelope points. This is the "
              "non-circular comparison; the fit to the model-derived frontier "
              "below is near-exact for the power law BY CONSTRUCTION."))

    # --- (4) Monte Carlo: +/-2% loss noise -> refit L(N,D) -> threshold@4% ---
    mc = []
    for _ in range(800):
        Lp = L * (1 + RNG.normal(0, 0.02, len(L)))
        try:
            p2 = fit_ND(N, D, Lp)
            nf, lf = frontier(p2)
            gg = gain_curve(nf, lf, ns)
            t = threshold(ns, gg, 0.04)
            if not np.isnan(t):
                mc.append(np.log10(t))
        except Exception:
            pass
    mc = np.array(mc)
    det_log10 = th["4%"]["log10N"]
    mc_summary = dict(
        median=float(np.median(mc)),
        ci_low=float(np.percentile(mc, 2.5)),
        ci_high=float(np.percentile(mc, 97.5)),
        n_valid=int(len(mc)),
        deterministic_log10=det_log10,
        # [NEW-2] the deterministic value need not equal the MC median:
        # the threshold is a nonlinear functional of the refitted law, so
        # symmetric loss noise produces a slightly asymmetric (downward-
        # shifted) threshold distribution.
        median_minus_deterministic=float(np.median(mc) - det_log10))

    results = dict(
        L_ND=p,
        observed_N_range=[N_obs_min, N_obs_max],
        threshold_by_hurdle=th,
        gain_at_1e9=float(np.interp(np.log10(1e9), np.log10(ns), g)),
        gain_at_1e10=float(np.interp(np.log10(1e10), np.log10(ns), g)),
        frontier_loss={f"1e{int(np.log10(x))}":
                       float(np.interp(np.log10(x), np.log10(Nf), Lf))
                       for x in [1e8, 1e9, 1e10]},
        model_derived_frontier_fits=dict(
            log_fit=dict(params=dict(c=float(pl[0]), a=float(pl[1]),
                                     b=float(pl[2]), E=float(pl[3])),
                         param_SE=dict(c=float(se_l[0]), a=float(se_l[1]),
                                       b=float(se_l[2]), E=float(se_l[3])),
                         identifiability_note=(
                             "b is unidentifiable over this range "
                             "(SE(b) >> |b|); aN >> b for all observed N, so "
                             "the form degenerates to c/ln(aN)+E. Reported "
                             "for transparency."),
                         **g_log),
            power_fit=dict(params=dict(A=float(pp[0]), beta=float(pp[1]),
                                       E=float(pp[2])), **g_pow),
            note=("Fitted to 80 model-derived frontier points. The power "
                  "law's near-perfect fit here is expected by construction "
                  "and is NOT evidence of empirical superiority; see "
                  "empirical_frontier_fits for the non-circular test.")),
        empirical_frontier_fits=emp,
        monte_carlo_threshold_4pct=mc_summary)

    with open(OUT_RES + "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # corrected Table 1 (regime structure) from the frontier
    rows = []
    for a_, b_ in [(1e8, 2e8), (2e8, 5e8), (5e8, 1e9), (1e9, 2e9),
                   (2e9, 5e9), (5e9, 1e10), (1e10, 2e10)]:
        lNf = np.log10(Nf)
        Ls = lambda x: np.interp(np.log10(x), lNf, Lf)
        gn = (Ls(a_) - Ls(b_)) / Ls(a_)
        rows.append(dict(N_from=a_, N_to=b_, L_from=float(Ls(a_)),
                         gain_pct=round(gn * 100, 2)))
    pd.DataFrame(rows).to_csv(OUT_RES + "table1_corrected.csv", index=False)

    _figures(N, L, Nf, Lf, ns, g, pl, pp, th, mc, N_obs_max)
    print(json.dumps(results, indent=2))

def _figures(N, L, Nf, Lf, ns, g, pl, pp, th, mc, N_obs_max):
    inrange = Nf <= N_obs_max

    # Fig 1: data + frontier + fits
    plt.figure(figsize=(7, 5))
    plt.scatter(N, L, s=10, alpha=.25, color="#888",
                label="Reconstructed points (n=245)")
    plt.plot(Nf[inrange], Lf[inrange], lw=2.5, color="#1f3b73",
             label="Compute-optimal frontier  L*(N)")
    plt.plot(Nf[inrange], log_model(Nf[inrange], *pl), "--", color="#c0392b",
             lw=1.8, label="Log saturation fit")
    plt.plot(Nf[inrange], power_law(Nf[inrange], *pp), ":", color="#27733b",
             lw=1.8, label="Power-law fit")
    plt.xscale("log"); plt.xlabel("Model parameters N")
    plt.ylabel("Validation loss")
    plt.title("Validation loss vs. model size (Chinchilla reconstruction)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUT_FIG + "fig1_fit.png", dpi=160); plt.close()

    # Fig 2: marginal gain + hurdle band
    plt.figure(figsize=(7, 5))
    plt.plot(ns, g * 100, lw=2.2, color="#1f3b73")
    for h, c in zip([0.02, 0.04, 0.06, 0.08],
                    ["#bbb", "#c0392b", "#999", "#bbb"]):
        plt.axhline(h * 100, ls="--", lw=1, color=c)
        t = th[f"{int(h*100)}%"]["N"]
        if not np.isnan(t):
            plt.axvline(t, ls=":", lw=.8, color=c)
    plt.axvline(N_obs_max, color="#333", lw=1.2, ls="-.",
                label="max observed N (beyond: extrapolation)")
    plt.xscale("log"); plt.xlabel("Model parameters N")
    plt.ylabel("Marginal loss reduction per compute doubling (%)")
    plt.title("Efficiency frontier: diminishing marginal gains")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUT_FIG + "fig2_marginal_gain.png", dpi=160); plt.close()

    # Fig 3: Monte Carlo threshold distribution
    plt.figure(figsize=(7, 5))
    plt.hist(mc, bins=30, color="#1f3b73", alpha=.8)
    plt.axvline(np.median(mc), color="#c0392b", lw=2,
                label=f"median ≈ 10^{np.median(mc):.2f}")
    plt.axvline(th["4%"]["log10N"], color="#27733b", lw=2, ls="--",
                label=f"deterministic ≈ 10^{th['4%']['log10N']:.2f}")
    plt.xlabel("log10(threshold N*) at 4% hurdle"); plt.ylabel("Frequency")
    plt.title("Monte Carlo robustness (±2% loss noise, 800 runs)")
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(OUT_FIG + "fig3_montecarlo.png", dpi=160); plt.close()

    # Fig 4: threshold sensitivity to hurdle
    hs = np.linspace(0.02, 0.08, 25)
    Ts = np.array([threshold(ns, g, h) for h in hs])
    plt.figure(figsize=(7, 5))
    plt.plot(hs * 100, Ts, "-o", color="#1f3b73", ms=4)
    plt.axhline(N_obs_max, color="#c0392b", lw=1.2, ls="--",
                label="max observed N (above: extrapolation)")
    plt.yscale("log"); plt.xlabel("Hurdle rate (%)")
    plt.ylabel("Efficiency threshold N*")
    plt.title("Threshold is hurdle-dependent (the dominant source of variation)")
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(OUT_FIG + "fig4_hurdle_sensitivity.png", dpi=160); plt.close()

if __name__ == "__main__":
    main()
