"""
Reproducible analysis pipeline for:
"Sustainable Resource Allocation in LLM Training: A Production Engineering
Framework with Efficiency Frontier Analysis"

Single entry point: produces all fitted parameters, the compute-optimal
frontier, marginal-gain thresholds, a real Monte Carlo robustness test, and
every figure / corrected table from one dataset. No manual processing.

Data: reconstructed Hoffmann et al. (2022) points from Besiroglu et al. (2024),
Epoch AI repo `epoch-research/analyzing-chinchilla`.
"""
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares, minimize_scalar

RNG = np.random.default_rng(7)
DATA = "data/llm_scaling_dataset.csv"
OUT_FIG = "figures/"
OUT_RES = "results/"

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def load():
    d = pd.read_csv(DATA)
    return d.N.values, d.D.values, d.loss.values

# ----------------------------------------------------------------------
# (1) Full Chinchilla law L(N,D) = E + A/N^a + B/D^b  (Huber, log-domain)
#     Answers Reviewer 2: uses real (N, D, C) jointly.
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
# (2) True compute-optimal frontier: for each budget C, N* = argmin_N L(N, C/6N)
# ----------------------------------------------------------------------
def frontier(p, Cs):
    Nf, Lf = [], []
    for C in Cs:
        f = lambda ln: L_ND(10 ** ln, C / (6 * 10 ** ln), p)
        r = minimize_scalar(f, bounds=(6, 11), method="bounded")
        Nf.append(10 ** r.x); Lf.append(f(r.x))
    o = np.argsort(Nf)
    return np.array(Nf)[o], np.array(Lf)[o]

# ----------------------------------------------------------------------
# Candidate single-variable forms (fitted to the frontier curve)
# ----------------------------------------------------------------------
def log_model(N, c, b, E):   # reciprocal-log saturation (this paper)
    return c / (np.log(N) + b) + E
def power_law(N, A, beta, E):
    return A * np.power(N, -beta) + E

def gof(f, p, N, L, k):
    pred = f(N, *p); n = len(L); rss = np.sum((L - pred) ** 2)
    r2 = 1 - rss / np.sum((L - L.mean()) ** 2)
    mape = np.mean(np.abs((L - pred) / L)) * 100
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return dict(R2=r2, MAPE=mape, AIC=aic, BIC=bic)

# ----------------------------------------------------------------------
# (3) Marginal gain per doubling along the frontier & threshold by hurdle
# ----------------------------------------------------------------------
def gain_curve(Nf, Lf, ns):
    return np.array([(np.interp(x, Nf, Lf) - np.interp(2 * x, Nf, Lf)) /
                     np.interp(x, Nf, Lf) for x in ns])

def threshold(ns, g, hurdle):
    ix = np.where(g < hurdle)[0]
    return ns[ix[0]] if len(ix) else np.nan

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    import os
    for d in (OUT_FIG, OUT_RES):
        os.makedirs(d, exist_ok=True)
    N, D, L = load()
    p = fit_ND(N, D, L)
    predND = L_ND(N, D, p)
    p["R2"] = float(1 - np.sum((L - predND) ** 2) / np.sum((L - L.mean()) ** 2))

    Cs = np.logspace(18, 22, 60)
    Nf, Lf = frontier(p, Cs)
    ns = np.logspace(8, 10, 400)
    g = gain_curve(Nf, Lf, ns)

    hurdles = [0.02, 0.04, 0.06, 0.08]
    th = {f"{int(h*100)}%": float(threshold(ns, g, h)) for h in hurdles}

    # single-variable fits to the frontier (for the theoretical narrative)
    pl, _ = curve_fit(log_model, Nf, Lf, p0=[15, -5, 2.0],
                      bounds=([0, -8, 1.0], [60, 5, 2.4]), maxfev=200000)
    pp, _ = curve_fit(power_law, Nf, Lf, p0=[400, 0.34, 1.7], maxfev=200000)
    g_log = gof(log_model, pl, Nf, Lf, 3)
    g_pow = gof(power_law, pp, Nf, Lf, 3)

    # (4) Real Monte Carlo: +/-2% loss noise -> refit L(N,D) -> threshold@4%
    mc = []
    for _ in range(800):
        Lp = L * (1 + RNG.normal(0, 0.02, len(L)))
        try:
            p2 = fit_ND(N, D, Lp)
            nf, lf = frontier(p2, Cs)
            gg = gain_curve(nf, lf, ns)
            t = threshold(ns, gg, 0.04)
            if not np.isnan(t):
                mc.append(np.log10(t))
        except Exception:
            pass
    mc = np.array(mc)
    mc_summary = dict(median=float(np.median(mc)),
                      ci_low=float(np.percentile(mc, 2.5)),
                      ci_high=float(np.percentile(mc, 97.5)))

    results = dict(L_ND=p, threshold_by_hurdle=th,
                   gain_at_1e9=float(np.interp(1e9, ns, g)),
                   frontier_loss={f"1e{int(np.log10(x))}": float(np.interp(x, Nf, Lf))
                                  for x in [1e8, 1e9, 1e10]},
                   log_fit=dict(c=pl[0], b=pl[1], E=pl[2], **g_log),
                   power_fit=dict(A=pp[0], beta=pp[1], E=pp[2], **g_pow),
                   monte_carlo_threshold_4pct=mc_summary,
                   N_range=[float(N.min()), float(N.max())])
    with open(OUT_RES + "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # corrected Table 1 (regime structure) from the frontier
    rows = []
    for a, bnd in [(1e8, 2e8), (2e8, 5e8), (5e8, 1e9), (1e9, 2e9),
                   (2e9, 5e9), (5e9, 1e10)]:
        gn = (np.interp(a, Nf, Lf) - np.interp(bnd, Nf, Lf)) / np.interp(a, Nf, Lf)
        rows.append(dict(N_from=a, N_to=bnd, L_from=float(np.interp(a, Nf, Lf)),
                         gain_pct=round(gn * 100, 2)))
    pd.DataFrame(rows).to_csv(OUT_RES + "table1_corrected.csv", index=False)

    _figures(N, D, L, p, Nf, Lf, ns, g, pl, pp, th, mc)
    print(json.dumps(results, indent=2))

def _figures(N, D, L, p, Nf, Lf, ns, g, pl, pp, th, mc):
    # Fig 1: data + frontier + fits
    plt.figure(figsize=(7, 5))
    plt.scatter(N, L, s=10, alpha=.25, color="#888", label="Reconstructed points (n=245)")
    plt.plot(Nf, Lf, lw=2.5, color="#1f3b73", label="Compute-optimal frontier  L*(N)")
    plt.plot(Nf, log_model(Nf, *pl), "--", color="#c0392b", lw=1.8, label="Log saturation fit")
    plt.plot(Nf, power_law(Nf, *pp), ":", color="#27733b", lw=1.8, label="Power-law fit")
    plt.xscale("log"); plt.xlabel("Model parameters N"); plt.ylabel("Validation loss")
    plt.title("Validation loss vs. model size (Chinchilla reconstruction)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT_FIG + "fig1_fit.png", dpi=160); plt.close()

    # Fig 2: marginal gain + hurdle band
    plt.figure(figsize=(7, 5))
    plt.plot(ns, g * 100, lw=2.2, color="#1f3b73")
    for h, c in zip([0.02, 0.04, 0.06, 0.08], ["#bbb", "#c0392b", "#999", "#bbb"]):
        plt.axhline(h * 100, ls="--", lw=1, color=c)
        t = th[f"{int(h*100)}%"]
        if not np.isnan(t):
            plt.axvline(t, ls=":", lw=.8, color=c)
    plt.xscale("log"); plt.xlabel("Model parameters N")
    plt.ylabel("Marginal loss reduction per compute doubling (%)")
    plt.title("Efficiency frontier: diminishing marginal gains")
    plt.tight_layout(); plt.savefig(OUT_FIG + "fig2_marginal_gain.png", dpi=160); plt.close()

    # Fig 3: Monte Carlo threshold distribution
    plt.figure(figsize=(7, 5))
    plt.hist(mc, bins=30, color="#1f3b73", alpha=.8)
    plt.axvline(np.median(mc), color="#c0392b", lw=2, label=f"median ≈ 10^{np.median(mc):.2f}")
    plt.xlabel("log10(threshold N*) at 4% hurdle"); plt.ylabel("Frequency")
    plt.title("Monte Carlo robustness (±2% loss noise, 800 runs)")
    plt.legend(fontsize=9); plt.tight_layout(); plt.savefig(OUT_FIG + "fig3_montecarlo.png", dpi=160); plt.close()

    # Fig 4: threshold sensitivity to hurdle
    hs = np.linspace(0.02, 0.08, 25)
    Ts = [threshold(ns, g, h) for h in hs]
    plt.figure(figsize=(7, 5))
    plt.plot(hs * 100, Ts, "-o", color="#1f3b73", ms=4)
    plt.yscale("log"); plt.xlabel("Hurdle rate (%)"); plt.ylabel("Efficiency threshold N*")
    plt.title("Threshold is hurdle-dependent (the dominant source of variation)")
    plt.tight_layout(); plt.savefig(OUT_FIG + "fig4_hurdle_sensitivity.png", dpi=160); plt.close()

if __name__ == "__main__":
    main()
