# Changelog

## v3.0.0 — corrected marginal-gain metric and extrapolation bracket

**Breaking: the headline numbers of v2.0.0 were computed on a mislabelled metric.**

### Fixed
- **Marginal-gain metric.** v2 computed `g = [L*(N) − L*(2N)] / L*(N)` and labelled it
  "gain per compute doubling". On the compute-optimal frontier N grows as C^a with
  a ≈ 0.51, so N→2N corresponds to ≈3.4× the compute, not 2×. The metric is now
  defined per doubling of COMPUTE, which is the decision variable. Corrected gains:
  5.35% at N=1e8 (was 9.2%), 3.22% at 1e9 (was 5.3%), 1.71% at 1e10 (was 2.7%).
- **Thresholds.** At a 4% hurdle N* = 4.06e8 (v2 reported 2.7e9). At 2%, N* = 5.84e9
  (v2 reported 2.5e10, flagged extrapolated; now interpolated). Hurdles ≥6% now fall
  below the smallest observed model and are flagged as extrapolations — the direction
  of the interpolation/extrapolation flags is inverted relative to v2.
- **Estimation protocol.** v2 used a generic Huber `least_squares`, yielding β=0.476
  against the published 0.366. v3 replicates the source protocol exactly (Huber
  log-sum-exp, δ=1e-3, L-BFGS-B from a grid, on the 240 points that remain after the
  five high-loss outliers are excluded) and reproduces the published estimates to
  three decimals: α=0.3473, β=0.3672, E=1.8172, A=477.8, B=2143.4. Roughly 85% of the
  earlier divergence came from the outliers, ~15% from the estimator.
- **Frontier geometry.** With the corrected fit, C ∝ N^1.95 and D* ∝ N^0.95, so the
  near-quadratic geometry stated in the paper is now consistent with the fit (v2
  implied N^1.75 while claiming N²).
- **Closed-form frontier.** The path is evaluated analytically from the first-order
  condition rather than on a numerical grid, removing interpolation error and the
  clamping failure mode fixed in v2.0.0 by extending the grid.
- **Bootstrap.** Parameter CIs in the v2 development branch were artificially tight
  from early L-BFGS termination; tolerances are now set explicitly.

### Changed
- **The reciprocal-logarithmic form is the OPTIMISTIC arm, not a conservative bound.**
  Fitted to the empirical envelope its floor is E=1.33 against E=2.03 for the power
  law, so beyond the data it predicts larger remaining gains (0.98% vs 0.34% per
  compute doubling at N=1e11). Earlier releases described it as conservative; that
  characterisation was wrong. The two forms are now reported as an **extrapolation
  bracket**: indistinguishable within the observed range (0.5–1.4% apart), diverging
  beyond it (5.2–12.3%), with the gap between their stopping scales measuring how much
  of a recommendation rests on functional form rather than on evidence — 1.5× at a 2%
  hurdle, 12.7× at 1%.
- Monte Carlo perturbation is treated as a **sensitivity parameter** (±1%, ±2%, ±5%)
  rather than as a known digitisation error.
- Carbon reported through an explicit chain (ΔC → ×J/FLOP → ×PUE → ×grid CI) with
  hardware scenarios, instead of identifying FLOP with carbon intensity.
- Emissions internalised in the decision through an effective resource price
  c_eff = c_compute + p_CO2·e·PUE·CI. At 80 $/tCO2e the carbon term is 0.71% of the
  compute price: at prevailing prices emissions bind as budgets, not through prices.

### Notes
- `phase1_gate.json` is the single source of truth: parameters are frozen there before
  any downstream quantity is derived, and no value is inherited across versions.
- The interactive tool now exposes the extrapolation bracket directly.
