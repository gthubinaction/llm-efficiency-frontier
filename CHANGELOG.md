# Changelog

## v4.4.2 — figures at print resolution, Monte Carlo figure folded in

- All figures now render at 300 dpi (1920 px wide, 2880 for the two-panel one)
  instead of 170. In the manuscript they are placed at a uniform 5.6 in, giving
  about 343 effective dpi; previously Figure 1 was stretched to 6.0 in from a
  1088 px image, some 11% more than the others, which is why it looked softer.
- The Monte Carlo robustness figure is generated inside `figures()` rather than by
  a separate script, so every figure in the paper now comes from one run of this
  pipeline. Mixing pipeline figures with ad-hoc ones is how a stale value reached
  a caption in an earlier release.


## v4.4.1 — cost scenario arithmetic made explicit

- The compute price constant now carries its derivation: 2 $/h at a peak 1e15
  FLOP/s is 5.6e-19 $/FLOP if every peak operation were useful; model FLOPs
  utilisation of order 40% multiplies that by ~2.5, and storage, interconnect,
  checkpointing and abandoned runs add roughly a further 2x, giving the 3e-18
  adopted here, about 5.4x the peak-rate figure. Earlier releases stated the
  inputs and the adopted value without the factor between them, which made the
  choice look like a consequence of the arithmetic rather than an assumption.


## v4.4.0 — sample composition, individual ratios, scope checks

- `sample_composition` in results.json records exactly which observations each
  sample drops and how the two rules overlap: the two points excluded because
  inverting the law for their loss would leave the observed budget range (indices
  212 and 227, losses 4.67 and 5.01) are also among the five highest-loss points.
  That overlap is why dropping five points takes the count from 243 to 240 rather
  than to 238. Previously the counts were correct but unexplained.
- `excess_compute_ratios.csv` exports the individual ratios behind the median,
  p90, maximum and share above 2x, so the table can be reproduced point by point.
- The compute price is documented as a constructed scenario: an accelerator
  renting near 2 $/h at order 1e15 FLOP/s implies ~1e-18 $/FLOP; 3e-18 is adopted
  to allow for utilisation and overheads. Not a market survey.
- `verify_manuscript.py` gains SCOPE checks. Numeric checks cannot catch a claim
  that is arithmetically right but broader than the evidence: the band-sensitivity
  analysis showed the sign of the model-envelope difference is not stable, and the
  qualifier had reached only the results section. The verifier now requires a scope
  qualifier wherever the headline percentage appears, and requires the sample
  overlap to be explained. 37 checks.


## v4.3.0 — hurdle/value correspondence, band sensitivity, parameter provenance

- `hurdle_value_correspondence` in results.json: for each hurdle, the scale it
  selects, the finite increment dC there, and the implied break-even value of a 1%
  improvement (4% -> $12.76, 3% -> $175.77, 2% -> $5,064.82 at c_eff =
  3.0213e-18 $/FLOP). Earlier this correspondence was asserted in the manuscript
  without being computed anywhere in the pipeline; it is now emitted and checked.
- `envelope_band_sensitivity`: the compute-matched envelope is rebuilt with 10, 12,
  15, 20 and 25 logarithmic bands. The power law is preferred over the
  reciprocal-logarithmic form in every case (LOOCV 0.010-0.015 against 0.155-0.188),
  so the functional-form conclusion does not depend on the binning. The agreement
  between path and envelope stopping budgets holds to within about 20% in every
  case, but its SIGN is not stable: with 10 bands the envelope requires about 6%
  less compute than the path, with 15 bands about 18% more. Reported as such.
- Illustrative economic and energy parameters (compute price, carbon price, J/FLOP,
  PUE, grid intensity) are now named constants at the top of the module, emitted in
  results.json, and documented as scenario inputs rather than measurements.
- `verify_manuscript.py` extended to cover the correspondence table, the printed
  c_eff, and the presence of the band-sensitivity statement. 35 checks.


## v4.2.0 — manuscript verifier

- `verify_manuscript.py` checks every numeric claim in the manuscript against
  `results/results.json` and fails the build on any mismatch. Three check types:
  VALUE (number matches the pipeline), DIRECTION (a ratio points the way the
  pipeline says it points), ECHO (a quantity repeated across sections agrees with
  itself). It also asserts the absence of superseded claims.
- Motivation: across three revisions the algebraic core verified correct while the
  presentation layer drifted. In one revision a ratio was stated correctly in the
  results section and inverted in the abstract, discussion and conclusions. Manual
  propagation of a corrected number into four places has a high per-pass error rate
  that does not fall with care; the checks are now mechanical.
- Usage: `python verify_manuscript.py Manuscript.docx results/results.json`


## v4.1.0 — allocation sensitivity, figure regeneration

- `decomposition()` now reports a sensitivity table for the excess-compute
  diagnostic: full sample (n=243, median 1.08, p90 1.73, max 24.6, 7.0% above 2x)
  against the estimation sample without the five high-loss observations (n=240,
  median 1.08, p90 1.68, max 6.6, 5.8%). Points whose implied optimum falls
  outside the observed budget range are dropped and counted rather than silently
  included; the earlier maximum of 372 came from such a point.
- Monte Carlo figure added so that every figure in the paper is generated by this
  pipeline. Earlier releases mixed pipeline figures with ad-hoc ones, which is how
  a stale LOOCV value reached a published caption.
- Agreement between path and empirical envelope is reported as a ratio (0.85 at
  every hurdle) rather than as a percentage, since the percentage depends on which
  side is the denominator.


## v4.0.0 — compute-matched frontier, consolidated decision rule

**Breaking: the extrapolation bracket of v3.0.0 was withdrawn. It compared
incomparable objects and did not contain the main result.**

### Fixed
- **Comparison object.** v3 fitted the two candidate forms to an envelope defined
  over bands of MODEL SIZE. That object selects the best model of a given size
  irrespective of the budget that produced it, and is not comparable to a locus
  derived under a budget constraint. The "bracket" therefore did not contain the
  model-derived threshold: at a 2% hurdle the arms gave 1.1e9 and 1.6e9 while the
  path gave 5.8e9, a factor of 3.6 outside. The envelope is now built over bands
  of TRAINING COMPUTE (13 populated bands of 15).
- **Functional form.** On the compute-matched envelope the power law attains
  R2=0.999, MAPE 0.31%, LOOCV RMSE 0.014; the reciprocal-logarithmic form attains
  R2=0.826, MAPE 4.99%, LOOCV RMSE 0.155 - an elevenfold difference in
  out-of-sample error, dAIC ~ 72. On the model-size envelope the same two forms
  are close to indistinguishable (LOOCV 0.070 vs 0.072). The "statistically
  indistinguishable" claim of v3 was an artefact of the wrong object. The power
  law is adopted; the logarithmic form has no empirical support in range.
- **Agreement is reported in COMPUTE, not model size.** The stopping budgets of
  path and empirical envelope agree to within 15% (2.4e19 vs 2.8e19 FLOP at a 4%
  hurdle). Expressed as model sizes the gap looks like 8%, but that translation
  uses the path's own budget-to-size map and compresses the discrepancy by the
  frontier elasticity. The compute figure is the honest one.
- **N-doubling factor.** 2^(1/a) = 3.85, not 3.4. The 3.4 came from the discarded
  v2 estimator (a=0.571) and had been carried forward.

### Changed
- **Landauer gap withdrawn from the quantitative output.** Converting a floor on
  irreversible bit erasure into a bound per floating-point operation requires
  assumptions about erasures per operation that we cannot justify; numerical
  precision does not establish the equivalence. The constant remains in the
  source for reference but no reported result depends on it.
- **Allocation diagnostic qualified.** Realised compute over the compute the path
  requires for the same loss: median 1.08, p90 1.73, max 25, 7% above 2x. Two
  observations requiring inversion beyond the observed budgets are excluded. The
  reference locus is estimated, so the ratio mixes misallocation with fit
  residual and is reported as excess compute against an estimated reference,
  not as waste.
- **Replication claim.** The protocol recovers the published estimates to within
  0.4% on exponents and 2.8% on scale coefficients, with published values inside
  the bootstrap intervals. Earlier releases claimed agreement "to three decimals",
  which is not true of alpha, beta, A or B.
- Decision rule stated once, in discrete form: scaling is justified while V1
  exceeds c_eff * dC(N). The earlier continuous statement and the undemonstrated
  equivalence with the elasticity hurdle were removed.
- An absolute carbon cap bounds the reachable budget and stops the programme
  earlier along the same path; it does not alter the optimal input mix at a given
  budget, as earlier releases stated.

### Notes
- `results/results.json` is the single source of truth. Parameters are frozen
  after replicating the source protocol and before any downstream derivation.
- The interactive tool shows the consistency check in place of the bracket.
