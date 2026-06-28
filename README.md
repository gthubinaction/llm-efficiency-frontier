# LLM Efficiency Frontier — reproducible analysis

Reproducibility package for a production-engineering framework that converts
**established LLM scaling laws** (power-law / Chinchilla) into **resource- and
energy-allocation decisions** under joint **cost** and **carbon** constraints —
identifying, as a context-dependent decision rule, the point at which vertical
scaling stops being efficient.

One command regenerates every figure, table and fitted parameter from a single
dataset.

## Contribution (scope note)
The empirical engine is the established power-law scaling of validation loss.
The contribution is the **decision framework layered on top of it**: an economic
hurdle-rate criterion and a carbon-intensity term that together answer a
question scaling laws alone do not — *when to stop scaling* — and translate it
into an operational threshold (Kaizen vs. Kaikaku, multi-criteria utility).

The reciprocal-logarithmic saturation form is included as a **theoretically
motivated, conservative bound** grounded in thermodynamic/material limits, **not**
as a superior empirical fit. Over the observed range it is not claimed to
outperform the power law (see "Honest summary" below).

## Data provenance
Reconstructed Hoffmann et al. (2022) points from **Besiroglu et al. (2024)**,
Epoch AI repo `epoch-research/analyzing-chinchilla`. See `data/README_data.md`.
245 points, N ∈ [5.7e7, 1.6e10], single tokenizer/dataset (consistent loss scale).

## Reproduce
```bash
pip install -r requirements.txt
PYTHONPATH=. python src/analysis.py
```
Outputs: `results/results.json`, `results/table1_corrected.csv`,
`figures/fig1..fig4`. The notebook `notebooks/analysis.ipynb` mirrors the
pipeline with narrative.

## What the pipeline does
1. Fits the full Chinchilla law L(N,D)=E+A/N^a+B/D^b (Huber, log-domain) to the
   245 real points — addresses the (N, D, C) critique with real data.
2. Derives the true compute-optimal frontier N*(C)=argmin_N L(N, C/6N).
3. Computes marginal loss reduction per compute doubling and the efficiency
   threshold as a function of the economic hurdle rate (2–8%).
4. Fits the reciprocal-log and power-law forms to the frontier and compares
   them honestly (R², MAPE, AIC, BIC).
5. Runs a real Monte Carlo robustness test (±2% loss noise, refit, threshold).

## Honest summary of findings
- The efficiency threshold sits at the order of 1e9 (≈2.7e9 at a 4% hurdle),
  **inside** the observed range — interpolation, not extrapolation.
- The dominant source of threshold uncertainty is the **hurdle rate**, not data
  noise: the Monte Carlo CI is tight (~1.9e9–3.0e9), while the hurdle band spans
  ~2e8–8e9.
- Over the observed ~2.5 orders of magnitude in N, the reciprocal-log form is
  **not** empirically superior to the power law and is weakly identified on this
  range. Its role is theoretical (bounded saturation under thermodynamic limits).
  No claim of statistical superiority is made.

## Citation
See `CITATION.cff`. Please also cite Hoffmann et al. (2022), Besiroglu et al.
(2024) and Kaplan et al. (2020) for the underlying scaling laws and data.

## License
MIT — see `LICENSE`.
