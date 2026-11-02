# Changelog

## 0.1.0

- Extract model-independent prior/likelihood power-scaling sensitivity from
  BayesianRegressionModels.jl.
- Expose `cjs_dist`, `powerscale_weights`, `powerscale_sensitivity`, and
  `PowerscaleSensitivity`, with Pareto-k and reliability diagnostics.
- Use PSIS.jl for all Pareto smoothing; sort each variable once per summary.
- Validate finite powers, efficiencies, thresholds, and named draw alignment.
- Include synthetic conjugate-model tests and reproducible reference checks.
