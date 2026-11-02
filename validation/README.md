# Numerical validation

The package tests check CJS identities and invariances, input preservation,
validation failures, agreement with PSIS.jl, and an analytic conjugate Normal
model. The analytic CJS oracle integrates the two closed-form posterior CDFs
on the empirical sample's finite support, independently of importance weights.

`references.jl` adds a comparison to the original BRM generic implementation
and to R `priorsense`, using only synthetic Normal models. It reads BRM's
generic definitions directly from a caller-supplied checkout; no BRM code or
model dependencies are bundled into this package.

With R `priorsense` installed, run from the repository root:

```julia
using Pkg
Pkg.test(; test_args=["references", "/path/to/BayesianRegressionModels.jl", "/path/to/output"])
```

The BRM acceptance tolerance is `atol=rtol=1e-8` for sensitivity and Pareto-k;
weight tolerance is `atol=1e-12, rtol=1e-8`. The R check passes Julia's weights
to R for a separate CJS comparison (`atol=1e-10, rtol=1e-8`), then compares the
complete R and Julia summaries (`atol=5e-4, rtol=0.005`) to accommodate the
different PSIS implementations in R `loo` and Julia `PSIS`.

The release verification receipt records the measured errors and package
versions. CI runs the portable Julia suite; the optional cross-language check
requires R and an external BRM checkout.
