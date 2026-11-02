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
different PSIS implementations used by R `priorsense` and Julia `PSIS`.

## Verification receipt (2026-11-02)

The command above passed offline on Julia 1.10.12 with PowerScaling 0.1.0,
PSIS 0.9.10, R 4.5.2, priorsense 1.4.0, and posterior 1.7.0: 116 portable
Julia assertions, 36 BRM comparisons, and 3 R comparisons (155 total).
The BRM checkout was `9b56261dbb2d6140c8c1040be9663b9c87a53e38`;
the two reference source files were unchanged from the original extraction
checkout `9210eeb2271dfbfade66361a99e6626edbfb8f77`.

| BRM comparison | Maximum absolute error |
| --- | ---: |
| Sensitivity | 4.599521935197415e-12 |
| Normalized PSIS weights | 9.75781955236954e-19 |
| Pareto-k | 7.951972413877684e-15 |

| R comparison | Julia | R priorsense |
| --- | ---: | ---: |
| CJS using the same weights | 0.0017880402235187765 | 0.0017880402235180347 |
| Prior sensitivity | 0.12457191859556127 | 0.12456797051043805 |
| Likelihood sensitivity | 0.13629035617558058 | 0.13627758078567165 |

CI runs the portable Julia suite; the optional cross-language check requires
R and an external BRM checkout. These comparisons cover synthetic Normal
models, not every posterior geometry or severe importance-sampling failure.

The docs environment was resolved offline with Documenter 1.19.0 and
DocumenterVitepress 0.3.9. Documenter's executable example, cross-references,
and `checkdocs=:exports` passed with `build_vitepress=false` and remote source
links disabled for the local check. The VitePress bundle and deployment are
left to the repository's Docs workflow after publication.
