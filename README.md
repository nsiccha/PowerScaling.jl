# PowerScaling.jl

Prior and likelihood sensitivity analysis from posterior draws, using
[power-scaling](https://doi.org/10.1007/s11222-023-10366-5) and
[PSIS.jl](https://julia.arviz.org/PSIS/stable/).

```julia
using PowerScaling, Random

# Prior μ ~ Normal(0, 1), one observation y = 2 with standard error 1.
# The posterior is Normal(1, 1/√2), so no model-fitting dependency is needed.
mu = 1 .+ randn(Xoshiro(42), 4_000) ./ sqrt(2)
log_prior = -mu.^2 ./ 2
log_lik = -(mu .- 2).^2 ./ 2
result = powerscale_sensitivity(reshape(mu, :, 1), log_prior, log_lik;
                              names=[:mu])
display(result)
result.prior       # CJS distance gradient under prior perturbation
result.likelihood  # CJS distance gradient under likelihood perturbation
result.reliable    # Check both components before interpreting the diagnosis

weights = powerscale_weights(log_prior; alpha=1.01)
sum(weights.weights .* mu) # Mean under the strengthened prior
```

Inputs use **draws × variables** orientation. Log densities contain one joint
component value per draw, in the same order. The package does not fit models
or reconstruct priors. For a hierarchical model, explicitly choose which
prior levels to perturb and pass their log-density contribution.

The summary averages the two CJS distance gradients at `lower_alpha=0.99`
and `upper_alpha=1.01`. At the default threshold of 0.05, sensitivity to both
components flags potential prior-data conflict; prior sensitivity with a weak
likelihood response flags potential strong-prior/weak-likelihood behavior.
These are diagnostic indications, not hypothesis tests.

Inspect all four `pareto_k` values and `reliable`. A nonfinite k or k above
0.7 means reweighting is unreliable; do not interpret the resulting sensitivity
as evidence either way. `reff=1` assumes independent draws. For correlated
draws, supply the relative efficiency of the inverse importance ratios as
described by PSIS.jl; the summary accepts a common conservative value.
Moment matching and refitting are outside this package.

PowerScaling is unregistered. Once published, install the reviewed release
commit from `https://github.com/nsiccha/PowerScaling.jl`; pin its commit in the
consumer's resolver and use `[compat] PowerScaling = "0.1.0"`.
Julia 1.10 users need `Pkg.add(url=..., rev=...)` or `Pkg.develop(path=...)`;
Julia 1.11+ also supports a corresponding `[sources.PowerScaling]` entry.

Run the package tests with `julia --project=. -e 'using Pkg; Pkg.test()'`.
See the [validation notes](validation/README.md) for independent reference checks.

The generic analysis was extracted from BayesianRegressionModels.jl. Model
assembly stays with the calling package; Pareto smoothing is supplied entirely
by PSIS.jl. MIT licensed; the original BRM notice is retained in `LICENSE`.
