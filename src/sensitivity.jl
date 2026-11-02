function _finite_vector(x, label)
    v = Float64.(collect(x))
    isempty(v) && throw(ArgumentError("$label must not be empty"))
    all(isfinite, v) || throw(ArgumentError("$label must contain only finite values"))
    return v
end

function _positive_reff(reff)
    r = Float64(reff)
    isfinite(r) && r > 0 || throw(ArgumentError("reff must be finite and positive"))
    return r
end

"""
    powerscale_weights(log_component; alpha, reff=1.0)

Pareto-smoothed weights for raising a prior or likelihood component to `alpha`.
The importance log ratios are `(alpha - 1) * log_component`. Supply one finite
joint log-density per posterior draw, in draw order. Additive constants cancel.
`alpha` must be finite and nonnegative; `reff` is the relative efficiency of
the inverse importance ratios, as in `PSIS.psis` (1 for independent draws).

Returns a named tuple with normalized `weights`, normalized `log_weights`,
`pareto_k`, `tail_length`, `ess`, and `alpha`. ESS is `reff / sum(abs2, weights)`;
it is descriptive only when the Pareto fit is reliable. A nonfinite Pareto-k
or a value above 0.7 signals unreliable reweighting. PSIS warnings are retained.

At `alpha == 1`, returns uniform weights, `pareto_k = -Inf`, `tail_length = 0`,
and `ess = reff * length(log_component)`. Otherwise at least 21 draws and a
nonconstant component are required, and the PSIS tail must have at least five
draws. Constant components cannot diagnose sensitivity by power-scaling.
Inputs are never modified.
"""
function powerscale_weights(log_component::AbstractVector; alpha::Real, reff::Real=1.0)
    a = Float64(alpha)
    isfinite(a) && a >= 0 || throw(ArgumentError("alpha must be finite and nonnegative"))
    r = _positive_reff(reff)
    component = _finite_vector(log_component, "log_component")
    n = length(component)
    if a == 1
        weights = fill(1.0 / n, n)
        return (; log_weights=fill(-log(n), n), weights, pareto_k=-Inf,
                tail_length=0, ess=r * n, alpha=a)
    end
    n >= 21 || throw(ArgumentError("power-scaling requires at least 21 draws"))
    maximum(component) - minimum(component) >= eps(Float64) ||
        throw(ArgumentError("log_component is constant; power-scaling cannot diagnose sensitivity"))
    ratios = (a - 1) .* component
    all(isfinite, ratios) || throw(ArgumentError("power-scaling log ratios overflowed"))
    result = PSIS.psis(ratios, r; normalize=true)
    result.tail_length >= 5 || throw(ArgumentError("PSIS needs a tail of at least five draws; collect more draws for this reff"))
    return (; log_weights=vec(result.log_weights), weights=vec(result.weights),
            pareto_k=Float64(result.pareto_shape), tail_length=result.tail_length,
            ess=Float64(PSIS.ess_is(result; bad_shape_nan=false)), alpha=a)
end

"""
    PowerscaleSensitivity

Per-variable power-scaling diagnostics, returned by [`powerscale_sensitivity`](@ref).
`variables`, `prior`, `likelihood`, and `diagnosis` are aligned vectors.
Diagnoses are `:prior_data_conflict` (both sensitivities ≥ `threshold`),
`:strong_prior_weak_likelihood` (prior > threshold, likelihood < threshold),
or `:none`. These are potential concerns, not hypothesis tests.

`pareto_k` contains `prior_lower`, `prior_upper`, `likelihood_lower`, and
`likelihood_upper`. `reliable.prior` and `reliable.likelihood` indicate whether
both Pareto-k values for that component are finite and ≤ 0.7. These flags
assess importance sampling, not convergence of the input posterior draws.
Unreliable sensitivities remain visible and must not be interpreted as evidence
of sensitivity or its absence.
"""
struct PowerscaleSensitivity
    variables::Vector{Symbol}
    prior::Vector{Float64}
    likelihood::Vector{Float64}
    diagnosis::Vector{Symbol}
    pareto_k::NamedTuple{(:prior_lower, :prior_upper, :likelihood_lower, :likelihood_upper),NTuple{4,Float64}}
    reliable::NamedTuple{(:prior, :likelihood),Tuple{Bool,Bool}}
    threshold::Float64
    lower_alpha::Float64
    upper_alpha::Float64
end

function Base.show(io::IO, ::MIME"text/plain", s::PowerscaleSensitivity)
    println(io, "Power-scaling sensitivity (CJS gradient; threshold ", s.threshold,
            "; α ∈ [", s.lower_alpha, ", ", s.upper_alpha, "])")
    println(io, rpad("variable", 24), rpad("prior", 12), rpad("likelihood", 12), "diagnosis")
    for i in eachindex(s.variables)
        println(io, rpad(string(s.variables[i]), 24),
                rpad(string(round(s.prior[i]; digits=4)), 12),
                rpad(string(round(s.likelihood[i]; digits=4)), 12), s.diagnosis[i])
    end
    print(io, "Pareto-k: ", s.pareto_k, "; reliable: ", s.reliable)
end

function _select_variables(draws, names, variables)
    labels = Symbol.(collect(names))
    length(labels) == size(draws, 2) || throw(DimensionMismatch("names must match the draw columns"))
    length(unique(labels)) == length(labels) || throw(ArgumentError("column names must be unique"))
    selected = isnothing(variables) ? labels : Symbol.(collect(variables))
    isempty(selected) && throw(ArgumentError("select at least one variable"))
    length(unique(selected)) == length(selected) || throw(ArgumentError("variable selection must be unique"))
    lookup = Dict(name => i for (i, name) in enumerate(labels))
    indices = map(selected) do name
        haskey(lookup, name) || throw(ArgumentError("unknown variable: $name"))
        lookup[name]
    end
    return selected, indices
end

# All four perturbations share each variable's support and sorting permutation.
function _reweighted_distance(x, uniform, w)
    return max(_cjs_onesided(x, uniform, x, w),
               _cjs_onesided(-reverse(x), uniform, -reverse(x), reverse(w)))
end

"""
    powerscale_sensitivity(draws, log_prior, log_lik;
        names=Symbol.("x" .* string.(1:size(draws, 2))), variables=nothing,
        lower_alpha=0.99, upper_alpha=1.01, threshold=0.05, reff=1.0)
    powerscale_sensitivity(draws, log_prior, log_lik, names; kwargs...)

Prior and likelihood sensitivity following Kallioinen et al. (2024).
`draws` is a matrix with one draw per row and one variable per column;
`log_prior` and `log_lik` contain the joint log densities for those same draws.
`names` labels the columns; `variables` selects and orders names (default: all).
The optional fourth positional argument is equivalent to the `names` keyword.

For each component, PSIS reweights draws at `lower_alpha` and `upper_alpha`.
The sensitivity is the mean of the one-sided CJS distance gradients:
`(D_lower / -log2(lower_alpha) + D_upper / log2(upper_alpha)) / 2`.
Require `0 < lower_alpha < 1 < upper_alpha` and a finite nonnegative threshold.
`reff` is passed to all four PSIS fits; use 1 for independent draws. When the
four inverse-ratio efficiencies differ, supply their conservative minimum.
The lower-level [`powerscale_weights`](@ref) accepts a separate value per call.

Only the supplied density components are scaled. For hierarchical models,
choose the prior levels to perturb explicitly, and evaluate those components
in a consistent parameterization. Model assembly, density evaluation, refits,
and moment matching belong to the caller. Check `reliable` before interpreting
the returned [`PowerscaleSensitivity`](@ref).
"""
function powerscale_sensitivity(draws::AbstractMatrix, log_prior::AbstractVector,
                                log_lik::AbstractVector;
                                names=[Symbol("x", i) for i in axes(draws, 2)],
                                variables=nothing, lower_alpha::Real=0.99,
                                upper_alpha::Real=1.01, threshold::Real=0.05,
                                reff::Real=1.0)
    lo, hi, cutoff = Float64(lower_alpha), Float64(upper_alpha), Float64(threshold)
    0 < lo < 1 || throw(ArgumentError("require 0 < lower_alpha < 1"))
    isfinite(hi) && hi > 1 || throw(ArgumentError("upper_alpha must be finite and greater than 1"))
    isfinite(cutoff) && cutoff >= 0 || throw(ArgumentError("threshold must be finite and nonnegative"))
    r = _positive_reff(reff)
    n = size(draws, 1)
    length(log_prior) == n || throw(DimensionMismatch("log_prior must have one value per draw"))
    length(log_lik) == n || throw(DimensionMismatch("log_lik must have one value per draw"))
    selected, indices = _select_variables(draws, names, variables)
    all(i -> all(isfinite, view(draws, :, i)), indices) ||
        throw(ArgumentError("selected draws must be finite"))
    p_lo = powerscale_weights(log_prior; alpha=lo, reff=r)
    p_hi = powerscale_weights(log_prior; alpha=hi, reff=r)
    l_lo = powerscale_weights(log_lik; alpha=lo, reff=r)
    l_hi = powerscale_weights(log_lik; alpha=hi, reff=r)
    prior, likelihood = zeros(length(indices)), zeros(length(indices))
    uniform = fill(1.0 / n, n)
    for (j, i) in enumerate(indices)
        x = _finite_vector(view(draws, :, i), "draws")
        order = sortperm(x)
        x = x[order]
        for (lower, upper, output) in ((p_lo, p_hi, prior), (l_lo, l_hi, likelihood))
            d_lo = _reweighted_distance(x, uniform, lower.weights[order])
            d_hi = _reweighted_distance(x, uniform, upper.weights[order])
            output[j] = (d_lo / -log2(lo) + d_hi / log2(hi)) / 2
        end
    end
    diagnosis = map(prior, likelihood) do p, l
        p >= cutoff && l >= cutoff ? :prior_data_conflict :
        p > cutoff && l < cutoff ? :strong_prior_weak_likelihood : :none
    end
    k = (; prior_lower=p_lo.pareto_k, prior_upper=p_hi.pareto_k,
           likelihood_lower=l_lo.pareto_k, likelihood_upper=l_hi.pareto_k)
    good(a, b) = isfinite(a) && isfinite(b) && max(a, b) <= 0.7
    reliable = (; prior=good(k.prior_lower, k.prior_upper),
                  likelihood=good(k.likelihood_lower, k.likelihood_upper))
    return PowerscaleSensitivity(selected, prior, likelihood, diagnosis, k, reliable, cutoff, lo, hi)
end

powerscale_sensitivity(draws::AbstractMatrix, log_prior::AbstractVector,
                       log_lik::AbstractVector, names; kwargs...) =
    powerscale_sensitivity(draws, log_prior, log_lik; names, kwargs...)
