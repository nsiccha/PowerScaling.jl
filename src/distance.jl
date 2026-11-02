# Weighted ECDF of ASCENDING `sx` with aligned weights at `v` (right
# continuous, matching priorsense `ewcdf`). `cw` is the cumulative weight
# vector, so each query is a binary search, not a scan.
_ewcdf_at(sx::Vector{Float64}, cw::Vector{Float64}, total::Float64, v::Float64) =
    (i = searchsortedlast(sx, v); i == 0 ? 0.0 : cw[i] / total)

# One-sided CJS(P || Q) + CJS(Q || P), bound-normalised and square-rooted —
# the priorsense `.cjs_dist` body. `wx`/`wy` are normalised inside, exactly
# as there. A zero-width support (every draw identical on both sides) is a
# point mass compared against itself: distance 0. R priorsense returns NaN
# there (0/0 over an empty bin set); 0.0 is the mathematically correct value.
function _cjs_onesided(sx::Vector{Float64}, wx::Vector{Float64},
                           sy::Vector{Float64}, wy::Vector{Float64})
    wx_sum, wy_sum = sum(wx), sum(wy)
    if sx == sy
        widths = diff(sx)
        all(iszero, widths) && return 0.0
        px = @view cumsum(wx ./ wx_sum)[1:(end - 1)]
        qx = @view cumsum(wy ./ wy_sum)[1:(end - 1)]
        bins = widths
    else
        nbins = max(length(sx), length(sy))
        lo = min(first(sx), first(sy))
        hi = max(last(sx), last(sy))
        step = (hi - lo) / (nbins - 1)
        cw_x = cumsum(wx)
        cw_y = cumsum(wy)
        px = [_ewcdf_at(sx, cw_x, wx_sum, lo + (b - 1) * step)
              for b in 1:nbins]
        qx = [_ewcdf_at(sy, cw_y, wy_sum, lo + (b - 1) * step)
              for b in 1:nbins]
        bins = fill(step, nbins)
    end
    px_int = sum(px .* bins)
    qx_int = sum(qx .* bins)
    cjs_pq = 0.5 / log(2) * (qx_int - px_int)
    cjs_qp = 0.5 / log(2) * (px_int - qx_int)
    for i in eachindex(px, qx, bins)
        pxi, qxi, w = px[i], qx[i], bins[i]
        # Terms with a zero outer mass are NaN (`0 * -Inf`) and dropped in R
        # by `na.rm = TRUE`; skip them rather than special-casing the log.
        pxi > 0 && (cjs_pq += w * pxi * (log2(pxi) - log2((pxi + qxi) / 2)))
        qxi > 0 && (cjs_qp += w * qxi * (log2(qxi) - log2((qxi + pxi) / 2)))
    end
    bound = px_int + qx_int
    # Zero bound happens only on the identical-support path, where it means
    # both weight vectors concentrate on the final draw: two point masses at
    # the same point, distance 0. (R priorsense NaNs here, like the constant
    # case above.)
    bound > 0 || return 0.0
    # The sum is mathematically nonnegative (it is a divergence), but
    # near-identical inputs can push it a few ulps below zero; R priorsense
    # NaNs there via sqrt. Clamp: a divergence of -1e-16 is 0.
    sqrt(max((cjs_pq + cjs_qp) / bound, 0.0))
end

function _cjs_weights(w, n::Int, what::String)
    isnothing(w) && return fill(1.0 / n, n)
    wv = Float64.(vec(collect(w)))
    length(wv) == n || throw(DimensionMismatch(
        "cjs_dist: $what has $(length(wv)) weights for $n draws"))
    all(isfinite, wv) || throw(ArgumentError("cjs_dist: $what must be finite"))
    all(>=(0), wv) || throw(ArgumentError("cjs_dist: $what must be nonnegative"))
    maximum(wv) > 0 || throw(ArgumentError("cjs_dist: $what is all zero"))
    # Normalize after scaling to avoid overflow for finite, large input weights.
    wv ./= maximum(wv)
    wv
end

"""
    cjs_dist(x, y; x_weights=nothing, y_weights=nothing)

Cumulative Jensen-Shannon distance between two samples, following R
priorsense `cjs_dist` (base-2 logs, upper-bound normalisation, square-root
metric, unsigned max over `x`/`-x`). Weights default to uniform. `x` and `y`
are usually the SAME posterior draws with uniform versus power-scaling
weights, in which case the weighted-ECDF fast path applies. Unequal supports
use the priorsense equally spaced grid with `max(length(x), length(y))` bins;
this is a finite-sample approximation to the continuous distance.

Inputs must be finite with at least two draws each; weights must be finite,
nonnegative, and not all zero. A point mass compared against itself is
distance 0.0 (R priorsense returns NaN there, an arithmetic artifact).
"""
function cjs_dist(x::AbstractVector, y::AbstractVector;
                      x_weights=nothing, y_weights=nothing)
    xv = Float64.(vec(collect(x)))
    yv = Float64.(vec(collect(y)))
    length(xv) >= 2 || throw(ArgumentError("cjs_dist needs at least two draws in `x`"))
    length(yv) >= 2 || throw(ArgumentError("cjs_dist needs at least two draws in `y`"))
    all(isfinite, xv) || throw(ArgumentError("cjs_dist needs finite draws in `x`"))
    all(isfinite, yv) || throw(ArgumentError("cjs_dist needs finite draws in `y`"))
    wx = _cjs_weights(x_weights, length(xv), "`x_weights`")
    wy = _cjs_weights(y_weights, length(yv), "`y_weights`")
    # The unsigned max needs its OWN sort: negating a sorted vector reverses
    # it, and the one-sided body requires ascending inputs (priorsense
    # re-sorts inside each call for the same reason).
    fwd = _cjs_sorted(xv, wx, yv, wy)
    bwd = _cjs_sorted(-xv, wx, -yv, wy)
    max(_cjs_onesided(fwd...), _cjs_onesided(bwd...))
end

function _cjs_sorted(xv::Vector{Float64}, wx::Vector{Float64},
                          yv::Vector{Float64}, wy::Vector{Float64})
    ox, oy = sortperm(xv), sortperm(yv)
    (xv[ox], wx[ox], yv[oy], wy[oy])
end
