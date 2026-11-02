@testset "Cumulative Jensen–Shannon distance" begin
    x = [0.0, 1.0]
    expected = sqrt(0.2 * log2(0.4) + 0.8 * log2(1.6))
    @test cjs_dist(x, x; x_weights=[0.2, 0.8], y_weights=[0.8, 0.2]) ≈ expected
    @test cjs_dist(x, x) == 0
    @test cjs_dist(fill(2.0, 10), fill(2.0, 10)) == 0
    # priorsense's unequal-support grid includes the final endpoint: nine
    # bins have CDFs (1,0), one has (1,1), hence normalized divergence 9/11.
    @test cjs_dist(fill(1.0, 10), fill(2.0, 10)) ≈ sqrt(9 / 11)
    @test cjs_dist(x, x; x_weights=[1.0, 0.0], y_weights=[0.0, 1.0]) ≈ 1

    rng = Xoshiro(481)
    a, b = randn(rng, 501), 0.3 .+ randn(rng, 703)
    wa, wb = rand(rng, 501), rand(rng, 703)
    inputs = deepcopy((a, b, wa, wb))
    d = cjs_dist(a, b; x_weights=wa, y_weights=wb)
    @test 0 < d < 1
    @test d ≈ cjs_dist(b, a; x_weights=wb, y_weights=wa)
    @test d ≈ cjs_dist(-a, -b; x_weights=wa, y_weights=wb)
    @test d ≈ cjs_dist(3a .+ 5, 3b .+ 5; x_weights=wa, y_weights=wb)
    @test d ≈ cjs_dist(a, b; x_weights=10wa, y_weights=0.1wb)
    @test (a, b, wa, wb) == inputs

    tied = [0.0, 0.0, 1.0, 2.0, 2.0]
    weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    permutation = [5, 2, 4, 1, 3]
    @test cjs_dist(tied, tied; y_weights=weights) ≈
          cjs_dist(tied[permutation], tied[permutation]; y_weights=weights[permutation])
    @test_throws ArgumentError cjs_dist([1.0], x)
    @test_throws ArgumentError cjs_dist([NaN, 1.0], x)
    @test_throws ArgumentError cjs_dist(x, x; y_weights=[-1, 2])
    @test_throws ArgumentError cjs_dist(x, x; y_weights=[0, 0])
    @test_throws ArgumentError cjs_dist(x, x; y_weights=[Inf, 1])
    @test_throws DimensionMismatch cjs_dist(x, x; y_weights=[1])
end

# Independent numerical integration of analytic (truncated) Normal CDFs.
# Same finite support as the empirical estimator, without reweighting draws.
function analytic_cjs(p, q, lo, hi; bins=40_000)
    z = range(lo, hi; length=bins + 1)
    mid = (z[1:end-1] .+ z[2:end]) ./ 2
    function truncated_cdf(d)
        (cdf.(d, mid) .- cdf(d, lo)) ./ (cdf(d, hi) - cdf(d, lo))
    end
    f, g = truncated_cdf(p), truncated_cdf(q)
    function divergence(a, b)
        total = sum(eachindex(a)) do i
            s = a[i] + b[i]
            s == 0 && return 0.0
            (a[i] == 0 ? 0.0 : a[i] * log2(2a[i] / s)) +
            (b[i] == 0 ? 0.0 : b[i] * log2(2b[i] / s))
        end
        sqrt(max(total, 0) / sum(a .+ b))
    end
    max(divergence(f, g), divergence(1 .- f, 1 .- g))
end
