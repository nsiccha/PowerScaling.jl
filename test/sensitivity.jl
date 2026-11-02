@testset "Weights use PSIS and preserve inputs" begin
    rng = Xoshiro(618)
    component = -abs2.(randn(rng, 2_000)) ./ 2
    original = copy(component)
    for alpha in (0.8, 1.01, 1.2), reff in (0.5, 1.0)
        result = powerscale_weights(component; alpha, reff)
        reference = PSIS.psis((alpha - 1) .* component, reff)
        @test result.weights ≈ reference.weights
        @test result.log_weights ≈ reference.log_weights
        @test result.pareto_k ≈ reference.pareto_shape
        @test result.tail_length == reference.tail_length
        @test result.ess ≈ reff / sum(abs2, result.weights)
        @test sum(result.weights) ≈ 1
        @test exp.(result.log_weights) ≈ result.weights
        @test powerscale_weights(component .+ 100; alpha, reff).weights ≈ result.weights
    end
    @test component == original
    base = powerscale_weights(component; alpha=1, reff=0.5)
    @test base.pareto_k == -Inf
    @test base.tail_length == 0
    @test base.ess == 1_000
    @test base.weights == fill(1 / 2_000, 2_000)
    for alpha in (-1, Inf, NaN)
        @test_throws ArgumentError powerscale_weights(component; alpha)
    end
    for reff in (0, -1, Inf, NaN)
        @test_throws ArgumentError powerscale_weights(component; alpha=1, reff)
    end
    @test_throws ArgumentError powerscale_weights(Float64[]; alpha=1)
    @test_throws ArgumentError powerscale_weights(fill(1.0, 100); alpha=1.1)
    @test_throws ArgumentError powerscale_weights([NaN, 0.0]; alpha=1)
    @test_throws ArgumentError powerscale_weights(component[1:20]; alpha=1.1)
    @test_throws ArgumentError powerscale_weights(component; alpha=1.1, reff=1e8)

    # A severe density tilt must remain visibly unreliable, not disappear.
    x = ((1:1_000) .- 0.5) ./ 1_000
    bad = powerscale_sensitivity(reshape(x, :, 1), -200log.(1 .- x), -x.^2 ./ 2)
    @test !bad.reliable.prior
    @test any(k -> !isfinite(k) || k > 0.7,
              (bad.pareto_k.prior_lower, bad.pareto_k.prior_upper))
end

@testset "Conjugate Normal posterior and analytic CJS gradient" begin
    # Prior N(0,1), observation 2 with unit standard error: posterior N(1,1/√2).
    # Deterministic posterior quantiles avoid Monte Carlo noise in a small gradient.
    n = 30_000
    posterior = Normal(1, inv(sqrt(2)))
    x = quantile.(posterior, ((1:n) .- 0.5) ./ n)
    lp, ll = -abs2.(x) ./ 2, -abs2.(x .- 2) ./ 2
    draws = hcat(x, exp.(x), fill(3.0, n))
    before = deepcopy((draws, lp, ll))
    s = powerscale_sensitivity(draws, lp, ll; names=[:mu, :exp_mu, :constant])
    @test s.variables == [:mu, :exp_mu, :constant]
    @test s.prior[3] == s.likelihood[3] == 0
    @test s.diagnosis[1] == :prior_data_conflict
    @test s.reliable == (prior=true, likelihood=true)
    @test (draws, lp, ll) == before
    for (component, output) in ((:prior, s.prior), (:likelihood, s.likelihood))
        distances = map((s.lower_alpha, s.upper_alpha)) do a
            target = Normal(component == :prior ? 2 / (1 + a) : 2a / (1 + a),
                            inv(sqrt(1 + a)))
            w = powerscale_weights(component == :prior ? lp : ll; alpha=a).weights
            @test sum(w .* x) ≈ mean(target) atol=5e-5
            @test sum(w .* (x .- mean(target)).^2) ≈ var(target) atol=2e-4
            analytic_cjs(posterior, target, first(x), last(x))
        end
        expected = (distances[1] / -log2(s.lower_alpha) + distances[2] / log2(s.upper_alpha)) / 2
        @test output[1] ≈ expected rtol=0.003
    end
    selected = powerscale_sensitivity(draws, lp, ll, ["mu", "exp_mu", "constant"]; variables=[:exp_mu, :mu])
    @test selected.prior ≈ s.prior[[2, 1]]
    @test selected.likelihood ≈ s.likelihood[[2, 1]]
    @test powerscale_sensitivity(draws, lp, ll; threshold=1e6).diagnosis == fill(:none, 3)
    @test occursin("Pareto-k", sprint(show, MIME"text/plain"(), s))
    @test powerscale_sensitivity(reshape(x, :, 1), lp, ll).variables == [:x1]

    # A narrow prior with a weak likelihood recovers the other diagnosis.
    y = quantile.(Normal(0, sqrt(1 / 1.01)), ((1:4_000) .- 0.5) ./ 4_000)
    weak = powerscale_sensitivity(reshape(y, :, 1), -y.^2 ./ 2, -y.^2 ./ 200)
    @test weak.diagnosis == [:strong_prior_weak_likelihood]

    @test_throws DimensionMismatch powerscale_sensitivity(draws, lp[2:end], ll)
    @test_throws DimensionMismatch powerscale_sensitivity(draws, lp, ll, [:one])
    @test_throws ArgumentError powerscale_sensitivity(draws, lp, ll; variables=[:missing])
    @test_throws ArgumentError powerscale_sensitivity(draws, lp, ll; variables=Symbol[])
    @test_throws ArgumentError powerscale_sensitivity(draws, lp, ll; variables=[:x1, :x1])
    @test_throws ArgumentError powerscale_sensitivity(draws, lp, ll; names=[:a, :a, :b])
    @test_throws ArgumentError powerscale_sensitivity(draws, lp, ll; upper_alpha=Inf)
    @test_throws ArgumentError powerscale_sensitivity(draws, lp, ll; lower_alpha=1)
    @test_throws ArgumentError powerscale_sensitivity(draws, lp, ll; threshold=NaN)
    @test_throws ArgumentError powerscale_sensitivity(draws, lp, ll; threshold=-1)
end
