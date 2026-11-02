# Executed in Pkg.test's environment with test_args=["references", BRM_PATH, OUTPUT_DIR].
# BRM's generic definitions are evaluated verbatim, without loading its model stack.
using LogExpFunctions

length(ARGS) == 3 || error("expected references BRM_CHECKOUT OUTPUT_DIR")
brm_path, output_dir = ARGS[2:3]
mkpath(output_dir)

module BRMReference
using Statistics, LogExpFunctions
end

let source = read(joinpath(brm_path, "src", "posterior_diagnostics.jl"), String)
    expressions = Meta.parseall(source).args
    helper = only(filter(expressions) do expr
        expr isa Expr && expr.head == :function &&
            expr.args[1] isa Expr && expr.args[1].head == :call &&
            first(expr.args[1].args) == :_brm_check_draw_names
    end)
    Core.eval(BRMReference, helper)
end
let source = read(joinpath(brm_path, "src", "powerscale.jl"), String)
    marker = findfirst("# ---- Part 4", source)
    isnothing(marker) && error("BRM source layout changed: inspect the extraction boundary")
    include_string(BRMReference, source[1:first(marker)-1], "BRM-generic-powerscale.jl")
end

@testset "BRM extraction parity" begin
    max_sensitivity_error = 0.0
    max_weight_error = 0.0
    max_k_error = 0.0
    rng = Xoshiro(20261102)
    for (mu0, tau, y, sigma) in ((0.0, 1.0, 2.0, 1.0), (0.0, 1.0, 0.0, 10.0),
                                (0.0, 10.0, 2.0, 0.5))
        variance = inv(inv(tau^2) + inv(sigma^2))
        center = variance * (mu0 / tau^2 + y / sigma^2)
        x = center .+ sqrt(variance) .* randn(rng, 4_000)
        lp, ll = logpdf.(Normal(mu0, tau), x), logpdf.(Normal(y, sigma), x)
        draws = hcat(x, exp.(x), x.^2)
        mine = powerscale_sensitivity(draws, lp, ll, [:mu, :exp_mu, :mu_squared])
        old = BRMReference.brm_powerscale_sensitivity(draws, lp, ll, [:mu, :exp_mu, :mu_squared])
        @test mine.prior ≈ old.prior atol=1e-8 rtol=1e-8
        @test mine.likelihood ≈ old.likelihood atol=1e-8 rtol=1e-8
        @test mine.diagnosis == old.diagnosis
        @test mine.reliable == old.reliable
        max_sensitivity_error = max(max_sensitivity_error,
            maximum(abs.(mine.prior .- old.prior)), maximum(abs.(mine.likelihood .- old.likelihood)))
        for component in (lp, ll), alpha in (0.99, 1.01)
            a = powerscale_weights(component; alpha)
            b = BRMReference.brm_powerscale_weights(component; alpha)
            @test a.weights ≈ b.weights atol=1e-12 rtol=1e-8
            @test a.pareto_k ≈ b.pareto_k atol=1e-8 rtol=1e-8
            max_weight_error = max(max_weight_error, maximum(abs.(a.weights .- b.weights)))
            max_k_error = max(max_k_error, abs(a.pareto_k - b.pareto_k))
        end
    end
    println("BRM_PARITY max_sensitivity_abs=", max_sensitivity_error,
            " max_weight_abs=", max_weight_error, " max_k_abs=", max_k_error)
end

# R receives the exact same draws, density values, and Julia-smoothed weights.
# This separates the CJS comparison from differences in R loo's smoothing.
n = 4_000
x = 1 .+ randn(Xoshiro(803), n) ./ sqrt(2)
lp, ll = -x.^2 ./ 2, -(x .- 2).^2 ./ 2
w = powerscale_weights(lp; alpha=1.01).weights
open(joinpath(output_dir, "draws.csv"), "w") do io
    println(io, "mu,lprior,log_lik,julia_weights")
    for i in eachindex(x)
        println(io, join((x[i], lp[i], ll[i], w[i]), ','))
    end
end
run(`Rscript $(joinpath(@__DIR__, "reference.R")) $output_dir`)
reference = parse.(Float64, split(strip(read(joinpath(output_dir, "r_values.txt"), String))))
summary = powerscale_sensitivity(reshape(x, :, 1), lp, ll)
distance = cjs_dist(x, x; y_weights=w)
@testset "R priorsense cross-check" begin
    @test distance ≈ reference[1] atol=1e-10 rtol=1e-8
    @test summary.prior[1] ≈ reference[2] atol=5e-4 rtol=0.005
    @test summary.likelihood[1] ≈ reference[3] atol=5e-4 rtol=0.005
end
println("R_PARITY CJS Julia=", distance, " R=", reference[1],
        " prior Julia=", summary.prior[1], " R=", reference[2],
        " likelihood Julia=", summary.likelihood[1], " R=", reference[3])
println("VERSIONS Julia=", VERSION, " PowerScaling=", pkgversion(PowerScaling),
        " PSIS=", pkgversion(PSIS), " BRM=", readchomp(`git -C $brm_path rev-parse HEAD`))
