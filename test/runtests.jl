using Test, PowerScaling, PSIS, Distributions, Random, Statistics

include("distance.jl")
include("sensitivity.jl")

if !isempty(ARGS) && first(ARGS) == "references"
    include(joinpath(@__DIR__, "..", "validation", "references.jl"))
end
