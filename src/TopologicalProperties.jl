using Revise
using ITensors
using ITensorInfiniteMPS

using CurveFit

include("DehnTwist.jl")
include("MinimalEntangledStates.jl")



function topologicalShift(RootPattern::Vector{Int64})
    
    topoShift = 0

    if length(RootPattern) == 4
        if RootPattern == [2, 2, 1, 1]
            topoShift = 1/4
        elseif RootPattern == [2,1,2,1]
            topoShift = -1/2

        end
    else
        if RootPattern==[2,1,1]
            topoShift = 1
        elseif RootPattern == [1,1,2]
            topoShift = -1
        end
    end

    return topoShift
end


function BerryPhase(rp::Vector{Int64}, L::Float64; χ::Int64, θ::Float64, tag::String, Ncell::Int64, Nτ::Int64, kwargs...)

    #=
    if tag == ""
        tag = length(rp)==4 ? "Pfaffian" : "Laughlin"
    end
    =#


    path = "DMRG/Data/$(tag)Infinite/MES/"
    name = "rp$(RootPattern_to_string(rp; first_term=tag))_Ly$(round(L, digits=5))_theta0.0_chi$(χ)_alpha0.0.jld2"

    #check is there is a file, if not run idmrg loop
    !isfile(path*name) && idmrgLoop(rp, L, tag, θ; kwargs...)

    #calculate berryPhase due to modular T transform
    shift = topologicalShift(rp)
    dmrgStruct = load(path*name, "dmrgStruct")
    ψ = dmrgStruct.ψ    

    return DehnTwist(ψ, Nτ, Ncell, shift; kwargs...)
end

function BerryPhase(rp::Vector{Int64}; setL::LinRange{Float64, Int64}, kwargs...)
    
    setBerryPhase = []

    for L in setL 
        append!(setBerryPhase, BerryPhase(rp, L; kwargs...))
    end

    return setBerryPhase
end

function BerryPhase(;setRP::Vector{Vector{Int64}}, kwargs...)
    
    DictB = Dict()
    for rp in setRP

        println("\n####################################\n          Root pattern : $(RootPattern_to_string(RootPattern))   \n####################################\n")
        flush(stdout)
        
        DictB[rp] = BerryPhase(rp; kwargs...)
    end

    return DictB
end

