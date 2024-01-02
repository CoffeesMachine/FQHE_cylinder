using Revise
using ITensors
using ITensorInfiniteMPS

using CurveFit

include("DehnTwist.jl")
include("iDMRGLoop.jl")



function topologicalShift(RootPattern::Vector{Int64})
    
    topoShift = 0

    if length(RootPattern) == 4
        if RootPattern == [2, 2, 1, 1]
            topoShift = 2
        elseif RootPattern == [2, 1, 2, 1]
            topoShift = 1
        end
    else
        if RootPattern == [2,1,1]
            topoShift = 1
        elseif RootPattern == [1,1,2]
            topoShift = -1
        end
    end

    return topoShift
end


function BerryPhase(rp::Vector{Int64}, L::Float64; χ::Int64, θ::Float64, tag::String, Ncell::Int64, Nτ::Int64, kwargs...)

    path = "/scratch/bmorier/$(tag)/"
    println(kwargs[end])
    flush(stdout)
    gapTag = kwargs[end] ? "gap_" : ""
    name = gapTag*"rp$(RootPattern_to_string(rp))_chiMax$(maximum(kwargs[1]))_Ly$(round(L, digits=5))_theta$(round(θ, digits=5))_maxiters100_chi$(χ)_alpha0.0.jld2"

    #check is there is a file, if not run idmrg loop
    !isfile(path*name) && idmrgLoop(rp, L, tag, θ; kwargs...)
   
    #calculate berryPhase due to modular T transform
    println("\nCalculating Dehn twist for L = $(L)")
    flush(stdout)
    shift = topologicalShift(rp)
    
    dmrgStruct = load(path*name, "dmrgStruct")
    ψ = dmrgStruct.ψ
    
    nameBlock = path*"BlockMPS/Ncell$(Ncell)_"*name
    return DehnTwist(ψ, Nτ, Ncell, shift, path*"BlockMPS/Ncell$(Ncell)_"*name; kwargs...)
end


function BerryPhase(rp::Vector{Int64}; skipDT, setL, kwargs...)
    setBerry = []
    for L in setL
        if skipDT 
            idmrgLoop(rp, L; kwargs...)
        else
        el = BerryPhase(rp, L; kwargs...)
        append!(setBerry, el)
        end
    end
    
    return setBerry
end


function BerryPhase(;rp::Vector{Int64}, kwargs...)
    
    DictB = Dict()

    DictB[rp] = BerryPhase(rp; kwargs...)
    return DictB
end