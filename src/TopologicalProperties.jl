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
            topoShift = 2
        elseif RootPattern == [2,1,2,1]
            topoShift = 1

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


    path = "DMRG/Data/$(tag)Infinite/"
    name = "rp$(RootPattern_to_string(rp; first_term=tag))_Ly$(round(L, digits=5))_theta0.0_chi$(χ)_alpha0.0.jld2"

    #check is there is a file, if not run idmrg loop
    !isfile(path*"MES/"*name) && idmrgLoop(rp, L, tag, θ; kwargs...)

    #calculate berryPhase due to modular T transform
    println("\nCalculating Dehn twist for L = $(L) !")
    shift = topologicalShift(rp)
    
    dmrgStruct = load(path*"MES/"*name, "dmrgStruct")
    ψ = dmrgStruct.ψ    
    
    return DehnTwist(ψ, Ncell, shift, path*"BlockMPS/Ncell$(Ncell)_"*name, L; kwargs...)
end

function BerryPhase(rp::Vector{Int64}; setL::LinRange{Float64, Int64}, kwargs...)
    
    setBerryPhase = []
    for L in setL
      
        el, W = BerryPhase(rp, L; kwargs...)
        append!(setBerryPhase, el)
    end

    @show linear_fit(setL.*setL, setBerryPhase)

    return setBerryPhase
end

function BerryPhase(;setRP::Vector{Vector{Int64}}, kwargs...)
    
    DictB = Dict()
    for rp in setRP

        println("\n####################################\n          Root pattern : $(RootPattern_to_string(rp))   \n####################################\n")
        flush(stdout)
        #=
        name = "DMRG/Data/LaughlinInfinite/test/frp$(RootPattern_to_string(rp))_Flux$(round(kwargs[16], digits=5)).jld2"
        el = []
        if isfile(name)
            el = load(name, "berry")
        else
            
            save(name, "berry", el)
        end
        =#
        el = BerryPhase(rp; kwargs...)
        DictB[rp] = el

    end

    return DictB
end


function plotBerryPhase(setL::Vector{Float64}, DictDT::Dict)

    setFit = setfit(setL, DictDT)

    fig = plot()


end



function setfit(setL::Vector{Float64}, DictDT::Dict)
    setF = []
    for(k, v) in DictDT

        f = linear_fit(setL.*setL, v)
        # setF[k] = f
        append!(setF, f)
    end

    return setF
end