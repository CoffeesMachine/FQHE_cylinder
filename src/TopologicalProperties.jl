using Revise
using ITensors
using ITensorInfiniteMPS

using CurveFit
using KrylovKit

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


function BerryPhase(rp::Vector{Int64}, L::Float64; χ::Int64, θ::Float64, tag::String, Ncell::Int64, Nτ::Int64, cut::Int64, kwargs...)


    path = "Data/$(tag)Infinite/"
    name = "rp$(RootPattern_to_string(rp; first_term=tag))_Ly$(round(L, digits=5))_theta0.0_chi$(χ)_alpha0.0.jld2"

    #check is there is a file, if not run idmrg loop
    !isfile(path*"MES/"*name) && idmrgLoop(rp, L, tag, θ; kwargs...)

    #calculate berryPhase due to modular T transform
    println("\nCalculating Dehn twist for L = $(L) !")
    shift = topologicalShift(rp)
    
    dmrgStruct = load(path*"MES/"*name, "dmrgStruct")
    ψ = dmrgStruct.ψ    

    return DehnTwist(ψ, Ncell, shift, path*"BlockMPS/Ncell$(Ncell)_cut$(cut)_"*name, L, cut; kwargs...)
end

function BerryPhase(rp::Vector{Int64}; setL::LinRange{Float64, Int64}, kwargs...)
    
    setBerryPhase = []
    for L in setL
      
        el= BerryPhase(rp, L; kwargs...)
        append!(setBerryPhase, el)
    end

    return setBerryPhase
end

function BerryPhase(;setRP::Vector{Vector{Int64}}, kwargs...)
    
    DictB = Dict()
    for rp in setRP

        println("\n####################################\n          Root pattern : $(RootPattern_to_string(rp))   \n####################################\n")
        flush(stdout)

        el = BerryPhase(rp; kwargs...)
        DictB[rp] = el

    end

    return DictB
end



function topologicalProperties(setL, BerryPhaseD, phiX)
    println("")
    setX = setL.*setL
    dictFit = Dict()
    flag = false
    for (k,v) in BerryPhaseD
        fit = linear_fit(setX, v)
        if length(k) == 3
            flag = true
        end
        dictFit[k] = fit 
        eta = 2*π^2*4*fit[2]

        modular = fit[1]
        #assuming central charge 1/24
        hₐ = modular


        println("hₐ for Φx = $phiX and root pattern $(RootPattern_to_string(k)) : $(hₐ)")
        println("ηₕ for Φx = $phiX and root pattern $(RootPattern_to_string(k)) : $eta\n")
    end
    
    if flag
        println("h₀- h₁ = $((dictFit[[2,1,1]][1]-dictFit[[1,2,1]][1]))")
        println("h₀- h₂ = $((dictFit[[2,1,1]][1]-dictFit[[1,1,2]][1]))")
        println("h₁- h₂ = $((dictFit[[1,2,1]][1]-dictFit[[1,1,2]][1]))")
    else
        println("h₀- h₁ = $((dictFit[[2,2,1,1]][1]-dictFit[[2,1,2,1]][1]))")
    end
    return dictFit
end
    