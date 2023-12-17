using Revise
using ITensors
using ITensorInfiniteMPS

using Statistics
include("InfiniteCylinder.jl")

idmrgLoop(; rp::Vector{Int64}, setL::LinRange{Float64, Int64}, tag::String, θ::Float64, kwargs...) = idmrgLoop(rp, setL[1], tag, θ; kwargs...)
    


function idmrgLoop(RootPattern::Vector{Int64}, Ly::Float64, tag::String, θ::Float64; setχ::Vector{Int64}, V2b::Vector{Float64}, V3b::Vector{Float64}, prec::Float64, maxIter::Int64, kwargs...)
    
    println("Calulating for L=$(Ly)")

    for χ in setχ

        path = "/scratch/bmorier/$(tag)/rp$(RootPattern_to_string(RootPattern))_chiMax$(maximum(setχ))_"
        nameN = path*"Ly$(round(Ly, digits=5))_theta$(θ)"
        savedpath =  path*"/scratch/bmorier/saved/rp$(RootPattern_to_string(RootPattern))_chiMax$(maximum(setχ))_Ly$(round(Ly-1, digits=5))_theta$(round(θ, digits=5))_maxiters$(maxIter)_chi$(χ)_alpha0.0.jld2"

        dmrgStruct = FQHE_idmrg(RootPattern, Ly, θ, savedpath; V2b=V2b, V3b=V3b, prec=prec)
        
        idmrgLoop(dmrgStruct, nameN, χ, maxIter; kwargs...)
        
    end 
end


function idmrgLoop(dmrgStruct, path, χ, maxIter; ener_tol, ent_tol, alphas, save_every, kwargs...)
   
    println("Starting χ = $χ")
    all_ener = Float64[]
    all_err = Float64[]
    all_entr = Float64[]
    
    
    for alpha in alphas
        
        println("Starting alpha = $alpha")
        flush(stdout)

        xs = alpha == 0 ? (1:maxIter÷save_every) : (1:2)
        
        for x in xs
            @time ener, err, entr = idmrg(dmrgStruct; mixer = true, alpha = alpha, maxdim = χ, nb_iterations = save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-8, output_level = 1)
            append!(all_ener, ener); append!(all_err, err); append!(all_entr, entr)
            
            if alpha != 0
                @time ener, err, entr = idmrg(dmrgStruct; mixer = true, alpha = 0, maxdim = χ, nb_iterations = save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-8, output_level = 1)
            end
            
            append!(all_ener, ener); append!(all_err, err); append!(all_entr, entr)
            
            save(path*"_maxiters$(maxIter)_chi$(χ)_alpha$(alpha).jld2", "dmrgStruct", copy(dmrgStruct), "energies", all_ener, "errors", all_err, "entropies", all_entr)
            
            length(all_ener)<10 && continue
            save_every*x < 6 && continue

            if std(all_ener[end-5:end]) < ener_tol && std(all_entr[end-5:end])< ent_tol
                println("Finishing early")
                break
            end
        end
    end
end