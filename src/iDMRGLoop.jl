using Revise
using ITensors
using ITensorInfiniteMPS

using Statistics
include("InfiniteCylinder.jl")

idmrgLoop(; rp::Vector{Int64}, setL::LinRange{Float64, Int64}, tag::String, θ::Float64, kwargs...) = idmrgLoop(rp, setL[1], tag, θ; kwargs...)
    
idmrgLoop(;rp::Vector{Int64}, setL::Vector{Float64}, tag::String, θ::Float64, kwargs...) = idmrgLoop(rp, setL[1], tag, θ; kwargs...)

function idmrgLoop(RootPattern::Vector{Int64}, Ly::Float64, tag::String, θ::Float64; Haffnian::Bool, gap::Bool, χMaxReload::Int64, ReRunReload::Bool, reloadStruct::Bool, ChiTest::Int64, stepL::Float64, setχ::Vector{Int64}, V2b::Vector{Float64}, V3b::Vector{Float64}, prec::Float64, maxIter::Int64, NoiseReload::Float64, kwargs...)
    
    typeint = tag == "2b_3b" ? "Two-body three-body interaction" : "Three-body four-body interaction"
    fill = mod(length(RootPattern), 3) == 0 ? "1/3" : "1/2"
    println("Parameters are : \n####################################\n          Theta : $(θ)\n           Root pattern : $(RootPattern_to_string(RootPattern))\n          Ly : $(Ly) \n $(typeint)\n         filling : $(fill)   \n####################################\n")
    flush(stdout)
    path = "/scratch/bmorier/$(tag)/"
    gapTag  = gap ? "gap_" : ""
    hafTag = Haffnian ? "haf_" : ""
    nameN = path*hafTag*gapTag*"rp$(RootPattern_to_string(RootPattern))_chiMax$(maximum(setχ))_Ly$(round(Ly, digits=5))_theta$(round(θ, digits=5))"
    filename = hafTag*gapTag*"rp$(RootPattern_to_string(RootPattern))_chiMax$(χMaxReload)_Ly$(round(Ly-stepL, digits=5))_theta$(round(θ, digits=5))_maxiters$(maxIter)_chi$(ChiTest)_alpha0.0.jld2"
    
    if !reloadStruct
        savedpath =  path*filename
        
        if !isfile(savedpath)
            savedpath = "scratch/bmorier/saved/"*filename
        end

        type = tag == "3b_4b" ? "four" : "three"

        dmrgStruct = FQHE_idmrg(RootPattern, Ly, θ, type, savedpath; V2b=V2b, V3b=V3b, prec=prec, gap=gap, Haffnian=Haffnian)

        for χ in setχ
            idmrgLoop(dmrgStruct, nameN, χ, maxIter; kwargs...)
        end 
    else
        println("Reloading structure ")
        flush(stdout)
        filename = hafTag*gapTag*"rp$(RootPattern_to_string(RootPattern))_chiMax$(χMaxReload)_Ly$(round(Ly, digits=5))_theta$(round(θ, digits=5))_maxiters$(maxIter)_chi$(ChiTest)_alpha$(NoiseReload).jld2"
        if !isfile(path*filename)
            println("No file was found for χ=$(ChiTest), try a smaller one")
            ChiTest = div(ChiTest, 2)
            filename = hafTag*gapTag*"rp$(RootPattern_to_string(RootPattern))_chiMax$(χMaxReload)_Ly$(round(Ly, digits=5))_theta$(round(θ, digits=5))_maxiters$(maxIter)_chi$(ChiTest)_alpha$(NoiseReload).jld2"
        end
        dmrgStruct = load(path*filename, "dmrgStruct")
        
        for χ in setχ
            if ReRunReload
                χ < ChiTest && continue
                idmrgLoop(dmrgStruct, nameN, χ, maxIter; kwargs...)
            else
                χ <= ChiTest && continue
                idmrgLoop(dmrgStruct, nameN, χ, maxIter; kwargs...)
            end
        end 
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