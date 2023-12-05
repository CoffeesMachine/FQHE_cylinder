using Revise
using ITensors
using ITensorInfiniteMPS

using Statistics
include("InfiniteCylinder.jl")


function GenerateBasisLaugh(Ly::Float64, setχ::Vector{Int64}, RP::Vector{Int64}, path::String)
    
    maxIter = 40
    save_every =2  
    
    ener_tol = 1e-8
    ent_tol = 1e-8

    maxχ = maximum(setχ)
    alphas = [1e-8, 0.0]
    
    dmrgStruct = Laughlin_struct(Ly, [1.], RP)
    
    for χ in setχ
        println("Starting chi = $χ")
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
                
                save(path*"Ly$(round(Ly, digits=5))_chi$(χ)_alpha$(alpha)_RP$(RootPattern_to_string(RP)).jld2", "dmrgStruct", copy(dmrgStruct), "energies", all_ener, "errors", all_err, "entropies", all_entr)
                
                length(all_ener)<10 && continue
                save_every*x < 6 && continue

                if std(all_ener[end-5:end]) < ener_tol && std(all_entr[end-5:end])< ent_tol
                    println("Finishing early")
                    break
                end
            end
        end
    end
end

function GenerateBasisLaugh(Ly::Float64, setχ::Vector{Int64}, RP::Vector{Int64}, path::String, LeftEnv, RightEnv)
    
    maxIter = 40
    save_every =2  
    
    ener_tol = 1e-8
    ent_tol = 1e-8

    maxχ = maximum(setχ)
    alphas = [1e-8, 0.0]
    
    dmrgStruct = Laughlin_struct(Ly, [1.], RP, LeftEnv, RightEnv)
    
    for χ in setχ
        println("Starting chi = $χ")
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
                
                save(path*"Ly$(round(Ly, digits=5))_chi$(χ)_alpha$(alpha)_RP$(RootPattern_to_string(RP)).jld2", "dmrgStruct", copy(dmrgStruct), "energies", all_ener, "errors", all_err, "entropies", all_entr)
                
                length(all_ener)<10 && continue
                save_every*x < 6 && continue

                if std(all_ener[end-5:end]) < ener_tol && std(all_entr[end-5:end])< ent_tol
                    println("Finishing early")
                    break
                end
            end
        end
    end
end

function GenerateBasisLaugh(Lmax::Float64, χmax::Int64)
    Lmin = 10.
    setRP = [[1,1,2], [1,2,1], [2,1,1]]
    setL = LinRange(Lmin, Lmax, 5)

    setχ = collect(100:100:χmax)

    path = "DMRG/Data/LaughlinInfinite/Basis/"

    for rp in setRP 
        println("####################################\n      Root pattern : $(rp) \n####################################\n")
        
        for L in setL
            println("Calculating for L = $(L)")
            
            GenerateBasisLaugh(L, setχ, rp, path)
        end
    end
    println("Finish generating Laughlin states")
end

function LoadMES(filename::String, LeftEnv::Bool; range=10)
    
    @assert isfile(filename)
    
    dmrgstruct = load(filename, "dmrgStruct")

    ψ = dmrgstruct.ψ

    vectorMES = Array{ITensor}(undef, nsites(ψ)*range)
    if LeftEnv
        newψ = ψ.AL

        for i in 1:nsites(ψ)*range
            vectorMES[i] = newψ[i]
        end

    else
        newψ = ψ.AR

        for i in 1:nsites(ψ)*range
            vectorMES[i] = newψ[i]
        end

    end
end





function MES(L::Float64, χmax::Int64, a::Int64)
    
    pathLaugh = "DMRG/Data/LaughlinInfinite/Basis/"
    
    RP = []
    
    if a == 0 
        RP = [1,2,1]
    elseif a == 1
        RP = [2,1,1]
    else
        RP = [1,1,2]
    end
    
    filename = "Ly$(round(L, digits=5))_chi$(χmax)_alpha0.0_"
    
    topologicalSector = "RP$(RootPattern_to_string(RP)).jld2"
    identitySector = "RP010.jld20"
    isfile(pathLaugh*filename) || GenerateBasisLaugh(L, χmax)

    #choose the identity "state" to be RP = [0, 1, 0]
    LeftEnv = LoadMES(pathLaugh*filename*identitySector, true)
    RightEnv = LoadMES(pathLaugh*filename*topologicalSector, false)

    pathMES = "DMRG/Data/LaughlinInfinite/MES/"
    GenerateBasisLaugh(Ly, 100:100:χmax, RP, pathMES, LeftEnv, RightEnv)
    println("Done")
end


MES(15., 500, 0)