using Revise
using ITensors
using ITensorInfiniteMPS

using Statistics
include("InfiniteCylinder.jl")


function idmrgLoop(RootPattern::Vector{Int64}, Ly::Float64, tag::String, θ::Float64; setχ::Vector{Int64}, V2b::Vector{Float64}, V3b::Vector{Float64}, prec::Float64, kwargs...)
    
    println("Calulating for L=$(Ly)")
    
    dmrgStruct = FQHE_idmrg(RootPattern, Ly, θ; V2b=V2b, V3b=V3b, prec=prec)

    path = "DMRG/Data/$(tag)Infinite/MES/rp$(RootPattern_to_string(RootPattern; first_term=tag))_Ly$(round(Ly, digits=5))_theta$θ"

    
    idmrgLoop(dmrgStruct, path, setχ; kwargs...)

end


function idmrgLoop(dmrgStruct, path, setχ; ener_tol, ent_tol, alphas, maxIter, save_every, kwargs...)
   
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
                
                save(path*"_chi$(χ)_alpha$(alpha).jld2", "dmrgStruct", copy(dmrgStruct), "energies", all_ener, "errors", all_err, "entropies", all_entr)
                
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


########################################################################################
# Below functions can be used to determine Minimal Entangled States but not compulsory #
########################################################################################


function MES(RootPattern::Vector{Int64}, L::Float64; χmax = 500, tag::String="", V2b::Vector{Float64}=[1.], V3b::Vector{Float64}=[0., 0., 1.], θ::Float64=0.0)

    kwargs = (
            V2b=V2b, 
            V3b=V3b, 
            prec=1e-10,
            ener_tol = 1e-8,
            ent_tol = 1e-8,
            alphas = [1e-8, 0.0],
            maxIter = 50,
            save_every = 2
    )

    setχ = 100:100:χmax

    idmrgLoop(RootPattern, L, setχ, tag, θ; kwargs...)
end

function MES(RootPattern, L, θmin, θmax, Nθ)

    setθ = LinRange(θmin, θmax, Nθ)

    for θ in setθ
        println("Calulating for θ = $(θ)")
        MES(RootPattern, L, θ=θ)
    end
    
end

function run_mes(RootPattern::Vector{Int64}, Lmin::Float64, Lmax::Float64, N::Int64; θmin = 0., θmax=0., Nθ=1)

    println("\n####################################\n          Root pattern : $(RootPattern_to_string(RootPattern))   \n####################################\n")

    setL = LinRange(Lmin, Lmax, N)
    for L in setL
        println("Calulating for L=$(L)")
        MES(rp, L, θmin, θmax, Nθ)
    end
end

function run_mes_laugh(Lmin::Float64, Lmax::Float64, N::Int64; θmin = 0., θmax=0., Nθ=1)

    setRP = [[2,1,1], [1,2,1], [1,1,2]]

    for rp in setRP
        run_mes(rp, Lmin, Lmax, N; θmin=θmin, θmax= θmax, Nθ=Nθ)
    end

    println("\n Finish job")
end


#run_mes(20., 20., 1)