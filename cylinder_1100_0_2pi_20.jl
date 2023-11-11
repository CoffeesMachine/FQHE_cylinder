using MKL
using ITensors
using ITensorInfiniteMPS

using LinearAlgebra
using Statistics
include("InfiniteCylinder.jl");

ITensors.enable_threaded_blocksparse()
BLAS.set_num_threads(1)
ITensors.Strided.disable_threads()

Ly = parse(Float64, ARGS[1])

function run_iDMRG(Ly, RootPattern, θ, path, SetAlpha, Setχ)

    maxIter = 50
    name = "Ly$(Ly)_theta$(θ)_maxiters$(maxIter)"
    
    
    save_every = 2

    dmrgStruct = FQHE_idmrg(RootPattern, Ly, [1.], [0., 0., 1.], θ)

    for χ in Setχ
		
        println("Starting χ = $χ")
		flush(stdout)
        
        all_ener = Float64[];
        all_err = Float64[];
        all_entr = Float64[];


        for alpha in SetAlpha
            
            println("Starting alpha = $alpha")
            flush(stdout)

            xs = alpha == 0 ? (1:maxIter÷save_every) : (1:2)
			
			for x in xs
                @time ener, err, entr = idmrg(dmrgStruct; mixer = true, alpha = alpha, maxdim = χ, nb_iterations = save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-8, output_level = 1)
                append!(all_ener, ener); append!(all_err, err); append!(all_entr, entr)
                
                if alpha != 0
                    @time ener, err, entr = idmrg(dmrgStruct; mixer = true, alpha = 0, maxdim = χ, nb_iterations = save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-8, output_level = 1)
                end

                save(path*name*"_chi$(χ)_noise$(alpha).jld2", "dmrgStruct", dmrgStruct, "energies", all_ener, "errors", all_err, "entropies", all_entr)
                length(all_ener)<10 && continue
                save_every*x < 6 && continue
                if std(all_ener[end-5:end]) < 1e-8 && std(all_entr[end-5:end])< 1e-8
                    println("Finishing early")
                    flush(stdout)
                    break
                end
            end
        end
    end
end


function run(Ly)
    println("Starting with Ly=$(Ly)")

    RootPattern = [2,2,1,1]
    tag = RootPattern ==[2,2,1,1] ? "1100" : "1010"

    setAlpha = [1e-8, 0.]
    setChi = [2^n for n=8:10]

    ThetaMin=0
    ThetaMax=2*pi
    
    N_Theta = 20

    setTheta = LinRange(ThetaMin, ThetaMax, N_Theta)

    path = "/scratch/bmorier/2b_3b/RootPattern$(tag)_chiMax$(maximum(setChi))_"
    
    for theta in setTheta
        println("\n########################### \n# Calculating for θ = $theta #\n###########################\n")

        if theta == setTheta[1] ||theta == setTheta[2]
            continue
        end

        run_iDMRG(Ly, RootPattern, theta, path, setAlpha, setChi)
    end
end

run(Ly)