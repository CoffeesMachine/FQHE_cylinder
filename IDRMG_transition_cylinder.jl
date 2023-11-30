#using MKL
using Revise
using ITensors
using ITensorInfiniteMPS

using Statistics
using Plots
using KrylovKit
include("InfiniteCylinder.jl");


#=
BLAS.set_num_threads(1)
ITensors.Strided.disable_threads()
ITensors.enable_threaded_blocksparse()
=#

using MKL
using ITensors
using ITensorInfiniteMPS

using LinearAlgebra
using Statistics
include("InfiniteCylinder.jl");

ITensors.enable_threaded_blocksparse()
BLAS.set_num_threads(1)
ITensors.Strided.disable_threads()

function run_iDMRG(Ly, RootPattern, θ, path, SetAlpha, Setχ)

    maxIter = 10
    name =  "Ly$(Ly)_theta$(round(θ, digits=5))_"
    
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

                save(path*name*"chi$(χ)_noise$(alpha)_t.jld2", "dmrgStruct", dmrgStruct, "energies", all_ener, "errors", all_err, "entropies", all_entr)
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



function fidelity(Ly::Float64, θ1::Float64, θ2::Float64, tag::String, alpha::Float64, maxiters::Int64, χ::Int64)

    path= "DMRG/test/"
    name1 = "Ly$(Ly)_theta$(round(θ1, digits=5))_chi$(χ)_noise$(alpha)_t.jld2"
    name2 = "Ly$(Ly)_theta$(round(θ2, digits=5))_chi$(χ)_noise$(alpha)_t.jld2"

    dmrgStruct_1 = load(path*name1, "dmrgStruct")
    dmrgStruct_2 = load(path*name2, "dmrgStruct")
    ψ1 = dmrgStruct_1.ψ
    ψ2 = dmrgStruct_2.ψ


    replace_siteinds!(ψ2, siteinds(ψ1))
    N_eigen=1
    T = TransferMatrix(ψ1.AL, ψ2.AL)
    
        
    vtest = randomITensor(QN(("Nf", 0), ("NfMom", 0)), dag(input_inds(T)))
    if norm(vtest) < 1e-10
        println("No 0 sector")
        return 0.
    end

    vⁱᴿ = translatecell(translator(ψ1),vtest , -1)

    
    λᴿ, _, _ = eigsolve(T, vⁱᴿ, N_eigen, :LM; tol=1e-8)
    
    λmax = λᴿ[1]
    @show λmax

    return abs(λmax)
end


function runI(Ly, RootPattern, alpha, maxiters, χ, θmin, θmax, N_θ)

    setθ = LinRange(θmin, θmax, N_θ+1)
    
    Vs_2b = [1.]
    Vs_3b = [0., 0., 1.]
    
    set_Ener = []
    set_Err = []
    set_Entr = []

    tag = RootPattern == [2,2,1,1] ? "11" : "10"
    
    #=
    for θ in setθ

        println("Calculating for θ=$(θ)")

        run_iDMRG(Ly, RootPattern, θ, "DMRG/test/", [1.e-8, .0], [200])
    end
    =#
    
    #plotEnergy(setθ, set_Ener, Ly, set_Err)
    setFi = []
    setθFi = []
    for ind in 1:N_θ
        append!(setFi, fidelity(Ly,setθ[ind], setθ[ind+1], tag, alpha, maxiters, χ))
        push!(setθFi, setθ[ind])
    end

    plotFidelity(setθFi, setFi, Ly)



end

####################
####################

function plotFidelity(Setθ, Fi, Ly)
    fig = plot()
    scatter!(fig, Setθ, Fi)
    title!("Fidelity for Ly=$(Ly)")
    vline!([n*pi/4 for n=3:4:8])
    display(fig)
end


function plotEnergy(Setθ, E, Ly, Err)
    fig = plot()
    plot!(fig, Setθ, E)
    title!("Energy for Ly=$(Ly)")
    vline!([n*pi/4 for n=3:4:8])
    display(fig)
end


runI(10., [2,2,1,1], 1e-8, 10, 200, 0., 2*pi, 20)