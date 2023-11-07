using Revise
using ITensors
using ITensorInfiniteMPS

using Statistics
using Plots
using KrylovKit
include("InfiniteCylinder.jl");





function run_iDMRG(Ly, θ, tag, dmrgStruct, alpha, maxiters, χ)

    path = "DMRG\\Data\\TransitionInfiniteCylinder\\State\\"
    name = "$(tag)_idmrg_Ly$(Ly)_theta$(θ)_maxiters$(maxiters)_chi$(χ)_noise$(alpha).jld2"

    save_every = 2
    all_ener = Float64[]
    all_err = Float64[]
    all_entr = Float64[]

    if isfile(path*name)
        all_ener, all_err, all_entr = load(path*name, "energies", "errors", "entropies")
    else
        for x in 1:maxiters÷save_every
            ener, err, entr = idmrg(dmrgStruct; mixer = true, alpha = alpha, maxdim = χ, nb_iterations = 2*save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-8, output_level = 1)
            ener, err, entr = idmrg(dmrgStruct; mixer = true, alpha = 0, maxdim = χ, nb_iterations = 4*save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-8, output_level = 1)
            
            append!(all_ener, ener); append!(all_err, err); append!(all_entr, entr)
            save(path*name, "dmrgStruct", dmrgStruct, "energies", all_ener, "errors", all_err, "entropies", all_entr)
            length(all_ener)<10 && continue
            save_every*x < 6 && continue
            if std(all_ener[end-5:end]) < 1e-8 && std(all_entr[end-5:end])< 1e-8
                println("Finishing early")
                break
            end
        end
    end

    return all_ener, all_err, all_entr
end


function fidelity(Ly::Float64, θ1::Float64, θ2::Float64, tag::String, alpha::Float64, maxiters::Int64, χ::Int64)

    path= "/DMRG/Data/TransitionInfiniteCylinder/State/"
    name1 = "$(tag)_idmrg_Ly$(Ly)_theta$(θ1)_maxiters$(maxiters)_chi$(χ)_noise$(alpha).jld2"
    name2 = "$(tag)_idmrg_Ly$(Ly)_theta$(θ2)_maxiters$(maxiters)_chi$(χ)_noise$(alpha).jld2"

    dmrgStruct_1 = load(path*name1, "dmrgStruct")
    dmrgStruct_2 = load(path*name2, "dmrgStruct")
    ψ1 = dmrgStruct_1.ψ
    ψ2 = dmrgStruct_2.ψ


    replace_siteinds!(ψ2, siteinds(ψ1))
    N_eigen=4
    T = TransferMatrix(ψ1.AL, ψ2.AL)

    vtest = randomITensor(dag(input_inds(T)))
    @show norm(vtest)
    vⁱᴿ = translatecell(translator(ψ1),vtest , -1)
 
    
    λᴿ, _, _ = eigsolve(T, vⁱᴿ, N_eigen, :LM; tol=1e-8)
    
    λmax = λᴿ[1]
    @show λmax

    return -log(abs(λmax))
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

        abs(cos(θ)) <= 0.1 && continue
        dmrgStruct = FQHE_idmrg(RootPattern, Ly, Vs_2b, Vs_3b, θ)

        E, Err, Entr = run_iDMRG(Ly, θ, tag, dmrgStruct, alpha, maxiters, χ)

        push!(set_Ener, mean(E))
        push!(set_Err, std(E))
        push!(set_Entr, Entr[end])
    end
    =#
    
    #plotEnergy(setθ, set_Ener, Ly, set_Err)
    setFi = []
    setθFi = []
    for ind in 1:N_θ
        @show ind
        if abs(cos(setθ[ind])) <= 0.1 || abs(cos(setθ[ind+1])) <= 0.1
            continue
        end
        append!(setFi, fidelity(Ly,setθ[ind], setθ[ind+1], tag, alpha, maxiters, χ))
        push!(setθFi, setθ[ind])
    end

    plotFidelity(setθFi, setFi, Ly)
    plotEnergy(setθ, set_Ener, Ly, set_Err)




end

####################
####################

function plotFidelity(Setθ, Fi, Ly)
    fig = plot()
    scatter!(fig, Setθ, Fi)
    title!("Fidelity for Ly=$(Ly)")
    vline!([n*pi/4 for n=1:1:8])
    display(fig)
    
end


function plotEnergy(setθ, E, Ly, Err)
    newSetθ = []
    for θ in setθ
        abs(cos(θ)) <= 0.1 && continue
        push!(newSetθ, θ)
    end

    fig = plot()
    plot!(fig, newSetθ, E, yerr=Err)
    title!("Energy for Ly=$(Ly)")
    vline!([n*pi/4 for n=3:4:8])
    display(fig)
end


runI(8., [2,2,1,1], 1e-6, 5, 200, 0., 2*pi, 10)