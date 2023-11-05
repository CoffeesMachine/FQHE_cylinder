using Revise
using ITensors
using ITensorInfiniteMPS

using Statistics
using plots
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

    path = "Data\\TransitionInfiniteCylinder\\State\\"
    name1 = "$(tag)_idmrg_Ly$(Ly)_theta$(θ1)_maxiters$(maxiters)_chi$(χ)_noise$(alpha).jld2"
    name2 = "$(tag)_idmrg_Ly$(Ly)_theta$(θ2)_maxiters$(maxiters)_chi$(χ)_noise$(alpha).jld2"

    dmrgStruct_1 = load(path*name1, "dmrgStruct")
    dmrgStruct_2 = load(path*name2, "dmrgStruct")

    T = TransferMatrix(dmrgStruct_1.ψ.AL, dmrgStruct_2.ψ.AL, Cell(4))

    vⁱᴿ = randomITensor(dag(input_inds(T)))
    
    λᴿ, vᴿ,_ = eigsolve(T, vⁱᴿ, 5, :LM; tol=1e-9)
    
    return λᴿ, vᴿ
end



function runI(Ly, RootPattern, alpha, maxiters, χ, θmin, θmax, N_θ)

    setθ = LinRange(θmin, θmax, N_θ+1)
    
    Vs_2b = [1.]
    Vs_3b = [0., 0., 1.]
    
    set_Ener = []
    set_Entr = []


    for θ in setθ

        println("Calculating for θ=$(θ)")

        abs(cos(θ)) <= 0.1 && continue
        dmrgStruct = FQHE_idmrg(RootPattern, Ly, Vs_2b, Vs_3b, θ)

        tag = RootPattern == [2,2,1,1] ? "11" : "10"

        E, _, Entr = run_iDMRG(Ly, θ, tag, dmrgStruct, alpha, maxiters, χ)

        push!(set_Ener, E[end])
        push!(set_Entr, Entr[end])
    end





end

function plotEnerg(Ly, RootPattern, alpha, maxiters, χ, θmin, θmax, N_θ)
    setE = []
    sett = []
    for t in LinRange(θmin, θmax, N_θ+1)
        abs(cos(t)) <= 0.1 && continue
        dmrgStruct = FQHE_idmrg(RootPattern, Ly, [1.], [0., 0., 1.], t)
        _, _, E = run_iDMRG(Ly, t, "11", dmrgStruct, alpha, maxiters, 350)

        push!(setE, E[end])
        push!(sett, t)
    end

    plot()
    scatter!(sett, setE)


end


runI(8., [2,2,1,1], 1e-6, 10, 300, 0., 2*pi, 20)