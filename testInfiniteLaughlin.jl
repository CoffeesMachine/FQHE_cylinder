using Revise
using ITensors
using ITensorInfiniteMPS
using Plots

using LinearAlgebra
using LaTeXStrings
using Statistics
using CurveFit



include("InfiniteCylinder.jl");
include("DehnTwist.jl")
include("entanglement_spectrum.jl")

ITensors.set_warn_order(20)


function mainLaugh(Ly, chi)
   
    
	model_params = (Ly=Ly, Vs=[1.],)
    maxIter = 40
	save_every =2  
    
    ener_tol = 1e-8
    ent_tol = 1e-8

    setχ = 20:20:chi

    saving_prefix = "DMRG\\Data\\LaughlinInfinite\\r010_"
    dmrgStruc = Laughlin_struct(Ly, [1.], [1,2,1])
    for χ in setχ
        println("Starting chi = $χ")
        all_ener = Float64[]
        all_err = Float64[]
        all_entr = Float64[]
        #for alpha in alphas
        for x in 1:maxIter÷save_every
            ener, err, entr = idmrg(dmrgStruc; mixer = true, alpha = 1e-8, maxdim = χ, nb_iterations = save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-12, output_level = 1)
            ener, err, entr = idmrg(dmrgStruc; mixer = true, alpha = 0., maxdim = χ, nb_iterations = save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-12, output_level = 1)
            append!(all_ener, ener); append!(all_err, err); append!(all_entr, entr)
            save(saving_prefix*"Ly_$(round(Ly, digits=5))_chi-$(χ).jld2", "dmrgStruc", copy(dmrgStruc), "energies", all_ener, "errors", all_err, "entropies", all_entr, "params", model_params)
            length(all_ener)<10 && continue
            save_every*x < 6 && continue
            if std(all_ener[end-5:end]) < ener_tol && std(all_entr[end-5:end])< ent_tol
                println("Finishing early")
                break
            end
        end
    end
end;



function main(Lmax)
    Lmin = 15
    chi = 200
    setL = LinRange(Lmin, Lmax, 10)[1:6]
    for L in setL
        name = "DMRG\\Data\\LaughlinInfinite\\r010_Ly_$(round(L, digits=5))_chi-$(chi).jld2"
        isfile(name) ||  mainLaugh(L, chi)
        psi = load(name, "dmrgStruc")
        D = compute_entanglement_spectrum(psi.ψ)
        plot_entanglement_spectrum(D[3], 0, L, chi)
    end
    
    
    setBerryPhase100 = []
    setBerryPhase010 = []
    
    Ncell = 100
         
    for L in setL
        
        Nτ  = 200
        namesave =  "DMRG/Data/LaughlinInfinite/DehnTwist/Berryphase_Ly$(round(L, digits=5))_NTwist$(Nτ)_Ncell$(Ncell).jld2"
        println("Calculating Berryphase for Ly=$(L)")
        
        if !isfile(namesave)
            name100 = "DMRG\\Data\\LaughlinInfinite\\r100_Ly_$(round(L, digits=5))_chi-$(chi).jld2"
            name010 = "DMRG\\Data\\LaughlinInfinite\\r010_Ly_$(round(L, digits=5))_chi-$(chi).jld2"
            
            psi2 = load(name100, "dmrgStruc")
            psi1 = load(name010, "dmrgStruc")
            
            psi100 = psi1.ψ
            psi010 = psi2.ψ
            
            @time Berryphase, Nτ = BerryPhaseLoop(Nτ, copy(psi100), Ncell)
            @time BerryPhase010, Nτ  = BerryPhaseLoop(Nτ, copy(psi010), Ncell)
            append!(setBerryPhase100, Berryphase)
            append!(setBerryPhase010, Berryphase)
            #save(namesave, "berryphase", Berryphase)
        else 
            Berryphase = load(namesave, "berryphase")
            append!(setBerryPhase, Berryphase)
        end
    end

    setLsquare = [l^2 for l in setL]


    setBerryPhase100 = convert(Array{Float64,1}, setBerryPhase100)
    setBerryPhase010 = convert(Array{Float64,1}, setBerryPhase010)
    setX = setLsquare

    
    fit1 = linear_fit(setX, setBerryPhase100)
    fit2 = linear_fit(setX, setBerryPhase010)
    @show fit1
    @show fit2


    fig = plot()
    scatter!(fig, setX, setBerryPhase100)
    scatter!(fig, setX, setBerryPhase010)
    plot!(fig, setX, fit1[1] .+ fit1[2].*setX, linestyle=:dash, linewidth=2)
    plot!(fig, setX, fit2[1] .+ fit2[2].*setX, linestyle=:dash, linewidth=2)
    xlabel!(L"$L_{x}^2$")
    ylabel!(L"$U_{T}/\pi$")
    title!("Berry Phase for Dehn twist with $(Ncell) electrons", titlefont = font(10,"Computer Modern"))
    display(fig)
    

end

main(22.)

######################################
######################################

function mainPfaff(Ly)

    #dmrgstruct = Pfaff_struct(8., [0., 0., 1.])
    dmrgStruct =  FQHE_idmrg([2, 1, 2, 1], Ly, [1.], [0., 0., 1.], 0.)
	model_params = (Ly=Ly, Vs=[0., 0., 1.])
    maxIter = 4
	save_every = 2
    

    # setχ = 200:50:200
    setχ = 300:300
    saving_prefix = "DMRG\\Data\\PfaffianInfinite\\"

    for χ in setχ
        
        println("Starting χ = $χ")
        all_ener = Float64[]
        all_err = Float64[]
        all_entr = Float64[]
        #for alpha in alphas
        for x in 1:maxIter÷save_every

            ener, err, entr = idmrg(dmrgStruct; mixer = true, alpha = 1e-6, maxdim = χ, nb_iterations = save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-12, output_level = 1)
            ener, err, entr = idmrg(dmrgStruct; mixer = true, alpha = 0, maxdim = χ, nb_iterations = save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-12, output_level = 1)
            
            append!(all_ener, ener); append!(all_err, err); append!(all_entr, entr)
            
            save(saving_prefix*"Ly_$(Ly)_chi-$(χ).jld2", "dmrgStruc", dmrgStruct, "energies", all_ener, "errors", all_err, "entropies", all_entr, "params", model_params)
            length(all_ener)<10 && continue
            save_every*x < 6 && continue
            if std(all_ener[end-5:end]) < ener_tol && std(all_entr[end-5:end])< ent_tol
                println("Finishing early")
                break
            end
        end
    end
end;