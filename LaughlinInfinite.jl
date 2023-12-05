using Revise
using ITensors
using ITensorInfiniteMPS
using Plots

using LinearAlgebra
using LaTeXStrings
using Statistics
using CurveFit


include("src/InfiniteCylinder.jl");
include("src/DehnTwist.jl")
include("src/entanglement_spectrum.jl")

function iDMRG_Laughlin(Ly::Float64, maxχ::Int64, RootPattern::Vector{Int64}, path::String)
   
    maxIter = 40
	save_every =2  
    
    ener_tol = 1e-8
    ent_tol = 1e-8

    setχ = 100:100:maxχ
    alphas = [1e-8, 0.0]
    
    dmrgStruct = Laughlin_struct(Ly, [1.], RootPattern)



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
                
                save(path*"Ly$(round(Ly, digits=5))_chi$(χ)_alpha$(alpha).jld2", "dmrgStruct", copy(dmrgStruct), "energies", all_ener, "errors", all_err, "entropies", all_entr)
                
                length(all_ener)<10 && continue
                save_every*x < 6 && continue

                if std(all_ener[end-5:end]) < ener_tol && std(all_entr[end-5:end])< ent_tol
                    println("Finishing early")
                    break
                end
            end
        end
    end
end;

#=
function iDMRG_Pfaff(Ly::Float64, maxχ::Int64, RootPattern::Vector{Int64}, path::String)
   
    maxIter = 40
	save_every =2  
    
    ener_tol = 1e-8
    ent_tol = 1e-8

    setχ = 100:100:maxχ
    alphas = [1e-8, 0.0]
    
    dmrgStruct = Pfaff_struct(Ly, RootPattern)



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
                
                save(path*"Ly$(round(Ly, digits=5))_chi$(χ)_alpha$(alpha).jld2", "dmrgStruct", copy(dmrgStruct), "energies", all_ener, "errors", all_err, "entropies", all_entr)
                
                length(all_ener)<10 && continue
                save_every*x < 6 && continue

                if std(all_ener[end-5:end]) < ener_tol && std(all_entr[end-5:end])< ent_tol
                    println("Finishing early")
                    break
                end
            end
        end
    end
end;
=#

function mainLoop(Ly::Float64, maxχ::Int64, RootPattern::Vector{Int64}; alpha=0.0, sectorEntanglement = 0, Ncell=50, Nτ=500)

    path = "DMRG/Data/LaughlinInfinite/"
    rootTag = "r"*RootPattern_to_string(RootPattern)*"_"
    
    prefix = path*rootTag
    nameState = "Ly$(round(Ly, digits=5))_chi$(maxχ)_alpha$(alpha).jld2"
    #generating LaughlinState
     
    isfile(prefix*nameState) || iDMRG_Laughlin(Ly, maxχ, RootPattern, prefix)
    
    psi = load(prefix*nameState, "dmrgStruct")
    ψ = psi.ψ
    
    topologicalSector = 0
    
    if RootPattern == [2,1,1]
        topologicalSector = +1
    elseif RootPattern == [1,1,2]
        topologicalSector = -1
    end

    #Plotting the Entanglement spectrum
    #D = compute_entanglement_spectrum(copy(ψ))
    #plot_entanglement_spectrum(D[3], sectorEntanglement, Ly, maxχ, RootPattern)


    Ncell = 4
    #Calculating the BerryPhase
    Berryphase, Nτ = BerryPhaseLoop(copy(ψ), Nτ, Ncell, topologicalSector)

    return Berryphase
end

#=
function mainLoopPf(Ly::Float64, maxχ::Int64, RootPattern::Vector{Int64}; topologicalSector=0, alpha=0.0, sectorEntanglement = 0, Ncell=50, Nτ=500)

    path = "DMRG/Data/PfaffianInfinite/"
    rootTag = "r"*RootPattern_to_string(RootPattern)*"_"
    
    prefix = path*rootTag
    nameState = "Ly$(round(Ly, digits=5))_chi$(maxχ)_alpha$(alpha).jld2"
    #generating LaughlinState
     
    isfile(prefix*nameState) || iDMRG_Pfaff(Ly, maxχ, RootPattern, prefix)
    
    psi = load(prefix*nameState, "dmrgStruct")
    ψ = psi.ψ
    
    #Plotting the Entanglement spectrum
    #D = compute_entanglement_spectrum(copy(ψ))
    #@show keys(D)
    #plot_entanglement_spectrum(D[4], sectorEntanglement, Ly, maxχ, RootPattern)


    Ncell = 30
    #Calculating the BerryPhase
    Berryphase, Nτ = BerryPhaseLoop(copy(ψ), Nτ, Ncell, topologicalSector)

    return Berryphase
end
=#

function AllBerryPhaseLaugh(Lmax)
   
    Lmin = 13
    chi = 500
    setL = LinRange(Lmin, Lmax, 10)
    
    setRP = [[2,1,1], [1,2,1], [1,1,2]]

    BerryMES = Dict()
    for rp in setRP
        println("####################################\n      Root pattern : $(rp) \n####################################\n")
        setBerryPhase= [] 
        
        for L in setL
            println("Calculating for L=$(L)")
            BerryPhase = mainLoop(L, chi, rp; Nτ=300)
            append!(setBerryPhase, BerryPhase)
        end

        BerryMES[RootPattern_to_string(rp)] = setBerryPhase
    end

    save("DMRG/Data/LaughlinInfinite/DehnTwist/test_Lmax$(Lmax).jld2", "berry", BerryMES, "L", setL)
    return BerryMES, setL
end





function plotBerryphase(Lmax)

    DictB, setL = load("DMRG/Data/LaughlinInfinite/DehnTwist/test_Lmax$(Lmax).jld2", "berry", "L")
    Ncell = 50
    range = 3:10
    B100 = DictB["100"][range]
    B010 = DictB["010"][range]
    B001 = DictB["001"][range]
    @show B001[3]-B100[3]
    newsetL = collect(setL)[range]

    setX = [l^2 for l in newsetL]
    @show setL
    #fit
    fit100 = linear_fit(setX, B100)
    fit010 = linear_fit(setX, B010)
    fit001 = linear_fit(setX, B001)

    @show fit100
    @show fit010
    @show fit001

    fig = plot()

    scatter!(setX, B100, marker=:dot, color="red", label="rp : 100")
    scatter!(setX, B010, marker=:square, color="blue", label="rp : 010")
    scatter!(setX, B001, marker=:diamond, color="green", label="rp : 001")
    plot!(fig, setX, fit100[1] .+ fit100[2].*setX, linestyle=:dash, color="red", linewidth=2)
    plot!(fig, setX, fit010[1] .+ fit010[2].*setX, linestyle=:dash, color="blue", linewidth=2)
    plot!(fig, setX, fit001[1] .+ fit001[2].*setX, linestyle=:dash, color="green", linewidth=2)
    xlabel!(L"$L_{x}^2$")
    ylabel!(L"$U_{T}/\pi$")
    title!("Berry Phase for Dehn twist with $(Ncell) electrons", titlefont = font(10,"Computer Modern"))

    display(fig)

    println(L"$h_{a} -h_0 =$"*"$((fit100[1]-fit001[1])/2)")
    println(L"$h_{0}-\frac{c}{24} = $", 24*fit100[1]/2)



end


AllBerryPhaseLaugh(22.)
plotBerryphase(22.)
######################################
######################################


#=
function AllBerryPhasePf(Lmax)
   
    Lmin = 7.
    chi = 500
    setL = LinRange(Lmin, Lmax, 10)
    
    setRP = [[2,2,1,1], [2,1,2,1]]

    BerryMES = Dict()
    for rp in setRP
        println("####################################\n      Root pattern : $(rp) \n####################################\n")
        setBerryPhase= [] 
        
        for L in setL
            println("Calculating for L=$(L)")
            BerryPhase = mainLoopPf(L, chi, rp; Nτ=300)
            append!(setBerryPhase, BerryPhase)
        end

        BerryMES[RootPattern_to_string(rp)] = setBerryPhase
    end

    save("DMRG/Data/PfaffianInfinite/DehnTwist/test_Lmax$(Lmax).jld2", "berry", BerryMES, "L", setL)
    return BerryMES, setL
end
=#

function plotBerryphase(Lmax)

    DictB, setL = load("DMRG/Data/PfaffianInfinite/DehnTwist/test_Lmax$(Lmax).jld2", "berry", "L")
    Ncell = 50
    range = 2:10
    B1100 = DictB["1100"][range]
    B1010 = DictB["1010"][range]
    newsetL = collect(setL)[range]

    setX = [l^2 for l in newsetL]
    
    #fit
    fit1100 = linear_fit(setX, B100)
    fit1010 = linear_fit(setX, B010)

    @show fit1100
    @show fit1010


    fig = plot()

    scatter!(setX, B1100, marker=:dot, color="red", label="rp : 1100")
    scatter!(setX, B1010, marker=:square, color="blue", label="rp : 1010")
    plot!(fig, setX, fit1100[1] .+ fit100[2].*setX, linestyle=:dash, color="red", linewidth=2)
    plot!(fig, setX, fit1010[1] .+ fit010[2].*setX, linestyle=:dash, color="blue", linewidth=2)
    xlabel!(L"$L_{x}^2$")
    ylabel!(L"$U_{T}/\pi$")
    title!("Berry Phase for Dehn twist with $(Ncell) electrons", titlefont = font(10,"Computer Modern"))

    display(fig)

    println("h_a -h_b ="*"$((fit1100[1]-fit1010[1])/2)")
    #println("h_a-c/24} = ", 24*fit1100[1]/2)
    #println(L"$h_b-c/24 = ", 24*fit1010[1]/2)



end

#AllBerryPhasePf(11.)