using Revise
using ITensors
using ITensorInfiniteMPS
using FileIO
using JLD2
using Statistics
using Plots
using KrylovKit

function replaceTranslator(ψ1::InfiniteMPS, translator)
    ψ = deepcopy(ψ1)
    return InfiniteMPS(CelledVector(ψ.data.data, translator), ψ.llim, ψ.rlim, ψ.reverse)
end

function replaceTranslator(ψ1::InfiniteCanonicalMPS, translator)
    ψ = deepcopy(ψ1)
    return InfiniteCanonicalMPS(replaceTranslator(ψ.AL, translator), replaceTranslator(ψ.C, translator), replaceTranslator(ψ.AR, translator))
end



function fermion_momentum_translater_four(i::Index, n::Int64; N=4)
    ts = tags(i)
    translated_ts = translatecelltags(ts, n)
    new_i = replacetags(i, ts => translated_ts)
    for j in 1:length(new_i.space)
    ch = new_i.space[j][1][1].val
    mom = new_i.space[j][1][2].val
    new_i.space[j] = Pair(QN(("Nf", ch ), ("NfMom", mom + n*N*ch)),  new_i.space[j][2])
    end
    return new_i
end;


function LoadData(Ly, Settheta,  chi, noise, RootPattern, maxiters, Chimax)
    DictR = Dict()
    path = "DMRG/Data/TransitionInfiniteCylinder/2b_3b/"
    setE = []
    setPsi = Array{iDMRGStructure}(undef, length(Settheta))
    setErr =[]
    setEntr =[]
    
    for (ind, theta) in enumerate(Settheta)
        
        name  = "RootPattern$(RootPattern)_chiMax$(Chimax)_Ly$(Ly)_theta$(round(theta, digits=5))_maxiters$(maxiters)_chi$(chi)_noise$(noise).jld2"
        openName = path*name
        @show round(theta, digits=5)
        if !isfile(openName)
            println("try")
            name = "RootPattern$(RootPattern)_chiMax$(Chimax)_Ly$(Ly)_theta$(round(round(theta, digits=5)+1e-5, digits=5))_maxiters$(maxiters)_chi$(chi)_noise$(noise).jld2"
            if !isfile(path*name)
                println("sdfdgf")
                name = "RootPattern$(RootPattern)_chiMax$(Chimax)_Ly$(Ly)_theta$(round(round(theta, digits=5)-1e-5, digits=5))_maxiters$(maxiters)_chi$(chi)_noise$(noise).jld2"
            end
            openName = path*name
        end
            
        jldopen(openName, "r") do file
            setPsi[ind] = file["dmrgStruct"]
            append!(setE, file["energies"][end])
            append!(setErr,file["errors"])
            append!(setEntr, file["entropies"][end])
        end
        
    end
    DictR["Psi"] = setPsi
    DictR["Energy"] = setE
    DictR["Errors"] = setErr
    DictR["Entropy"] = setEntr

    return DictR
end


function PlotEnergy(Entr, setTheta, chi, Ly; savefig=true)
    fig= plot()

    scatter!(fig, setTheta, Entr, label="Energy")
    xlabel!(fig, "θ")
    ylabel!(fig, "Energy")
    title!(fig, "Energy for Ly=$(Ly) and Chi=$(chi)")
    #vline!([n*pi/4 for n=3:4:8])
    
    if savefig
        namesave = "DMRG/Figures/Energy_Ly$(Ly)_chi$(chi)"
        Plots.png(fig, namesave)
        Plots.svg(fig, namesave)
    end
    
    display(fig)
end

function PlotEntropy(Entr, setTheta, chi, Ly; savefig=true)
    fig= plot()

    scatter!(fig, setTheta, Entr, label="E")
    xlabel!(fig, "θ")
    ylabel!(fig, "Entropy")
    title!(fig, "Entropy for Ly=$(Ly) and Chi=$(chi)")
    vline!([n*pi/4 for n=3:4:8])
    
    if savefig
        namesave = "DMRG/Figures/Entropy_Ly$(Ly)_chi$(chi)"
        Plots.png(fig, namesave)
        Plots.svg(fig, namesave)
    end
    display(fig)
end

function plotFidelity(idmrgStruct, setTheta, chi, Ly; savefig=true)
    setλ = []
    setθ = [0.5*(setTheta[i+1]+setTheta[i]) for i in 1:(length(setTheta)-1)]

    for j in 1:(length(idmrgStruct)-1)
        ID1 = idmrgStruct[j]
        ID2 = idmrgStruct[j+1]
        ψ1 = ID1.ψ
        ψ2 = ID2.ψ

        ψ1 = replaceTranslator(ψ1, fermion_momentum_translater_four)
        ψ2 = replaceTranslator(ψ2, fermion_momentum_translater_four)
        replace_siteinds!(ψ2, siteinds(ψ1))

        T = TransferMatrix(ψ1.AL, ψ2.AL)
    
        
        vtest = randomITensor(dag(input_inds(T)))
        λ = []

        if norm(vtest) < 1e-10
            println("No 0 sector")
            λ = [0]
        else

            v = translatecell(translator(ψ1), vtest , -1)

        
            λ, _, _ = eigsolve(T, v, 1, :LM; tol=1e-10)
        end
        append!(setλ, abs(λ[1]))
    
    end

    fig = plot()
    scatter!(setθ, setλ, label="Fidelity")
    xlabel!("θ")
    ylabel!("fidelity")
    title!("Fidelity for Ly =$(Ly) and Chi = $(chi)")
    #vline!([n*pi/4 for n=3:4:8])

    
    if savefig
        namesave = "DMRG/Figures/Fidelity_Ly$(Ly)_chi$(chi)"
        Plots.png(fig, namesave)
        Plots.svg(fig, namesave)
    end

    display(fig)
end


function main()

    chi = 2048
    chimax = 2048
    Ly = 8.
    setθ = LinRange(0, 2*pi, 20)
    D = LoadData(Ly, setθ, chi, 0., "1100", 100, chimax)

    PlotEnergy(D["Energy"], setθ, chi, Ly; savefig=true)

    PlotEntropy(D["Entropy"], setθ, chi, Ly; savefig=true)

    plotFidelity(D["Psi"], setθ, chi, Ly; savefig=true)
end

main()