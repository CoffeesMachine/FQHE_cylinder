using MKL
using Revise
using ITensors
using ITensorInfiniteMPS
using Plots


using LinearAlgebra
using LaTeXStrings
using Statistics
using CurveFit
using FileIO

include("src/TopologicalProperties.jl")
include("src/Entanglement_spectrum.jl")

function readInputFile(filename, theta) 
    
    vars = Dict()
    open(filename, "r") do f
        for line in eachline(f)
            var, val = split(line, " = ")
            if var != "tag"
                vars[strip(var)] = parse(Float64, val)
            else
                vars[strip(var)] = string(val)
            end
        end
    end

    setRP = vars["tag"] == "Laughlin" ? [[2,1,1], [1,2,1], [1,1,2]] : [[2,2,1,1], [2,1,2,1]]

    setchi = collect(Int64(vars["chiMin"]):100:Int64(vars["chiMax"]))

    V2 = [1.]
    V3 = [0., 0., 1.]

    alphas = [1e-8, 0.0]

    qt = vars["tag"]=="Laughlin" ? 3 : 4

    kwargs = (
        setRP=setRP,
        setL = LinRange(vars["Lmin"], vars["Lmax"], Int64(vars["NL"])),
        χ = Int64(vars["chiM"]),
        θ = theta,
        tag = vars["tag"],
        setχ = setchi,
        V2b= V2,
        V3b = V3,
        prec = vars["prec"],
        ener_tol = vars["ener_tol"],
        ent_tol = vars["ent_tol"],
        alphas=alphas,
        maxIter = Int64(vars["maxIter"]),
        save_every = Int64(vars["save_every"]),
        Ncell = Int64(vars["Ncell"]),
        Nτ = Int64(vars["Nt"]),
        Φx = vars["FluxX"], 
        Φy = vars["FluxY"],
        q = qt,
        Ne_unitCell = qt == 4 ? 2 : 1,
        cut =Int64(vars["cut"])
    )
 
   return kwargs
end


function RootPattern_to_string(RootPattern::Vector{Int64}; first_term="")
    s = ""
    for el in RootPattern
        s *= string(el-1)
    end
    s *= first_term
    return s
end


function run()

    filename = "DMRG/Input_Files_DehnTwist/test.in"
    θ = 0.

    kwargs = readInputFile(filename, θ)

    setL = kwargs[2]

    path = "DMRG/Data/Infinite/DehnTwist/" 
    PhiX = round(kwargs[17], digits=5)
    filename = "cffd ut$(kwargs[21])_DT_Lmin$(first(setL))_Lmax$(last(setL))_step$(length(setL))_chi$(kwargs[3])_Ncell$(kwargs[15])_Flux$(PhiX).jld2"
    #filename = "DT_Lmin$(first(setL))_Lmax$(last(setL))_step$(length(setL))_chi$(kwargs[3])_Ncell$(kwargs[6]).jld2"
    
    nameDehnTwist = path*filename
    BerryPhaseD = Dict()
    
    if isfile(nameDehnTwist)
        BerryPhaseD = load(nameDehnTwist, "dict twist")
    else
        BerryPhaseD = BerryPhase(;kwargs...)
        #save(nameDehnTwist, "dict twist", BerryPhaseD)
    end

    #=
    for (k,v) in BerryPhaseD
        BerryPhaseD[k] = v[7:13]
    end3
    =#
    setL = collect(setL)
    Fit = topologicalProperties(setL, BerryPhaseD, PhiX)
    
    setX = setL.*setL
    fig = plot()
    scatter!(fig, setX, BerryPhaseD[[2,1,1]], marker=:dot, color="red", label="rp : 100")
    scatter!(fig, setX, BerryPhaseD[[1,2,1]], marker=:utriangle, color="blue", label="rp : 010")
    scatter!(fig, setX, BerryPhaseD[[1,1,2]], marker=:diamond, color="green", label="rp : 001")
    title!(string(L"$\Phi_{x} = $", PhiX))
    xlabel!(L"$L^2$")
    ylabel!(L"$\mathcal{U}_{\mathcal{T}}/\pi$")
    display(fig)
end

# run()


function runPfaff()
    setRP = [[2,2,1,1], [2,1,2,1]]
    path = "Data/TransitionInfiniteCylinder/2b_3b/"
    BerryPhaseD = Dict() 
    
    range =1:15
    for rp in setRP
        filename = "DT_rp$(RootPattern_to_string(rp))_Lmin15.0_Lmax22.0_step15_chi2048_Ncell12_Flux3.14159.jld2"
        #filename = "DT_Lmin$(first(setL))_Lmax$(last(setL))_step$(length(setL))_chi$(kwargs[3])_Ncell$(kwargs[6]).jld2"

        nameDehnTwist = path*filename
        
        el = load(nameDehnTwist, "dict twist")

        @show el
        BerryPhaseD[rp] = el[rp][range]
    end
   
    setL = collect(LinRange(15., 22, 15))[range]
    
    
    for rp in setRP
        #plot_entanglement_spectrum(setL, rp)
    end
    
    PhiX = pi
    setX = setL.*setL

    
    fit = topologicalProperties(setL, BerryPhaseD, pi)
    f1 = fit[[2,2,1,1]]
    f2 = fit[[2,1,2,1]]

    fig = plot()
    scatter!(fig,setX, BerryPhaseD[[2,2,1,1]], marker=:dot, color="red", label="rp : 1100")
    scatter!(fig, setX, BerryPhaseD[[2,1,2,1]], marker=:diamond, color="blue", label="rp : 1010")
    plot!(fig, setX, f1[1].+f1[2].*setX)
    plot!(fig, setX, f2[1].+f2[2].*setX)
    title!(string(L"$\Phi_{x} = $", PhiX))
    xlabel!(L"$L^2$")
    ylabel!(L"$\mathcal{U}_{\mathcal{T}}$")
    display(fig)

    return nothing
end;
    
runPfaff();