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
        Ne_unitCell = qt == 4 ? 2 : 1
    )

   return kwargs
end



function run()

    filename = "DMRG/Input_Files_DehnTwist/test.in"
    θ = 0.

    kwargs = readInputFile(filename, θ)

    setL = kwargs[2]

    path = "DMRG/Data/$(kwargs[5])Infinite/DehnTwist/" 

    filename = "DT_Lmin$(first(setL))_Lmax$(last(setL))_step$(length(setL))_chi$(kwargs[3])_Ncell$(kwargs[15])_Flux$(round(kwargs[17], digits=5)).jld2"
    #filename = "DT_Lmin$(first(setL))_Lmax$(last(setL))_step$(length(setL))_chi$(kwargs[3])_Ncell$(kwargs[6]).jld2"
    
    nameDehnTwist = path*filename
    BerryPhaseD = Dict()
    
    if isfile(nameDehnTwist)
        BerryPhaseD = load(nameDehnTwist, "dict twist")
    else
        BerryPhaseD = BerryPhase(;kwargs...)
        save(nameDehnTwist, "dict twist", BerryPhaseD)
    end

    #=
    for (k,v) in BerryPhaseD
        BerryPhaseD[k] = v[7:13]
    end
    =#
    setL = collect(setL)

    ll = setfit(setL, BerryPhaseD)
    @show (ll[1], ll[2])
    @show (ll[3], ll[4])
    
    @show ll[1]-ll[3]
    @show ll[1]-ll[5]
    setX = setL.*setL
    fig = plot()
    scatter!(fig,setX, BerryPhaseD[[2,1,1]], marker=:dot, color="red", label="rp : 100")
    scatter!(fig, setX, BerryPhaseD[[1,2,1]], marker=:0, color="blue", label="rp : 010")
    scatter!(fig, setX, BerryPhaseD[[1,1,2]], marker=:diamond, color="green", label="rp : 001")

    display(fig)
end

run()
#=
function plotBerryphase(Lmax)

    DictB, setL = load("DMRG/Data/LaughlinInfinite/DehnTwist/test_Lmax$(Lmax).jld2", "berry", "L")
    Ncell = 50
    range = 2:5
    B100 = DictB["100"]
    B010 = DictB["010"]
    B001 = DictB["001"]
    newsetL = collect(setL)

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
    #plot!(fig, setX, fit100[1] .+ fit100[2].*setX, linestyle=:dash, color="red", linewidth=2)
    #plot!(fig, setX, fit010[1] .+ fit010[2].*setX, linestyle=:dash, color="blue", linewidth=2)
    #plot!(fig, setX, fit001[1] .+ fit001[2].*setX, linestyle=:dash, color="green", linewidth=2)
    xlabel!(L"$L_{x}^2$")
    ylabel!(L"$U_{T}/\pi$")
    title!("Berry Phase for Dehn twist with $(Ncell) electrons", titlefont = font(10,"Computer Modern"))

    display(fig)

    println("h_a -h_0 =$((fit100[1]-fit001[1])/2)")
    println("h_0 - c/24 = ", fit100[1]/2)
end
=#