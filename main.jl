using MKL
using Revise
using ITensors
using ITensorInfiniteMPS

using Statistics
using FileIO

include("src/TopologicalProperties.jl")

index = parse(Int64, ARGS[1])

function readInputFile(filename, index)
    
    vars = Dict()
    open(filename) do ff
        for line in eachline(ff)
            var, val = split(line, " = ")
            if var != "tag" && var != "FixL"
                vars[strip(var)] = parse(Float64, val)
            else
                vars[strip(var)] = string(val)
            end
        end
    end

    RP = vars["rp"] == 2. ? [2,2,1,1] : [2,1,2,1]

    setL = LinRange(vars["Lmin"], vars["Lmax"], Int64(vars["NL"]))
    setT = LinRange(vars["thetamin"], vars["thetamax"], Int64(vars["NT"]))

    theta = 0
    TypeOfMeasure = vars["FixL"] == "true" ? "Fidelity" : "Dehn twist"


    if vars["FixL"] == "true"
        theta = setT[index]    
    else
        #L = setL[index]
        theta = setT[1]
    end

    setChi = [2^n for n=Int64(vars["chiMin"]):Int64(vars["chiMax"])]



    V2 = [1.]
    V3 = [0., 0., 1.]

    alphas = [1e-8, 0.0]

    kwargs = (
        rp = RP,
        setL = setL,
        χ = Int64(vars["chiMeasure"]),
        θ = theta,
        tag = vars["tag"],
        setχ = setChi,
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
        q = 4,
        Ne_unitCell = 2
    )

   return kwargs, TypeOfMeasure
end

function run(index)
    filename = "ParametersFiles/PfaffianDehnTwist_T0_1100.in"

    kwargs, TypeOfMeasure = readInputFile(filename, index)
    if TypeOfMeasure == "Dehn twist"
        setL = kwargs[2]

        path = "/scratch/bmorier/$(kwargs[5])_DT/" 

        filename = "DT_rp$(RootPattern_to_string(kwargs[1]))_Lmin$(first(setL))_Lmax$(last(setL))_step$(length(setL))_chi$(kwargs[3])_Ncell$(kwargs[15])_Flux$(round(kwargs[17], digits=5)).jld2"
        
        nameDehnTwist = path*filename
        BerryPhaseD = Dict()
        
  
        BerryPhaseD = BerryPhase(; kwargs...)
        save(nameDehnTwist, "dict twist", BerryPhaseD)
    
    else
        println("Skip Dehn twist calculation")
        flush(stdout)
        idmrgLoop(; kwargs...)
    end

    println("Done")
end

run(index)
