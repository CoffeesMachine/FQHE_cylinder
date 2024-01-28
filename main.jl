using MKL
using Revise
using ITensors
using ITensorInfiniteMPS

using Statistics
using FileIO

include("src/TopologicalProperties.jl")

index = parse(Int64, ARGS[1])
# RP = "101000"
RP = ARGS[2]

function readInputFile(filename, index, tagRP)
    setBoolean = ["Haffnian", "tag", "FixL", "skipDT", "reloadStruct", "gap", "ReRunReload"]
    vars = Dict()
    open(filename) do ff
        for line in eachline(ff)
            var, val = split(line, " = ")
            
            vars[strip(var)] = var in setBoolean ? string(val) :  parse(Float64, val)
            
        end
    end
    
    RP = [parse(Int64, x)+1 for x in collect(tagRP)]
    setL = LinRange(vars["Lmin"], vars["Lmax"], Int64(vars["NL"]))
    setT = LinRange(vars["thetamin"], vars["thetamax"], Int64(vars["NT"]))

    theta = 0
    TypeOfMeasure = vars["FixL"] == "true" ? "Fidelity" : "Dehn twist"


    if vars["FixL"] == "true"
        theta = setT[index]    
    else
        L = setL[index]
        theta = setT[1]
        setL = [L]
    end

    setChi = [2^n for n=Int64(vars["chiMin"]):Int64(vars["chiMax"])]



    V2 = [1.]
    V3 = [0., 0., 1.]

    # alphas = vars["NoiseReload"] == 0 ? [0.0] : [1e-8, 0.0]
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
        stepL = vars["stepL"],
        ChiTest = Int64(vars["ChiTest"]),
        q = 4,
        Ne_unitCell = 2,
        skipDT = vars["skipDT"] == "true" ? true : false,
        reloadStruct = vars["reloadStruct"] == "true" ? true : false, 
        gap = vars["gap"] == "true" ? true : false,
        χMaxReload = Int64(vars["ChiMaxReload"]), 
        ReRunReload = vars["ReRunReload"] == "true" ? true : false,
        NoiseReload = vars["NoiseReload"],
        Haffnian = vars["Haffnian"] == "true" ? true : false,
    )

   return kwargs, TypeOfMeasure
end

function run(index, tagRP)
    filename = "ParametersFiles/Phase.in"

    kwargs, TypeOfMeasure = readInputFile(filename, index, tagRP)
    println("\n####################################\n          Root pattern : $(RootPattern_to_string(kwargs[1])), $(kwargs[5])   \n####################################\n")
    flush(stdout)
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


run(index, RP)