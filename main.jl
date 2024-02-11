using MKL
using Revise
using ITensors
using ITensorInfiniteMPS

using Statistics
using FileIO

include("src/iDMRGLoop.jl")

index = parse(Int64, ARGS[1])
# RP = "110000"
RP = ARGS[2]

function readInputFile(filename, index, tagRP)
    setBoolean = ["Haffnian", "tag", "FixL", "reloadStruct", "gap", "ReRunReload"]
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
  
    theta = 0.


    if vars["NL"] == 1.
        theta = setT[index]    
    else
        L = setL[index]
        theta = setT[1]
        setL = [L]
    end

    setChi = [2^n for n=Int64(vars["chiMin"]):Int64(vars["chiMax"])]

    parseBool(name::String) = vars[name] == "true" 

    V2 = [1.]
    V3 = [0., 0., 1.]

    # alphas = vars["NoiseReload"] == 0 ? [0.0] : [1e-8, 0.0]
    alphas = [1e-8, 0.0]

    kwargs = (
        rp = RP,
        setL = setL,
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
        stepL = vars["stepL"],
        ChiTest = Int64(vars["ChiTest"]),
        reloadStruct = parseBool("reloadStruct"),
        gap = parseBool("gap"),
        χMaxReload = Int64(vars["ChiMaxReload"]), 
        ReRunReload = parseBool("ReRunReload"),
        NoiseReload = vars["NoiseReload"],
        Haffnian = parseBool("Haffnian")
    )

   return kwargs
end

function run(index, tagRP)
    filename = "ParametersFiles/Phase.in"

    kwargs = readInputFile(filename, index, tagRP)
    
    println("Starting iDMRG loop ")
    flush(stdout)
    
    idmrgLoop(; kwargs...)

    println("Done")
end


run(index, RP)