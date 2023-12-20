include("/home/bmorier/DMRG/FQHE_cylinder/src/Coefficients/Coefficients.jl")


Lmin = parse(Float64, ARGS[1])
Lmax = parse(Float64, ARGS[2])
NL = parse(Int64, ARGS[3])
rp = parse(Int64, ARGS[4])
type = parse(Int64, ARGS[5])

function main(Lmin, Lmax, NL, rp, type)
    RP = rp == 2 ? [2,2,1,1] : [2,1,2,1]
    
    t = type == 3 ? "three" : "two"
    
    setL = LinRange(Lmin, Lmax, NL)
    
    for L in setL
        @show t
        Generate_Coeffs(RP, L, t)
        println("End, start new L")
    flush(stdout)
    end
end


main(Lmin, Lmax, NL, rp, type)