using MKL
using Revise
using ITensors
using ITensorInfiniteMPS

using FileIO
include("/home/bmorier/DMRG/FQHE_cylinder/src/MatrixElement.jl")
include("/home/bmorier/DMRG/ITensorInfiniteMPS.jl/src/models/fqhe13.jl")


function fermion_momentum_translater_General(i::Index, n::Int64, N)
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

fermion_momentum_translater_laugh(i::Index, n::Int64) = fermion_momentum_translater_General(i, n, 3)

fermion_momentum_translater_six(i::Index, n::Int64) = fermion_momentum_translater_General(i, n, 6)

fermion_momentum_translater_two(i::Index, n::Int64) = fermion_momentum_translater_General(i, n, 2)

fermion_momentum_translater_four(i::Index, n::Int64) = fermion_momentum_translater_General(i, n, 4)


function RootPattern_to_string(RootPattern::Vector{Int64}; first_term="")
    s = ""*first_term
    for el in RootPattern
        s *= string(el-1)
    end
    return s
end

function split_coeffs(dict; tol = 1e-7)
	
	new_dic = Dict{Int64, Dict{Vector{Int64}, valtype(dict)}}()
	for (k, v) in dict
		abs(v) < tol && continue
		if k[1] + k[2] + k[3] + k[4] != k[5] + k[6] + k[7] + k[8]
			error("")
		end
		m = abs(k[8]-k[1])
		if !haskey(new_dic, m)
			new_dic[m] = Dict{Vector{Int64}, valtype(dict)}()
		end
		new_dic[m][[k...]] = v
	end
	return new_dic
end


function run4B(RootPattern, Ly; spectag="")

    translatorGeneral = length(RootPattern) == 4 ? fermion_momentum_translater_four : fermion_momentum_translater_six
    translatorUnit = length(RootPattern) == 4 ? fermion_momentum_translater_two : fermion_momentum_translater_laugh

    gapTag = gap ? "gap" : ""
    tag = RootPattern_to_string(RootPattern)

    split_coeffs_name = "/scratch/bmorier/Coeff/Split_4b_Ly$(Ly)_$(tag)$(spectag).jld2"
    
    if !isfile(split_coeffs_name)
        s = generate_basic_FQHE_siteinds(length(RootPattern), RootPattern; conserve_momentum=true, translator=translatorGeneral)
        println("Number of threads is $(Threads.nthreads())")


        s3 = CelledVector([deepcopy(s[Int64(i)]) for i=1:length(RootPattern)/2], translatorUnit)
        CoeffName = "/scratch/bmorier/Coeff/Gen_4b_Ly$(Ly)_$(tag)$(spectag).jld2"
        
        Coeff = Dict()
        if !isfile(CoeffName)
            println("Generating new coefficients for Ly = $(Ly) and four body interaction")
            flush(stdout)
            rough_N = round(Int64, 2*Ly)-2
            Coeff =Generate_4Body(;Ly=Ly, N_phi=rough_N)
            save(CoeffName, "coefficients_Gen", Coeff, "s", s3)
        else
            Coeff, s3 = load(CoeffName, "coefficients_Gen", "s")
        end

        println("Split coefficients")
        new_coeffs = split_coeffs(Coeff)

        coeffs2 = Dict()
        for (k, v) in new_coeffs
            coeffs2[k] = further_split_coeffs(v, 750)
        end
        save(split_coeffs_name, "coeffs", coeffs2, "s", s3)
    else
        println("Coefficients already generated")
    end
end


