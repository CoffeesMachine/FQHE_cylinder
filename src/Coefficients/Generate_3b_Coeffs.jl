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

function split_coeffs3B(dict; tol = 1e-12)
	new_dic = Dict{Int64, Dict{Vector{Int64}, valtype(dict)}}()
	for (k, v) in dict
		abs(v) < tol && continue
   
		if k[1] + k[2] + k[3] != k[4] + k[5] + k[6] 
            println(k)
			error("")
		end
		m = abs(k[6]-k[1])
		if !haskey(new_dic, m)
			new_dic[m] = Dict{Vector{Int64}, valtype(dict)}()
		end
		new_dic[m][[k...]] = v
	end
	return new_dic
end

function further_split_coeffs(coeffs, maxSize)
    new_coeffs = Dict{Int64, typeof(coeffs)}()
    le = length(coeffs)
    current_count = 1
	temp_coeffs = Dict{keytype(coeffs), valtype(coeffs)}()
    done = Dict{keytype(coeffs), valtype(coeffs)}()
    ks = collect(keys(coeffs))
    for k in ks
        haskey(done, k) && continue
        temp_coeffs[k] = coeffs[k]
        temp_coeffs[reverse(k)] = coeffs[reverse(k)]
        done[k] = 1
        done[reverse(k)] = 1
		if length(temp_coeffs) > maxSize
	  		new_coeffs[current_count] = copy(temp_coeffs)
	  		current_count += 1
	  		temp_coeffs = Dict()
  		end
	end
	@assert length(coeffs) == length(done)
	if length(temp_coeffs) != 0
		new_coeffs[current_count] = copy(temp_coeffs)
	end
	println("Final number of blocks: $(length(new_coeffs))")
	return new_coeffs
end


function run3B(RootPattern, Ly; spectag="", gap=false)

    translatorGeneral = length(RootPattern) == 4 ? fermion_momentum_translater_four : fermion_momentum_translater_six
    translatorUnit = length(RootPattern) == 4 ? fermion_momentum_translater_two : fermion_momentum_translater_laugh

    gapTag = gap ? "gap" : ""
    tag = RootPattern_to_string(RootPattern)
    
    
    split_coeffs_name = "/scratch/bmorier/Coeff/Split_3b$(gapTag)_Ly$(round(Ly, digits=5))_$(tag)$(spectag).jld2"
    
    if !isfile(split_coeffs_name)
        s = generate_basic_FQHE_siteinds(length(RootPattern), RootPattern; conserve_momentum=true, translator=translatorGeneral)


        s3 = CelledVector([deepcopy(s[Int64(i)]) for i=1:length(RootPattern)/2], translatorUnit)
        CoeffName = "/scratch/bmorier/Coeff/Gen_3b$(gapTag)_Ly$(round(Ly, digits=5))_$(tag)$(spectag).jld2"
        
        Coeff = Dict()
        if !isfile(CoeffName)
            println("Generating new coefficients for Ly = $(Ly) with gap=$(gap)")
            Coeff = build_three_body_pseudopotentials(;Ly=Ly, N_phi=round(Int64, 2*Ly), gap=gap)
            save(CoeffName, "coefficients_Gen", Coeff, "s", s3)
        else
            Coeff, s3 = load(CoeffName, "coefficients_Gen", "s")
        end

        println("Split coefficients")
        new_coeffs = split_coeffs3B(Coeff)
        
        coeffs2 = Dict()
        for (k, v) in new_coeffs
            coeffs2[k] = further_split_coeffs(v, 750)
        end
        
        save(split_coeffs_name, "coeffs", coeffs2, "s", s3)
    else
        println("Coefficients already generated")
    end
end