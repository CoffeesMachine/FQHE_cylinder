using MKL
using Revise
using ITensors
using ITensorInfiniteMPS

using FileIO
include("/home/bmorier/DMRG/FQHE_cylinder/src/MatrixElement.jl")
include("/home/bmorier/DMRG/ITensorInfiniteMPS.jl/src/models/fqhe13.jl")


function fermion_momentum_translater_two(i::Index, n::Int64; N=2)
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

function split_coeffs3B(dict; tol = 1e-8)
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


function run3B(RootPattern, Ly)

    tag = RootPattern == [2,2,1,1] ? "1100" : "1010"
    split_coeffs_name = "/scratch/bmorier/Coeff/Split_3b_Ly$(round(Ly, digits=5))_$(tag).jld2"
    
    if !isfile(split_coeffs_name)
        s = generate_basic_FQHE_siteinds(4, RootPattern; conserve_momentum=true, translator=fermion_momentum_translater_four)
        println("Number of threads is $(Threads.nthreads())")


        s3 = CelledVector([deepcopy(s[1]), deepcopy(s[2])], fermion_momentum_translater_two)
        CoeffName = "/scratch/bmorier/Coeff/Gen_3b_Ly$(round(Ly, digits=5))_$(tag).jld2"
        
        Coeff = Dict()
        if !isfile(CoeffName)
            println("Generating new coefficients for Ly = $(Ly)")
            Coeff = build_three_body_pseudopotentials(;Ly=Ly, N_phi=Int64(2*Ly))
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