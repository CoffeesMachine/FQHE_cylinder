using Revise
using ITensors
using ITensorInfiniteMPS

include("Generate_3b_Coeffs.jl")
include("Generate_4b_Coeffs.jl")
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


function RootPattern_to_string(RootPattern::Vector{Int64}; first_term="")
    s = ""*first_term
    for el in RootPattern
        s *= string(el-1)
    end
    return s
end

function emptyMPO(V::Vector{Dict{Vector{Any}, Float64}})
    ret =  Dict{Vector{Any}, Float64}[]
    for el in V
        isempty(el) && continue
        push!(ret, el)
    end
    return ret
end


function Generate_MPO(rp::Vector{Int64}, Ly::Float64, type::String, translator; spectag="", gap=false)
    
    gapTag = gap ? "gap" : ""
    tagType = type == "three" ? "3b"*gapTag : "4b"
    tag = RootPattern_to_string(rp) 
    DirAdd = type == "three" ? "" : "FourBody/"
    dir = "/scratch/bmorier/IMPO"
    CoeffName =  "/scratch/bmorier/Coeff/Split_"*tagType*"_Ly$(round(Ly, digits=5))_$(tag)$(spectag).jld2"
    model = Model("fqhe_gen")
   
    isfile(CoeffName) || Generate_Coeffs(rp, Ly, type, spectag, gap)

    Coeff, s = load(CoeffName, "coeffs", "s")
    mpo_file = dir*DirAdd*"/Ly$(round(Ly, digits=5))_Int$(type)_RootPattern$(tag).jld2"

    H = 0; 
    L = 0; 
    R = 0; 
    counter = 0


    if isfile(mpo_file)
        println("Loading the MPO")
        flush(stdout)
        H, L, R, counter = load(mpo_file, "H", "L", "R", "counter")
        println("Done $(counter+1)/$(length(Coeff))")
        flush(stdout)
    else
        println("Starting $(counter+1)/$(length(Coeff))")
        flush(stdout)
    
        @time coeff_ham = Generate_Idmrg(Coeff[counter])
        println("starting infinite MPO")
        flush(stdout)
        HMPO = InfiniteMPOMatrix(model, s, translator; dict_coeffs = coeff_ham)
        (H, L, R), _, _ = ITensorInfiniteMPS.compress_impo(HMPO, projection = 1, cutoff = 1e-10, verbose = true, max_iter = 500);
        save(mpo_file, "H", H,  "L", L, "R", R, "counter", 0);
    end


    for m in counter+1:length(Coeff)-1
        println("Starting $(m+1)/$(length(Coeff))")
        flush(stdout)
        coeff_ham = Generate_Idmrg(Coeff[m])
        #@show typeof(coeff_ham)
        #coeff_ham = emptyMPO(coeff_ham)

        isempty(coeff_ham) && continue
        
        HMPO_1 = InfiniteMPOMatrix(model, s, translator; dict_coeffs = coeff_ham);
        (H1, L1, R1), _, _ = ITensorInfiniteMPS.compress_impo(HMPO_1, projection = 1, cutoff = 1e-10, verbose = true, max_iter = 500);

        newH = H + H1
		newL = Vector{ITensor}(undef, 4)
		newL[1] = L[1] + L1[1]; newL[2] = L[2]; newL[3] = L1[2]; newL[4] = L[end];
		newR = Vector{ITensor}(undef, 4)
		newR[1] = R[1]; newR[2] = R[2]; newR[3] = R1[2]; newR[4] = R[end] + R1[end];

        (H, L, R), _, _ = ITensorInfiniteMPS.compress_impo(newH; right_env = newR, left_env = newL, projection = 1, cutoff = 1e-10, verbose = true, max_iter = 500)
        save(mpo_file, "H", H,  "L", L, "R", R, "counter", m);
    end

    return H, L, R, s

end

function Generate_Coeffs(rp::Vector{Int64}, Ly::Float64, type::String, spectag, gap) 
    
    if type == "three" 
        run3B(rp, Ly; spectag=spectag, gap=gap)
    elseif type == "four" 
        run4B(rp, Ly; spectag=spectag)
    else
        println("type = $type")
        run3B(rp, Ly)
    end

end
 