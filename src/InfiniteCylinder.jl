using Revise
using ITensorInfiniteMPS
using ITensors
using FileIO
using JLD2
include("MatrixElement.jl");
include("/home/bmorier/DMRG/ITensorInfiniteMPS.jl/src/models/fqhe13.jl")
include("Coefficients/Coefficients.jl")

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

fermion_momentum_translater_two(i::Index, n::Int64) = fermion_momentum_translater_General(i, n, 2)

fermion_momentum_translater_four(i::Index, n::Int64) = fermion_momentum_translater_General(i, n, 4)

#################################
#################################

function RootPattern_to_string(RootPattern::Vector{Int64}; first_term="")
    s = ""*first_term
    for el in RootPattern
        s *= string(el-1)
    end
    return s
end
#################################
#################################

function GenMPOT(Ly::Float64, Vs::Array{Float64}, tag::String, type::String, model_params;spec="", s=nothing, rp=nothing, translator=nothing, gap = false)
    
    dir = "/scratch/bmorier/IMPO/"
    name= "Ly$(round(Ly, digits=5))_Interaction$(type)_RootPattern$(tag).jld2"

    if type == "two"
        if isfile(dir*name)
            H, L, R = load(dir*name, "H", "L", "R")
            return H, L, R, s
        
        else
            model = Model("fqhe_2b_pot") 
            
            infMPO = InfiniteMPOMatrix(model, s, translator; model_params...)
            
            (H, L ,R), _, _  = ITensorInfiniteMPS.compress_impo(infMPO, projection =  1, cutoff = 1e-10, verbose = true, max_iter = 500)
            save(dir*name, "H", H, "L", L, "R", R)
            return H, L, R, s
        end
    else
        return Generate_MPO(rp, Ly, type, translator;spectag=spec, gap=gap)
    end

end;   

#################################
#################################

function MPS_unitcell(s1)
    newSet = [deepcopy(s1[x]) for x=1:4]
    ss = []
    for n=1:2
        for l=1:2
            x = 2*(n-1)+l
            push!(ss, replacetags(newSet[x], "c=$(n),n=$(l)", "c=1,n=$(x)"))
        end
    end
    return CelledVector(ss, fermion_momentum_translater_four)
end;

function MPO_unitcell(H, L, R, TargetSize::Int64, ψ)
    sp = siteinds(ψ)
    sh = siteinds(H)
    cellSize = nsites(H)
    @assert cellSize == 2

    R2 = translatecell(fermion_momentum_translater_two, R, 1)
    newH = [deepcopy(H[x]) for x=1:4]
    
    for n in 1:2
        for l in 1:cellSize
            x = cellSize*(n-1) + l 
            replaceind!(newH[x], dag(sh[x]), dag(sp[x]))
            replaceind!(newH[x], prime(sh[x]), prime(sp[x]))
            replacetags!(newH[x], "Link,c=$n,n=$l", "Link,c=1,n=$x")
            if x == 1
				replacetags!(newH[x], "Link,c=0,n=$(cellSize)", "Link,c=0,n=$(TargetSize)")#; tags = "Link,c=0,pos=$(cell_size)")
			elseif l ==1
				replacetags!(newH[x], "Link,c=$(n-1),n=$(cellSize)", "Link,c=1,n=$(x-1)")#; tags = "Link,c=$(n-1),pos=$(cell_size)")
			else
				replacetags!(newH[x], "Link,c=$(n),n=$(l-1)", "Link,c=1,n=$(x-1)")#; tags = "Link,c=$(n),pos=$(l-1)")
			end
        end
    end

    replacetags!(R2[2], "Link,c=2,n=$(cellSize)", "Link,c=1,n=$(TargetSize)")
    replacetags!(L[2], "Link,c=0,n=$(cellSize)", "Link,c=0,n=$(TargetSize)")
    R = R2
    newH = InfiniteMPOMatrix(newH, fermion_momentum_translater_four)
    return newH, L, R  
end;

#################################
#################################

function GenerateBasicStructure(RootPattern::Vector{Int64}, Ly::Float64, θ::Float64, V2b::Vector{Float64}, V3b::Vector{Float64}; prec=1e-10, gap=false)

    s = generate_basic_FQHE_siteinds(length(RootPattern), RootPattern; conserve_momentum=true, translator=fermion_momentum_translater_four)
    
    s3 = CelledVector([deepcopy(s[1]), deepcopy(s[2])], fermion_momentum_translater_two)
    s2 = CelledVector([deepcopy(s[1]), deepcopy(s[2])], fermion_momentum_translater_two)
    params_3b = 0.
    params_2b = (Ly = Ly, Vs = V2b, prec = prec)

    H3, L3, R3, s3 = GenMPOT(Ly, V3b, RootPattern_to_string(RootPattern), "three", params_3b;s=s2, rp=RootPattern, translator=fermion_momentum_translater_two, gap=gap)
    H2, L2, R2, _ = GenMPOT(Ly, V2b, RootPattern_to_string(RootPattern), "two", params_2b; s=s3, translator=fermion_momentum_translater_two)
    
    s3 = siteinds(H3)
    s2 = siteinds(H2)

    for n in 1:4
        replaceind!(H2[n], dag(s2[n]), dag(s3[n]))
        replaceind!(H2[n], prime(s2[n]), prime(s3[n]))
    end
    
    
    
    iMPO_3b = (H3, L3, R3)
    iMPO_2b = (H2, L2, R2)
   
    s = MPS_unitcell(s3)

    function initstate(n)
        if RootPattern == [2,2,1,1]
            return mod(n, 5) <= 2 ? 2 : 1
        elseif RootPattern == [2,1,2,1]
            return mod(n, 2) == 1 ? 2 : 1
        else
            println("Pattern not implemented yet")
            return -1
        end
    end

    ψ = InfMPS(s, initstate)

    H1, L1, R1 = cos(θ)*iMPO_3b + sin(θ)*iMPO_2b
    println("ok")
    flush(stdout)
    H, L, R = MPO_unitcell(copy(H1), copy(L1), copy(R1), length(RootPattern), ψ)

    return H, L, R, ψ
end

function GenerateBasicStructure(RootPattern::Vector{Int64}, Ly::Float64, θ::Float64; prec=1e-10, gap=false)
    
    tag = RootPattern_to_string(RootPattern; first_term="4B_")
    
    function initstate(n)
        if RootPattern == [2,2,1,1]
            return mod(n, 5) <= 2 ? 2 : 1
        elseif RootPattern == [2,1,2,1]
            return mod(n, 2) == 1 ? 2 : 1
        else
            println("Pattern not implemented yet")
            return -1
        end
    end

    V3b = [0.]
    CellSize = length(RootPattern)
    
    params_4b=0.
    H4, L4, R4, s3 = GenMPOT(Ly, V3b, RootPattern_to_string(RootPattern), "four", 0.0; spec="4B", rp=RootPattern, translator=fermion_momentum_translater_two, gap=false)
    iMPO_4b = (H4, L4, R4)

    params_3b =0
    H3, L3, R3, _  = GenMPOT(Ly, V3b, RootPattern_to_string(RootPattern), "three", params_3b; spec="4B", s=s3, rp=RootPattern, translator=fermion_momentum_translater_two, gap=gap)
    iMPO_3b = (H3, L3, R3)
    s = MPS_unitcell(s3)

    s4 = siteinds(H4)
    s3 = siteinds(H3)

    for n in 1:4
        replaceind!(H3[n], dag(s3[n]), dag(s4[n]))
        replaceind!(H3[n], prime(s3[n]), prime(s4[n]))
    end




    ψ = InfMPS(s, initstate)

    H1, L1, R1 = cos(θ)*iMPO_3b + sin(θ)*iMPO_4b
    H, L, R = MPO_unitcell(copy(H1), copy(L1), copy(R1), CellSize, ψ)


    return copy(H), copy(L), copy(R), copy(ψ)

end

function prepareMPO(H, L, R, ψ1)
    ψ = copy(ψ1)
    sp = siteinds(ψ)
    newH = copy(H)
	temp_L = copy(L)

	for j in 1:length(L)
      
    	llink = only(commoninds(ψ.AL[0], ψ.AL[1]))
        @show llink
    	temp_L[j] = temp_L[j] * δ(llink, dag(prime(llink)))
	end
	
    temp_R = copy(R)
	for j in 1:length(R)
       
    	rlink = only(commoninds(ψ.AR[nsites(ψ)+1], ψ.AR[nsites(ψ)]))
    	temp_R[j] = temp_R[j] * δ(rlink, dag(prime(rlink)))
	end
	
    for j in 1:nsites(ψ)
    	newH.data.data[j][end, 1] .+= -1*op("N", sp[j])
	end
	ITensorInfiniteMPS.fuse_legs!(newH, temp_L, temp_R)
	newH, newL, newR = ITensorInfiniteMPS.convert_impo(newH, copy(temp_L), copy(temp_R));

    dmrgStruct = iDMRGStructure(copy(ψ), newH, copy(newL), copy(newR), 2);
    advance_environments(dmrgStruct, 10)


    return dmrgStruct
end

function FQHE_idmrg(RootPattern::Vector{Int64}, Ly::Float64, θ::Float64, type::String, SavePath::String = ""; V2b::Vector{Float64}=[1.], V3b::Vector{Float64}=[0., 0., 1.], prec=1e-10, gap=false)

    H = 0; L = 0; R = 0; ψ=0;
    
    if type != "four"
        H, L, R, ψ = GenerateBasicStructure(RootPattern, Ly, θ, V2b, V3b; prec=prec, gap=gap)
    else
        H, L, R, ψ = GenerateBasicStructure(RootPattern, Ly, θ; prec=prec, gap=gap)
    end

    if SavePath != "" && isfile(SavePath)
        println("Loading smaller State")
        flush(stdout)
        oldDmrg = load(SavePath, "dmrgStruct")
        ψ1 = oldDmrg.ψ
        sh = siteinds(ψ1)
        so = siteinds(H)
        println(sh)
        println(so)
        flush(stdout)
        for n in 1:4
            replaceind!(H[n], dag(so[n]), dag(sh[n]))
            replaceind!(H[n], prime(so[n]), prime(sh[n]))
        end

        ψ = copy(ψ1)
    end 
    println(siteinds(H))
    flush(stdout)
    return prepareMPO(H, L, R, ψ)
end