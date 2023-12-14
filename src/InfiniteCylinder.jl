using Revise
using ITensorInfiniteMPS
using ITensors
using FileIO
using JLD2
include("MatrixElement.jl");
include("C:/Users/basil/Documents/EPFL/Master/MasterProject/Code/ITensorInfiniteMPS.jl/src/models/fqhe13.jl")

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
    s = ""
    for el in RootPattern
        s *= string(el-1)
    end
    s *= first_term
    return s
end


#################################
#################################

#=
function MPOTwoBody(s::CelledVector, Ly::Float64, Vs::Array{Float64}, tag::String; translator=nothing)
    
    dir = "DMRG\\Data\\InfMPO\\InfMPO_2body\\"
    name= "Ly$(Ly)_Vs$(Vs)_root$(tag).jld2"

    if isfile(dir*name)
        H, L, R = load(dir*name, "H", "L", "R")
        return H, L, R
    else
        model = Model("fqhe_2b_pot")
        model_params = (Ly = Ly, Vs = Vs, prec = 1e-10)

        MPO_ = InfiniteMPOMatrix(model, s, translator;  model_params...)
        (H, L ,R), _, _  = ITensorInfiniteMPS.compress_impo(MPO_, projection =  1, cutoff = 1e-10, verbose = true, max_iter = 500)
        save(dir*name, "H", H, "L", L, "R", R)
        return H, L, R
    end
end;   
=#

function GenMPO(s::CelledVector, Ly::Float64, Vs::Array{Float64}, tag::String, type::String, model_params; translator=nothing)
    

    foldertype = type=="two" ? "InfMPO_2body/" : "InfMPO_3body/"
    dir = "DMRG/Data/InfMPO/"*foldertype

    name= "Ly$(Ly)_Vs$(Vs)_root$(tag).jld2"

    if isfile(dir*name)
        H, L, R = load(dir*name, "H", "L", "R")
        return H, L, R
    else
        model = type=="two" ? Model("fqhe_2b_pot") : Model("fqhe_gen")

        infMPO = type == "three" ? InfiniteMPOMatrix(model, s, translator;  dict_coeffs= model_params) : InfiniteMPOMatrix(model, s, translator; model_params...) 
        (H, L ,R), _, _  = ITensorInfiniteMPS.compress_impo(infMPO, projection =  1, cutoff = 1e-10, verbose = true, max_iter = 500)
        save(dir*name, "H", H, "L", L, "R", R)
        return H, L, R
    end
end;   

#=
function MPOThreeBody(s::CelledVector, Ly::Float64, Vs::Array{Float64}, tag::String; translator=nothing)
    
    dir = "DMRG\\Data\\InfMPO\\InfMPO_3body\\"
    name = "Ly$(Ly)_Vs$(Vs)_root$(tag).jld2"

    if isfile(dir*name)
        H, L, R = load(dir*name, "H", "L", "R")
        return H, L, R
    else
        model = Model("fqhe_gen")
        coeff = Generate_IdmrgCoeff(Ly, Vs; prec=1e-9, PHsym=false)
        
        MPO1 = InfiniteMPOMatrix(model, s, translator; dict_coeffs=coeff)

        println("Compressing the MPO")
        (H, L ,R), _, _ = ITensorInfiniteMPS.compress_impo(MPO1, projection = 1, cutoff = 1e-8, verbose = true, max_iter = 500)
        save(dir*name, "H", copy(H), "L", copy(L), "R", copy(R))
        return copy(H), L, R
    end
end;
=#
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
    @show typeof(H)
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

function GenerateBasicStructure(RootPattern::Vector{Int64}, Ly::Float64, θ::Float64, V2b::Vector{Float64}, V3b::Vector{Float64}; prec=1e-10)

    s = generate_basic_FQHE_siteinds(length(RootPattern), RootPattern; conserve_momentum=true, translator=fermion_momentum_translater_four)
    
    s3 = CelledVector([deepcopy(s[1]), deepcopy(s[2])], fermion_momentum_translater_two)
    s2 = CelledVector([deepcopy(s[1]), deepcopy(s[2])], fermion_momentum_translater_two)

    params_3b = (dict_coeffs = Generate_IdmrgCoeff(Ly, V3b; prec=prec, PHsym=false))
    params_2b = (Ly = Ly, Vs = V2b, prec = prec)

    iMPO_3b = GenMPO(s2, Ly, V3b, RootPattern_to_string(RootPattern), "three", params_3b; translator=fermion_momentum_translater_two)
    iMPO_2b = GenMPO(s3, Ly, V2b, RootPattern_to_string(RootPattern), "two", params_2b; translator=fermion_momentum_translater_two)

   
    s = MPS_unitcell(s2)

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
    H, L, R = MPO_unitcell(copy(H1), copy(L1), copy(R1), length(RootPattern), ψ)

    return H, L, R, ψ
end

function GenerateBasicStructure(RootPattern::Vector{Int64}, Ly::Float64, V::Vector{Float64}; prec=1e-10)

    s = generate_basic_FQHE_siteinds(3, RootPattern; conserve_momentum=true, translator=fermion_momentum_translater_laugh)

    params = (Ly = Ly, Vs = V, prec = prec)
    H, L, R = GenMPO(s, Ly, V, RootPattern_to_string(RootPattern;first_term="Laughlin"), "two", params; translator=fermion_momentum_translater_laugh)


    function initstate(n; pose_e=findmax(RootPattern)[2])
        if mod(n, 3) == mod(pose_e, 3)
            return 2
        else
            return 1
        end
    end

    ψ = InfMPS(s, initstate)

    return H, L, R, ψ
end

function InfCylinderStruct(RootPattern::Vector{Int64}, Ly::Float64, θ::Float64, V2b::Vector{Float64}, V3b::Vector{Float64}; prec=1e-10)
    if length(RootPattern) == 3
        return GenerateBasicStructure(RootPattern, Ly, V2b, prec=prec)
    else
        return GenerateBasicStructure(RootPattern, Ly, θ, V2b, V3b; prec=prec)
    end
end

function FQHE_idmrg(RootPattern::Vector{Int64}, Ly::Float64, θ::Float64; V2b::Vector{Float64}=[1.], V3b::Vector{Float64}=[0., 0., 1.], prec=1e-10)

    H, L, R, ψ = InfCylinderStruct(RootPattern, Ly, θ, V2b, V3b; prec=prec)
    
    sp = siteinds(ψ)
    newH = copy(H)
    so = siteinds(newH)

    for n in 1:4
        replaceind!(newH[n], dag(so[n]), dag(sp[n]))
        replaceind!(newH[n], prime(so[n]), prime(sp[n]))
    end


	temp_L = copy(L)
	for j in 1:length(L)
    	llink = only(commoninds(ψ.AL[0], ψ.AL[1]))
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