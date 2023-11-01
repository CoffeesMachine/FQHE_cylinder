using Revise
using ITensorInfiniteMPS
using ITensors
using FileIO
using JLD2
include("MatrixElement.jl");
include("C:/Users/basil/Documents/EPFL/Master/MasterProject/Code/ITensorInfiniteMPS.jl/src/models/fqhe13.jl")


function fermion_momentum_translater_laugh(i::Index, n::Int64; N=3)
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

#################################
#################################


function MPOTwoBody(s, Ly::Float64, Vs::Array{Float64}, tag::String; translator=nothing)
    
    dir = "DMRG\\InfMPO\\InfMPO_2body\\"
    name= "Ly$(Ly)_Vs$(Vs)_root$(tag).jld2"

    if isfile(dir*name)
        H, L, R = load(dir*name, "H", "L", "R")
        return H, L, R
    else
        model = Model("fqhe_2b_pot")
        model_params = (Ly = Ly, Vs = Vs, prec = 1e-10)

        MPO_ = InfiniteMPOMatrix(model, s, translator; model_params...)
        (H, L ,R), _,  = ITensorInfiniteMPS.compress_impo(MPO_, projection =  1, cutoff = 1e-10, verbose = true, max_iter = 500)
        save(dir*name, "H", H, "L", L, "R", R)
        return H, L, R
    end
end;   

function MPOThreeBody(s, Ly::Float64, Vs::Array{Float64}, tag::String; translator=nothing)
    
    dir = "DMRG\\InfMPO\\InfMPO_3body\\"
    name = "Ly$(Ly)_Vs$(Vs)_root$(tag).jld2"

    if isfile(dir*name)
        H, L, R = load(dir*name, "H", "L", "R")
        return H, L, R
    else
        model = Model("fqhe_gen")

        coeff = Generate_IdmrgCoeff(Ly, Vs; prec=1e-8, PHsym=true)

        MPO_ = InfiniteMPOMatrix(model, s, translator; dict_coeffs=coeff)
        println("Compressing the MPO")
        (H, L ,R), _, _ = ITensorInfiniteMPS.compress_impo(MPO_, projection = 1, cutoff = 1e-10, verbose = true, max_iter = 500)
        save(dir*name, "H", copy(H), "L", copy(L), "R", copy(R))
        return H, L, R
    end
end;

function add_MPO(MPO1, MPO2)
    
    H1, L1, R1 = MPO1
    H2, L2, R2 = MPO2

    #=
    sh1 = siteinds(H1)
    sh2 = siteinds(H2)
    for i=1:2
        replaceind!(H1[i], dag(sh1[i]), dag(sh2[i]))
        replaceind!(H2[i], prime(sh1[i]), prime(sh2[i]))
    end
    =#
    H1.data.data = a*H1.data.data
    H2.Hmpo = b*H2.Hmpo
    H = H1 + H2
    
    L = Vector{ITensor}(undef, 4)
    L[1] = L1[1] + L2[1]; 
    L[2] = L1[2]; 
    L[3] = L2[2]; 
    L[4] = L1[end];

    R = Vector{ITensor}(undef, 4)
    R[1] = R1[1]; 
    R[2] = R1[2]; 
    R[3] = R2[2]; 
    R[4] = R1[end] + R2[end];

    (newH, newL, newR), _, _ = ITensorInfiniteMPS.compress_impo(H; right_env = R, left_env = L, projection = 1, cutoff = 1e-10, verbose = true, max_iter = 500)
    return newH, newL, newR
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
    return newH, L, R2  
end;

#################################
#################################


function Generate_FQHE_MPO(RootPattern::Vector{Int64}, Ly::Float64, Vs2body::Vector{Float64}, Vs3body::Vector{Float64}, mult_2b::Float64, mult_3b::Float64)

    GenericPath = "DMRG\\InfMPO\\"

    tag = ""
    if RootPattern == [2,2,1,1]
        tag = "11"
    else
        tag = "10"
    end
    
    CellSize = length(RootPattern)

    s2 = []
    nameState = "\\States\\Ly$(Ly)_Vs2b_$(Vs2body)_Vs3b-$(Vs3body)_rootP-$(tag).jld2"
    
    if isfile(GenericPath*nameState)
        s2data = load(GenericPath*nameState, "S_two_sites")
        s2 = CelledVector(s2data, fermion_momentum_translater_two)
    else 
        s = generate_basic_FQHE_siteinds(CellSize, RootPattern; conserve_momentum=true, translator=fermion_momentum_translater_four)
        save(GenericPath*nameState, "S_two_sites", [deepcopy(s[1]), deepcopy(s[2])])
        s2 = CelledVector([deepcopy(s[1]), deepcopy(s[2])], fermion_momentum_translater_two)
    end
    
    
    iMPO_3b = MPOThreeBody(s2, Ly, Vs3body, tag; translator=fermion_momentum_translater_two)
    iMPO_2b = MPOTwoBody(s2, Ly, Vs2body, tag; translator=fermion_momentum_translater_two)
    
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

    nameAddition = "\\Add_MPO\\Ly$(Ly)_Vs2b_$(Vs2body)_Vs3b-$(Vs3body)_mult2b-$(mult_2b)_mult3b-$(mult_3b)_rootP-$(tag).jld2"
    if isfile(GenericPath*nameAddition)
        H, L, R = load(GenericPath*nameAddition, "H", "L", "R")
        
        return H, L, R, ψ
    else
        H, L, R = add_MPO(iMPO_2b, iMPO_3b, mult_2b, mult_3b)
        save(GenericPath*nameAddition, "H", H, "L", L, "R", R) 
    
        return H, L, R, ψ
    end
end 

function Generate_DMRG_struct(H, L, R, ψ)
    CellSize = nsites(H)
    H, L, R = MPO_unitcell(H, L, R, CellSize, ψ)
    sp = siteinds(ψ)
    newH = copy(H)
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

    dmrgStruc = iDMRGStructure(copy(ψ), newH, copy(newL), copy(newR), 2);
end


function FQHE_idmrg(RootPattern::Vector{Int64}, Ly::Float64, Vs2body::Vector{Float64}, Vs3body::Vector{Float64}, mult_2b::Float64, mult_3b::Float64)

    H, L, R, ψ = Generate_FQHE_MPO(RootPattern, Ly, Vs2body, Vs3body, mult_2b, mult_3b)

    dmrgStruct = Generate_DMRG_struct(H, L, R, ψ)
    
    advance_environments(dmrgStruct, 10)
    return dmrgStruct
end


function Laughlin_struct(Ly::Float64, Vs::Vector{Float64})
    s = generate_basic_FQHE_siteinds(3, [2,1,1]; conserve_momentum=true, translator=fermion_momentum_translater_laugh)

    H, L, R = MPOTwoBody(s, Ly, Vs; translator=fermion_momentum_translater_laugh, spec="Laugh")

    function initstate(n)
        if mod(n, 3) == 1
            return 2
        else
            return 1
        end
    end

    ψ = InfMPS(s, initstate)

    sp = siteinds(ψ)
    sh = siteinds(H)
    for x in 1:3
        replaceind!(H[x], dag(sh[x]), dag(sp[x]))
		replaceind!(H[x], prime(sh[x]), prime(sp[x]))
    end

    newH = copy(H)
    # newH = InfiniteMPO(H)
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

    dmrgStruc = iDMRGStructure(copy(ψ), newH, copy(newL), copy(newR), 2);
    advance_environments(dmrgStruc, 10);
    return dmrgStruc
end
