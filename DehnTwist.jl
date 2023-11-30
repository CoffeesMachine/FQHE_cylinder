using Revise
using ITensors
using ITensorInfiniteMPS

include("InfiniteCylinder.jl")

function torus_mps(ψ::InfiniteCanonicalMPS, range::AbstractRange)
    @assert isone(step(range))
    N = length(range)
    ψ_finite = ψ.AR[range]
    
    ψ_finite[1] *= ψ.C[first(range)-1]
    #=
    l0 = linkind(ψ.AL, first(range) - 1 => first(range))
    l̃0 = sim(l0)
    lN = linkind(ψ.AR, last(range) => last(range) + 1)
    l̃N = sim(lN)
    δl0 = δ(dag(l̃0), l0)
    δlN = δ(dag(l̃N), lN)
    ψ_finite[1] *= δl0
    ψ_finite[N] *= dag(δlN)
    ψ_finite = MPS([ψ_finiteᵢ for ψ_finiteᵢ in ψ_finite])
    =#
    return ψ_finite
end

function fermion_momentum_translater_laugh_values(i::Index, n::Int64; N=3)
    new_i = copy(i)
    for j in 1:length(new_i.space)
        ch = new_i.space[j][1][1].val
        mom = new_i.space[j][1][2].val
        new_i.space[j] = Pair(QN(("Nf", ch ), ("NfMom", mom + n*N*ch)),  new_i.space[j][2])
    end
    return new_i
end;


function DehnTwist(Ncell::Int64, ψ::InfiniteCanonicalMPS, τ::Float64, fluxY::Number)
    NePerUnitCell = 1
  
    ψ = copy(ψ)
    input_inds = inds(ψ.C[0])[1]
    diag =[]
    for qn in input_inds.space
        N = qn[1][1].val
        K = qn[1][2].val
        dim = qn[2]

        expFactor = exp(im*pi*(Ncell*NePerUnitCell - 1)*N)*exp(-2*pi*im*τ*K/2 + im*fluxY*N)
        Gp = expFactor*ones(dim)
        append!(diag, Gp)
    end

    output_inds = replacetags(copy(input_inds), "c=0" => "c=5")

    G = diagm(collect(Iterators.flatten(diag)))
    glueOperator = ITensor(G, dag(input_inds), output_inds)

    return glueOperator
end 

function BerryPhaseLoop(N_step::Int64, ψ::InfiniteCanonicalMPS, N_cell::Int64; q=3)
    BerryFac = 0.
    BerryMult = 1.
    setMPS = Array{ITensor}(undef, N_step)
    setTau = LinRange(0., 1., N_step)
    

    Gauge = q*pi*N_cell/12
    Gauge2 = q*pi*N_cell^2/12

    MPS_ = torus_mps(copy(ψ), 1:q*N_cell)
    #TwistMPS = exp(im*Gauge2).*copy(MPS_)
    
    #DenseMPS = ContractUnitCell(copy(MPS_), 3)
    #@show norm(DenseMPS)
    #DenseTwistMPS = ContractUnitCell(TwistMPS, 5)

    #BlockTwist = ContractMPS(MPS_)

    for (ind, τ) in enumerate(setTau)
        mod(ind, 50) == 0 && println("DehnTwist for τ=$(τ)")
        setMPS[ind]  =  DehnTwist(N_cell, ψ, τ, pi)
        
        ind == 1 && continue
        #ind == N_step && continue

        BerryEl = BerryConnection(MPS_, copy(setMPS[ind]), copy(setMPS[ind-1]))
        BerryMult*=BerryEl
        if abs(BerryEl) < 0.99
            @show abs(BerryEl)
            println("Modulus too small, retry with smallest step")
            return BerryPhaseLoop(N_step + 100, copy(ψ), N_cell)
        end
        
        BerryFac -= imag(log(BerryEl))
    end

    #BerryEl = BerryConnection(DenseTwistMPS, DenseMPS, setMPS[end], setMPS[end-1])
    BerryEl = BerryConnection(MPS_, setMPS[end], setMPS[1])
    #@show BerryEl
    #BerryFac = -imag(log(BerryMult*BerryEl*exp(-im*Gauge2)))
    #BerryFac += imag(log(exp(im*Gauge2)))
    
    @show BerryFac/pi

    return BerryFac/pi, N_step
end


function ContractUnitCell(psi::MPS, q::Int64)
    ψ = copy(psi)
    legsOut = [ψ[1], ψ[end]]

    ContractedMPS = Array{ITensor}(undef, div(length(ψ), q))

    for ind in 1:div(length(ψ), q)
        flag = ind==1 ? true : false
        interval = q*(ind-1)+2:q*ind+1
        ContractedMPS[ind] = ContractCell(ψ[interval], q, flag)
    end

    MPS_ = MPS([legsOut[1]; [ψᵢ for ψᵢ in ContractedMPS]; legsOut[2]])

    return MPS_
end


function ContractCell(psi1::MPS, q, flag::Bool)
    
    ψ1 = copy(psi1)
    C = combiner(inds(ψ1[q])[2], inds(ψ1[q-1])[2]; tags="c_$(q)")
    combined = ψ1[q]*ψ1[q-1]
    newT = combined*C
    for ind in q-2:-1:2
        C = combiner(inds(newT)[1], inds(ψ1[ind])[2]; tags="c_$(ind)")
        newComb = newT*ψ1[ind]
        newT = newComb*C
    end
    
    place = flag ? 1 : 2
    C = combiner(inds(newT)[1], inds(ψ1[1])[place]; tags="c_$(1)")
    newComb = newT*ψ1[1]
    newT = newComb*C

    return newT
end

function BerryConnection(MPS_::MPS, G1::ITensor, G2::ITensor)


    replaceind!(G2, inds(G2)[1], inds(MPS_[1])[3])
    replaceind!(G1, inds(G1)[1], inds(MPS_[1])[3])
    
    normM = norm(MPS_)
    ψ_up = copy(MPS_)
    ψ_dw = copy(MPS_)
    newUp = ψ_up[1]*G2
    newDw = ψ_dw[1]*G1
    ψ_up[1] = newUp
    ψ_dw[1] = newDw
    el =inner(ψ_dw, ψ_up)/normM^2
    return el
end

function BerryConnection(B1::MPS, B2::MPS, G1::ITensor, G2::ITensor)
    
    
    ψ_up = copy(B2)[2:end-1]
    ψ_dw = copy(B1)[2:end-1]


    replaceind!(G2, inds(G2)[1], inds(ψ_up[1])[3])
    replaceind!(G1, inds(G1)[1], inds(ψ_dw[1])[3])


    normM = norm(B1)


    
    newUp = ψ_up[1]*G2
    newDw = ψ_dw[1]*G1
    ψ_up[1] = newUp
    ψ_dw[1] = newDw


    ψ1 = contract(ψ_up)
    ψ2 = contract(ψ_dw)

    ψ1 = ψ1*delta(inds(dag(ψ1))[2], inds(dag(ψ1))[end])
    ψ2 = ψ2*delta(inds(dag(ψ2))[2], inds(dag(ψ2))[end])
    
    replaceinds!(ψ2, inds(ψ2), inds(ψ1))

    el =inner(ψ2, ψ1)/normM^2
    @show el
    return el
end

function ContractMPS(ψ::MPS)

    psi = copy(ψ)[2:end-1]
    ψd = prime(dag(copy(ψ)), 1, "Link")[2:end-1]
    Contractor = Array{ITensor}(undef, length(ψ)-2)
    for i in 1:length(psi)
        a = contract(psi[i], ψd[i])
        Contractor[i] = a
    end
    Contr1 = Contractor[1]*Contractor[2]
    for t in 3:length(Contractor)
        newContr = Contr1*Contractor[t]
        Contr1 =newContr
    end
    indsUp = [inds(dag(Contr1))[1], inds(dag(Contr1))[3]]
    indsDw = [inds(dag(Contr1))[2], inds(dag(Contr1))[4]]
    Contr = Contr1*delta(indsUp[1], indsUp[2])*delta(indsDw[1], indsDw[2])

    return newψ
end



#=
function ITensors.op(::OpName"ProjOcc", ::SiteType"FermionK,Site")
    M = zeros(2,2)
    M[2,2] = 1.
    return M
end


function BackOp(psi1::MPS, psi2::MPS)
    p1 = copy(dag(psi1))
    p2 = copy(psi2)

    ψ1 = prepareMPSTorus(p1)
    psi2 = p2[2:end-1]

    s = siteinds(psi2)
    @show s
    opp = OpSum()
    for (ind, site) in enumerate(s)
        gamma = pi*ind^2/(length(s)) + pi*length(s)/12
        add!(opp, exp(-im*gamma), "ProjOcc", ind)
    end

    GaugeOp = MPO(opp, s)
    ψ2 = prepareMPSTorus(p2)
    newψ2 = replaceprime(contract(GaugeOp, ψ2), 2 => 1)
    ITensors.orthogonalize!(ψ1, 1)
    ITensors.orthogonalize!(newψ2, 1)


    normalize!(ψ1)
    normalize!(newψ2)

    
    BerryFactor = ψ1[1]*newψ2[1]*delta(inds(dag(ψ1[1]))[end], inds(dag(newψ2[1]))[end])
    return BerryFactor[1]
end
=#

