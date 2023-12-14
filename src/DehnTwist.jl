using Revise
using ITensors
using ITensorInfiniteMPS

using LinearAlgebra

function TwistOperator(range::UnitRange{Int64}, ψ::InfiniteCanonicalMPS, τ::Float64, topologicalShift::Int64, q::Int64; Ne_unitCell::Int64=1, Φx::Float64, Φy::Float64, kwargs...)
    
    input_inds = inds(ψ.AL[1])[1]
    diag =[]
    for qn in input_inds.space
        N = qn[1][1].val/q
        K = (qn[1][2].val - topologicalShift + (Φx/2π)*qn[1][1].val)/q
        dim = qn[2]

        expFactor = exp(im*π*(nsites(ψ)*Ne_unitCell - 1)*N)*exp(-2π*im*τ*K + im*Φy*N)
        Gp = expFactor*ones(dim)
        append!(diag, Gp)
    end

    output_inds = replacetags(copy(input_inds), "c=0" => "c=$(div(last(range), nsites(ψ)))")

    G = diagm(collect(Iterators.flatten(diag)))
    TwistOperator = ITensor(G, dag(input_inds), output_inds)

    return TwistOperator
end 


function DehnTwist(ψ::InfiniteCanonicalMPS, Nτ::Int64, N_cell::Int64, topologicalSector::Int64, namesave::String; q::Int64, kwargs...)
    BerryFac = 0.

    setMPS = Array{ITensor}(undef, Nτ)
    setTau = LinRange(0., 1., Nτ)

    Gauge = q*pi*N_cell/12
    

    rangeT = 1:q*N_cell
    ψT = ψ.AL[rangeT]

    BlockMPS = 0
    if isfile(namesave)
        BlockMPS = load(namesave, "block")
    else
        @time BlockMPS = contractTorus(ψT)
        save(namesave, "block", BlockMPS)
    end

    for (ind, τ) in enumerate(setTau)
        mod(ind-1, 100) == 0 && println("Calculating Dehn twist for τ = $τ")
        flush(stdout)
        setMPS[ind]  =  TwistOperator(rangeT, ψ, τ, topologicalSector, q; kwargs...)
        
        ind == 1 && continue
        BerryEl = BerryConnection(ψT, copy(BlockMPS), copy(setMPS[ind]), copy(setMPS[ind-1]))
        
        
        if abs(BerryEl) < 0.99
            @show abs(BerryEl)
            println("Modulus too small, retry with smallest step")
            flush(stdout)
            return DehnTwist(copy(ψ), N_step + 100, N_cell, topologicalSector, namesave; kwargs...)
        end
        
        BerryFac -= imag(log(BerryEl))
    end
    #=
    println("Calculating Call-Back")
    newψ = newMPS(copy(ψT), topologicalSector, q*N_cell, Gauge)
    if isnothing(W)
        @time W = BerryConnection(ψT, newψ, copy(setMPS[1]), copy(setMPS[Nτ-1]))
    end
    @show imag(log(W)) 
    BerryFac -= imag(log(W))
    =#
    BerryFac /= π 
 
    return BerryFac
end


function BerryConnection(ψ::MPS, BlockMPS::ITensor, G1::ITensor, G2::ITensor)
    replaceind!(G2, inds(G2)[1], inds(ψ[1])[1])
    replaceind!(G1, inds(G1)[1], inds(ψ[1])[1])
    
    ψ_up = copy(ψ)
    ψ_dw = copy(ψ)

    newUp = ψ_up[1]*G2
    newDw = prime(dag(ψ_dw[1]*G1), 1, "Link")

    ContractedTorus = BlockMPS*newDw*newUp
    return ContractedTorus[]
end

function BerryConnection(ψ1::MPS, ψ2::MPS, G1::ITensor, G2::ITensor)
    replaceind!(G2, inds(G2)[1], inds(ψ2[1])[1])
    replaceind!(G1, inds(G1)[1], inds(ψ1[1])[1])
    
    ψ_up = copy(ψ2)
    ψ_dw = copy(ψ1)
    BlockMPSN = contractTorus(ψ_dw, ψ_up)


    newUp = ψ_up[1]*G2
    newDw = prime(dag(copy(ψ_dw)[1]*G1), 1, "Link")

    ContractedTorus = BlockMPSN*newDw*newUp
    return ContractedTorus[]
end


function contractTorus(psi::MPS)
    return contractTorus(psi, psi)
end

function contractTorus(psi2::MPS, psi::MPS)
    ψ = copy(psi)
    ψdag = copy(prime(dag(psi2), 1, "Link"))
    println("Starting contration of the torus")
    flush(stdout)
    O = ψ[2] * ψdag[2]
    for j in eachindex(ψ)[3:end]
        println("done $(j-2)/$(length(eachindex(ψ)[3:end]))")
        flush(stdout)
        O = O*(ψdag[j] * ψ[j])
    end
    
    return O
end

function BackProp(G, top)
    d = inds(G)[1]

    res = 0
    for qn in d.space
        for y in qn[2]
        res += (qn[1][2].val - top)/3
        end
    end
    @show mod2pi(res)
end



function ITensors.op(::OpName"Proj", ::SiteType"FermionK"; y=1)
    M = complex(zeros(2,2))
    M[1,1] = 1.
    M[2,2] = exp(-im*y)

    return M
end

function newMPS(psi::MPS, topologicalShift, Nsites, shift=0)
    
    Pos = 1
    if topologicalShift == 0
        Pos = 2
    elseif topologicalShift == -1
        Pos = 0
    end

    for i in 1:Nsites
        mod(i, 3) != Pos && continue

        gamma = pi*i^2/Nsites + shift
        O = op("Proj", inds(psi[i])[2], y=gamma)
        newP = psi[i]*O
        psi[i] = noprime(newP)
    end
    return psi
end

