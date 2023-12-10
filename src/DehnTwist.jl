using Revise
using ITensors
using ITensorInfiniteMPS

include("InfiniteCylinder.jl")


function TwistOperator(range, ψ::InfiniteCanonicalMPS, τ::Float64, topologicalShift::Int64, q::Int64; Ne_unitCell::Int64=1, Φx::Float64, Φy::Float64)
    
    input_inds = inds(ψ.AL[1])[1]
    diag =[]
    for qn in input_inds.space
        N = qn[1][1].val/q
        K = (qn[1][2].val - topologicalShift)/q + Φx/2π*N
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


function DehnTwist(ψ::InfiniteCanonicalMPS, Nτ::Int64, N_cell::Int64, topologicalSector::Int64; q::Int64, kwargs...)
    BerryFac = 0.

    setMPS = Array{ITensor}(undef, Nτ)
    setTau = LinRange(0., 1., Nτ)

    Gauge = q*pi*N_cell^2/12
    

    rangeT = 1:q*N_cell
    ψT = ψ.AL[rangeT]
    
    @time BlockMPS = contractTorus(ψT)

    for (ind, τ) in enumerate(setTau)
        mod(ind, 25) == 0 && println("Calculating Dehn twise for τ=$τ")
        setMPS[ind]  =  TwistOperator(rangeT, ψ, τ, topologicalSector, q; kwargs...)
        
        ind == 1 && continue

        BerryEl = BerryConnection(ψT, copy(BlockMPS), copy(setMPS[ind]), copy(setMPS[ind-1]))
        
        
        if abs(BerryEl) < 0.99
            @show abs(BerryEl)
            println("Modulus too small, retry with smallest step")
            flush(stdout)
            return DehnTwist(copy(ψ), N_step + 100, N_cell, topologicalSector; kwargs...)
        end
        
        BerryFac -= imag(log(BerryEl))
    end

    a = BerryConnection(ψT, BlockMPS, setMPS[end], setMPS[1]; gauge=Gauge)

    BerryFac += imag(log(a))
    BerryFac /= π 
 
    return BerryFac
end


function BerryConnection(ψ::MPS, BlockMPS::ITensor, G1::ITensor, G2::ITensor; gauge=1.)
    replaceind!(G2, inds(G2)[1], inds(ψ[1])[1])
    replaceind!(G1, inds(G1)[1], inds(ψ[1])[1])
    
    ψ_up = copy(ψ)
    ψ_dw = gauge*copy(ψ)

    newUp = ψ_up[1]*G2
    newDw = prime(dag(ψ_dw[1]*G1), 1, "Link")

    ContractedTorus = BlockMPS*newDw*newUp
    return ContractedTorus[]
end


function contractTorus(psi::MPS)
    ψ = copy(psi)
    ψdag = copy(prime(dag(psi), 1, "Link"))
    O = ψ[2] * ψdag[2]

    for j in eachindex(ψ)[3:end]
        O = (O*ψdag[j]) * ψ[j]
    end
    
    return O
end

#=
function torus_mps(ψ::InfiniteCanonicalMPS, range::AbstractRange)
    @assert isone(step(range))
    ψ_finite = ψ.AL[range]

    return ψ_finite
end

function normTorus(psi::MPS, BlockMPS::ITensor)

    ψ = copy(psi)

    newψ = ψ[1]*delta(inds(dag(ψ[1]))[1], inds(dag(ψ[end]))[end])
    
    ψ[1] = newψ
    ψdag = prime(dag(copy(ψ)), 1, "Link")


    Scal = BlockMPS*ψ[1]*ψdag[1]
    return Scal[]
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

function ExpectationValue(ψ)

    N = nsites(ψ)
    expect = 0
    for i in 1:N
        localExp = 0.
        _, S, _ = svd(ψ.C[i], inds(ψ.C[i])[1])

        spaceK = inds(S)[1].space
        
        n = 1
        for indK in spaceK
           #=localExp += indK[2]*indK[1][2].val
           size += indK[2]
           =#
           
            for dim in indK[2]
                localExp += S[n,n]^2*(indK[1][2].val)/3
                n += 1
            end
        end
        expect += localExp
    end
    return expect/3

end
=#