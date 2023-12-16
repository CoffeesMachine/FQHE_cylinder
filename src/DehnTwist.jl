using Revise
using ITensors
using ITensorInfiniteMPS

include("InfiniteCylinder.jl")


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


function DehnTwist(ψ::InfiniteCanonicalMPS, Nτ::Int64, N_cell::Int64, topologicalSector::Int64, namesave::String, W=nothing; q::Int64, kwargs...)
    BerryFac = 0.

    setMPS = Array{ITensor}(undef, Nτ)
    setTau = LinRange(0., 1., Nτ)

    Gauge = pi*N_cell/4  

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
        mod(ind-1, 50) == 0 && println("Calculating Dehn twist for τ = $τ")
        setMPS[ind]  =  TwistOperator(rangeT, ψ, τ, topologicalSector, q; kwargs...)
        #continue
        ind == 1 && continue
        ind == Nτ && continue
        BerryEl = BerryConnection(BlockMPS,setMPS[ind], setMPS[ind-1])
        
        if abs(BerryEl) < 0.99
            @show abs(BerryEl)
            println("Modulus too small, retry with smallest step")
            flush(stdout)
            return DehnTwist(copy(ψ), N_step + 100, N_cell, topologicalSector, namesave; kwargs...)
        end
        
        BerryFac -= imag(log(BerryEl))
    end

    #newψ = newMPS(copy(ψT), q*N_cell)
    #@show BerryConnection(ψT, newψ, copy(setMPS[Nτ]), copy(setMPS[1]))
    println("Calculating Call-Back")
    #newψ = exp*copy(ψT)
    
    W = BerryConnection(BlockMPS, copy(setMPS[Nτ]), copy(setMPS[1]))
    BerryFac += imag(log(W))
    @show imag(log(W))
    BerryFac /= π 
 
    return BerryFac, W
end


function BerryConnection(BlockMPS::ITensor, G1::ITensor, G2::ITensor)
    
    B = copy(BlockMPS)

    G_up = copy(G2)
    G_dw = copy(G1)
    G_dw = prime(dag(G_dw), 1, "Link")

    replaceind!(G_up, inds(G_up)[1], inds(B)[1])
    replaceind!(G_dw, inds(G_dw)[1], inds(B)[2])

   

    ContractedTorus = B*G_dw*G_up
    return ContractedTorus[]
end


function BerryConnection(ψ1::MPS, ψ2::MPS, G1::ITensor, G2::ITensor)
    
    ψ_up = copy(ψ2)
    ψ_dw = copy(ψ1)
    G_up = copy(G2)
    G_dw = copy(G1)

    BlockMPSN = contractTorus(ψ_dw, ψ_up)

    G_dw = prime(dag(G_dw), 1, "Link")

    replaceind!(G_up, inds(G_up)[1], inds(BlockMPSN)[1])
    replaceind!(G_dw, inds(G_dw)[1], inds(BlockMPSN)[2])

    ContractedTorus = BlockMPSN*G_up*G_dw
    @show ContractedTorus
    return ContractedTorus[]
end


function contractTorus(psi::MPS)
    return contractTorus(psi, psi)
end

function contractTorus(psi2::MPS, psi::MPS)
    ψ = copy(psi)
    ψdag = copy(prime(dag(psi2), 1, "Link"))
    O = ψ[1] * ψdag[1]
    println("done 1/$(length(eachindex(ψ)))")
    flush(stdout)
    for j in eachindex(ψ)[2:end]
        O = O*(ψdag[j] * ψ[j])
        println("done $(j)/$(length(eachindex(ψ)))")
        flush(stdout)
    end
    
    return O
end


function ITensors.op(::OpName"Proj", ::SiteType"FermionK"; y=0)
    M = complex(zeros(2,2))
    M[1,1] = 1.
    M[2,2] = exp(+im*y)

    return M
end

function newMPS(psi::MPS, Nsites, shift=0)    
    for i in 1:Nsites
        gamma = pi*(i-1)^2/Nsites
        O = op("Proj", inds(psi[i])[2]; y=gamma)
        @show O
        newP = psi[i]*O
        psi[i] = noprime(newP)
    end
    return psi
end

