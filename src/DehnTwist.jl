using Revise
using ITensors
using ITensorInfiniteMPS

include("InfiniteCylinder.jl")


function TwistOperator(range::UnitRange{Int64}, ψ::InfiniteCanonicalMPS, τ::Float64, topologicalShift::Int64, q::Int64, Φx::Float64, cut; Ne_unitCell::Int64=1, Φy::Float64, kwargs...)
    
    input_inds = inds(ψ.C[0])[1]
    
    diag =[]
    for qn in input_inds.space
        N = qn[1][1].val/3
        K = (qn[1][2].val-topologicalShift)/3 - N
        expFactor = exp(im*π*(nsites(ψ)*Ne_unitCell - 1)*N)*exp(-2π*im*τ*K+im*Φx*N*τ + im*Φy*N)
        Gp = expFactor*ones(qn[2])
        append!(diag, Gp)
    end
   
    output_inds = replacetags(copy(input_inds), "c=0" => "c=$(div(last(range), nsites(ψ)))")

    G = diagm(collect(Iterators.flatten(diag)))
    TwistOperator = ITensor(G, dag(input_inds), output_inds)

    return TwistOperator
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

function DehnTwist(ψ::InfiniteCanonicalMPS, N_cell::Int64, topologicalShift::Int64, namesave::String, L::Float64, cut::Int64; Φx::Float64, q::Int64, kwargs...)
    BerryFac = 0.
    Nτ = 200


    rangeT = (cut+1):(q*N_cell+cut)
    ψT = ψ.AR[rangeT]


    BlockMPS = 0
    if isfile(namesave)
        BlockMPS = load(namesave, "block")
    else
        @time BlockMPS = contractTorus(ψT)
        save(namesave, "block", BlockMPS)
    end

    setMPS = [TwistOperator(rangeT, ψ, 0.0, topologicalShift, q, Φx, cut; kwargs...), TwistOperator(rangeT, ψ, 1., topologicalShift, q, Φx, cut; kwargs...)]

    O = ((Φx/2π)^2 - Φx/2π + 1/6)/(2*3)
    #W = BerryConnection(BlockMPS, copy(setMPS[2]), copy(setMPS[1]))
    # C_mean = test(ψ)
    #H = Expect_Energy_UnitCell(ψ, topologicalShift, Φx, q, C_mean)
    # @show imag(log(W))


    BerryFac =  K(ψ, topologicalShift, Φx, q) - 1/(24*3)
    Berry = BerryFac
    return Berry
   
end

function K(ψ, topologicalShift, Φx, q)

   
    Hloc = 0
    i=0
    psi = ψ.C[i]
    ind = inds(psi)[1]
    _,S,_ = svd(psi, ind)

    n=1
    temp = inds(S)[1]
    for qn in temp.space
        for y in 1:qn[2] 
            KN = qn[1][2].val
            NN = qn[1][1].val
            
            Kloc = (KN -topologicalShift+0.5*NN)/q


            Hloc += S[n,n]^2*(-Kloc)
            
            n += 1
        end
    end
    
    return Hloc
end



function Expect_Energy_UnitCell(ψ, topologicalShift, Φx, q, C_mean)

    H = 0

    for i in 0:2
        Hloc = 0
        
        psi = ψ.C[i]
        ind = inds(psi)[1]
        _,S,_ = svd(psi, ind)

        n=1
        temp = inds(S)[1]
        for qn in temp.space
            for y in 1:qn[2] 
                KN = qn[1][2].val
                NN = qn[1][1].val
                
                Kloc = (KN -topologicalShift +(-1+Φx/2π)*NN)/q
                Nloc = NN/q


                Hloc += S[n,n]^2*(Kloc - (i-1/2 + Φx/2π)*(Nloc-C_mean))
                
                n += 1
            end
        end
        H += Hloc
    end
    H /= 3
    return H
end

function test(ψ)

    H = 0

    for i in 0:2
        Hloc = 0
        
        psi = ψ.C[i]
        ind = inds(psi)[2]
        _,S,_ = svd(psi, ind)

        n=1
        temp = inds(S)[1]
        for qn in temp.space
            for y in 1:qn[2] 
                NN = qn[1][1].val
                
                Nloc = NN/3


                Hloc += S[n,n]^2*(Nloc)
                
                n += 1
            end
        end
        H += Hloc
    end
    H /= 3
    println("<<C>> = $H")
    return H
end




#=
function ITensors.op(::OpName"Proj", ::SiteType"FermionK"; y=0)
    M = complex(zeros(2,2))
    M[1,1] = 1.
    M[2,2] = exp(+im*y)

    return M
end

function Energy(ψ::InfiniteCanonicalMPS,topologicalShift, Φx)

    res = 0
    
    for i in 0:2
        t = 0
        ψS = ψ.C[i]
        ind = inds(ψS)[1]

        _, S, _ = svd(ψS, ind)
        temp = inds(S)[1]
        n=1
        
        for qn in temp.space
            for y in 1:qn[2]
            N = qn[1][1].val/3
            K = (qn[1][2].val - topologicalShift + (Φx/2π)*qn[1][1].val)/3
            
            
            t += S[n,n]^2*(K - (i+Φx/(2*pi))*N)
            n+=1
            end
        end
        res += t
    end
    res /= 3


    return res
end

function ITensors.op(::OpName"Nn", ::SiteType"FermionK")
    M = complex(zeros(2,2))
    M[1,1] = 0.
    M[2,2] = 1.

    ret
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

function DehnTwist(ψ::InfiniteCanonicalMPS, Nτ::Int64, N_cell::Int64, topologicalShift::Int64, namesave::String, W=nothing; q::Int64, kwargs...)
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
        setMPS[ind]  =  TwistOperator(rangeT, ψ, τ, topologicalShift, q; kwargs...)
        #continue
        ind == 1 && continue
        continue
        # ind == Nτ && continue
        #BerryEl = BerryConnection(BlockMPS,setMPS[ind], setMPS[ind-1])
        
        if abs(BerryEl) < 0.99
            @show abs(BerryEl)
            println("Modulus too small, retry with smallest step")
            flush(stdout)
            return DehnTwist(copy(ψ), Nτ + 100, N_cell, topologicalShift, namesave; q, kwargs...)
        end
        
        BerryFac -= imag(log(BerryEl))
    end
    H =  Energy(ψ, topologicalShift, 0)
    #newψ = newMPS(copy(ψT), q*N_cell)
    #@show BerryConnection(ψT, newψ, copy(setMPS[Nτ]), copy(setMPS[1]))
    println("Calculating Call-Back")
    #newψ = exp*copy(ψT)
    
    W = BerryConnection(BlockMPS, copy(setMPS[Nτ]), copy(setMPS[1]))
    BerryFac = W + 2π*(-H + (1/6)/(2*q))
    @show BerryFac
    BerryFac /= π 
 
    return BerryFac, W
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
=#