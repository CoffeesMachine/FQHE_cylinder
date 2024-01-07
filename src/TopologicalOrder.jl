using Revise
using ITensors
using ITensorInfiniteMPS

using KrylovKit
using FileIO
using JLD2


CorrelationLength(ψ::InfiniteCanonicalMPS) = EigenValueTransferMatrix(ψ, ψ; returnEig=2)

function EigenValueTransferMatrix(ψ1in::InfiniteCanonicalMPS, ψ2in::InfiniteCanonicalMPS; returnEig=1)
    ψ1 = copy(ψ1in)
    ψ2 = copy(ψ2in)

    returnEig == 1 && replace_siteinds!(ψ2, siteinds(ψ1))

    T =  TransferMatrix(ψ1.AL, ψ2.AL)
    
    vtry = randomITensor(dag(input_inds(T)))

    if norm(vtry) < 1e-10
        println("No 0 sector")
        return 0
    else
       
        v = translatecell(translator(ψ1), vtry , -1)
        λ, _, _ = eigsolve(T, v, 2, :LM; tol=1e-10)
            

        return λ[returnEig]
    end
end

function Fidelity(setPsi::Vector{InfiniteCanonicalMPS}, setTheta)
    setλ = []
    setθ = [0.5*(setTheta[i+1]+setTheta[i]) for i in 1:(length(setTheta)-1)]

    for j in 1:(length(setPsi)-1)
        
        ψ1 = setPsi[j]
        ψ2 = setPsi[j+1]

        append(setλ, EigenValueTransferMatrix(ψ1, ψ2))
    
    end

    return setθ, setλ
end


function OverlapPF(setPsi::Vector{InfiniteCanonicalMPS}, psiPf::InfiniteCanonicalMPS)
    setλ = []

    ψPf = copy(psiPf)
    for j in 1:(length(setPsi))
        
        ψ = setPsi[j]

        append(setλ, EigenValueTransferMatrix(ψ, ψPf))
    end

    return setλ
end


function StructureFactor(ψ1::InfiniteCanonicalMPS, Ly::Float64, M_max::Int64, theta::Float64, tagRP::String; kmax=4.5, nstep = 20)
    setK = LinRange(0, kmax, nstep)

    ψ = finite_mps(copy(ψ1), 1:M_max)


    savename = "Data/Analysis/CorrelationMatrix_L$(L)_Nphi$(M_max)_theta$(theta)_rp$(tagRP).jld2"
    CorrMatrix = 0
    ExpectN = 0

    if isfile(savename)
        CorrMatrix, ExpectN = load(savename, "correlationMatrix", "expect")
    else

        CorrMatrix = correlation_matrix(ψ, "N", "N"; sites=2:(M_max +1))
        ExpectN= expect(ψ, "N";sites=2:(M_max+1))
        save(savename, "correlationMatrix", CorrMatrix, "expect", ExpectN)
    end
    
    StructFac = [StructureFactor(CorrMatrix, ExpectN, k, 2pi/Ly)/M_max for k in setK]

    return StructFac, setK
end



function StructureFactor(CorrelationMatrix::Matrix{Float64}, ExpectedOcc::Vector{Float64}, k::Float64, AspectRatio::Float64)
    structFac2 = 0
    for n in eachindex(ExpectedOcc)
        for m in eachindex(ExpectedOcc)
            structFac2 += exp(-im*k*AspectRatio*(n-m))*(CorrelationMatrix[n,m] -ExpectedOcc[n]*ExpectedOcc[m])
        end
    end
    return abs(structFac2)
end


function CorrelationLength(setψ::Vector{InfiniteCanonicalMPS})
    corrlength = []

    for ψ in setψ
        xi = CorrelationLength(ψ)
        @show xi
        append!(corrlength, -nsites(ψ)/log(abs(xi)))
    end
    

    return corrlength
end
