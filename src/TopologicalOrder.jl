using Revise
using ITensors
using ITensorInfiniteMPS

using KrylovKit
using FileIO
using JLD2

include("InfiniteCylinder.jl")


function EigenValueTransferMatrix(ψ::InfiniteMPS, N::Int64)

    T =  TransferMatrix(ψ, ψ)
    
    vtry = randomITensor(dag(input_inds(T)))

       
    v = translatecell(translator(ψ), vtry , -1)
    λ, _, _ = KrylovKit.eigsolve(T, v, N, :LM; tol=1e-10)
            

    return λ[1:N]
end

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
        λR, _, _ = eigsolve(T, v, 2, :LM; tol=1e-10)
        T =  transpose(T)
    
        vtry = randomITensor(dag(input_inds(T)))

        return abs(λR[returnEig])
    end
end

function Fidelity(setPsi::Vector{InfiniteCanonicalMPS}, setTheta)
    setλ = []
    setθ = [0.5*(setTheta[i+1]+setTheta[i]) for i in 1:(length(setTheta)-1)]

    for j in 1:(length(setPsi)-1)
        
        ψ1 = setPsi[j]
        ψ2 = setPsi[j+1]

        append!(setλ, EigenValueTransferMatrix(ψ1, ψ2))
    
    end

    return setθ, setλ
end


function OverlapPF(setPsi::Vector{InfiniteCanonicalMPS}, psiPf::InfiniteCanonicalMPS)
    setλ = []

    ψPf = copy(psiPf)
    for j in 1:(length(setPsi))
        
        ψ = copy(setPsi[j])

        append!(setλ, EigenValueTransferMatrix(ψ, ψPf))
    end

    return setλ
end

###########################################
###########################################
###########################################
###########################################


function DehnTwist(ψ::InfiniteCanonicalMPS, topologicalShift::Int64; i=0)

    BerryFac =  K(ψ, topologicalShift, 4; i=i) + 1/(2*24)
    return BerryFac
   
end

function K(ψ, topologicalShift, q; i=0, Φx=0)

    Ktot = 0
    psi = ψ.C[i]
    ind = inds(psi)[1]
    _,S,_ = svd(psi, ind)

    n=1
    temp = inds(S)[1]
    for qn in temp.space
        for y in 1:qn[2] 
            KN = qn[1][2].val
            NN = qn[1][1].val
            fluxShift = 0.

            Kloc = (KN - topologicalShift + fluxShift)/q


            Ktot += S[n,n]^2*(-Kloc)
            
            n += 1
        end
    end
    
    return Ktot
end

###########################################
###########################################
###########################################
###########################################

function Charge(ψ::InfiniteCanonicalMPS)
   
    Q =   AvCell(ψ)- Av(ψ, 0)

    return Q
end

function Av(ψ::InfiniteCanonicalMPS, i)

    Ntot = 0
    psi = ψ.C[i]
    ind = inds(psi)[1]
    _,S,_ = svd(psi, ind)

    n=1
    temp = inds(S)[1]
    for qn in temp.space
        for y in 1:qn[2] 
            NN = qn[1][1].val/nsites(ψ)
            Ntot += S[n,n]^2*(NN)
            n += 1
        end
    end
    
    return Ntot
end 

function AvCell(ψ::InfiniteCanonicalMPS)

    ret = 0 
    for i in 1:nsites(ψ)
        ret += Av(ψ, i)
    end

    return ret/nsites(ψ)
end


###########################################
###########################################
###########################################
###########################################


function StructureFactor(ψ1::InfiniteCanonicalMPS, Ly::Float64, M_max::Int64, theta::Float64, tagRP::String, chi::Int64;kmin=0.01, kmax=4.5, nstep = 20)
    setK = LinRange(kmin, kmax, nstep)
    @show M_max
    savename = "Data/Analysis/Infinite_CorrelationMatrix_L$(Ly)_Nphi$(M_max)_theta$(theta)_rp$(tagRP)_chi$(chi).jld2"
    CorrMatrix = 0
    ExpectN = 0
    
    if isfile(savename)
        CorrMatrix, ExpectN = load(savename, "correlationMatrix", "expect")
    else
        # psi = finite_mps(ψ1, 1:M_max)
        CorrMatrix = correlation_matrix(ψ1, "N", "N"; sites= -M_max+1:M_max,)
        ExpectN= expect(ψ1, "N";sites=-M_max+1:M_max+1)
        save(savename, "correlationMatrix", CorrMatrix, "expect", ExpectN)
    end
  
    nameStruct = "Data/Analysis/StructureFactor_L$(Ly)_Nphi$(M_max)_theta$(theta)_rp$(tagRP)_chi$(chi)_kmin$(kmin)_kmax$(kmax)_nstep$(nstep).jld2"

    StructFac = []

    if isfile(nameStruct)
        StructFac = load(nameStruct, "strc")
    else 
        StructFac = [StructureFactorEl(CorrMatrix, convert(Vector{Float64}, ExpectN), k, 2pi/Ly, -M_max+1:M_max) for k in setK]
        save(nameStruct, "strc", StructFac)
    end 
    
    return StructFac, setK
end


function StructureFactorEl(CorrelationMatrix::Matrix{Float64}, ExpectedOcc::Vector{Float64}, k::Float64, AspectRatio::Float64, range)
    structFac2 = 0

    for i in eachindex(range)
        for j in eachindex(range)

            Corr = CorrelationMatrix[i, j] - ExpectedOcc[i]*ExpectedOcc[j]
            n1 = range[i]
            n2 = range[j]
            structFac2 += exp(-im*k*(n1-n2)*AspectRatio)*Corr
        end   
    end 

    return abs(structFac2)/(length(range))
end

###########################################
###########################################
###########################################
###########################################

CorrelationLength(ψ::InfiniteCanonicalMPS) = EigenValueTransferMatrix(ψ, ψ; returnEig=2)

function CorrelationLength(setψ::Vector{InfiniteCanonicalMPS})
    corrlength = []

    for ψ in setψ
        xi = CorrelationLength(ψ)
        append!(corrlength, -nsites(ψ)/log(abs(xi)))
    end
    

    return corrlength
end


###########################################
###########################################
###########################################
###########################################
function orbital(x, y, n, gamma)
    Ox = exp(im*gamma*n*x)
    Oy = exp(-(y-gamma*n)^2/2)
    return Ox*Oy

end

function pairCorrelation(psi::MPS, Nphi::Int64, thetaL::Float64, CorrMatrix, ExpectN)

    s = siteinds(psi)
    CorrArray = zeros(Nphi, Nphi, Nphi, Nphi)
    for i in 1:Nphi
        for j in 1:Nphi
            i==j && continue
            for k in 1:Nphi
                for l in 1:Nphi
                    k == l && continue
                    if i != l && i != k
                        continue
                    end
                    if j != l && j != k
                        continue
                    end 
                 
                    CorrArray[i,j,k,l] =  i==l ? CorrMatrix[i,j] : -CorrMatrix[i,j]
                    
                end
            end
        end
    end
    
    save("Data/Analysis/Dim4_CorrelationMatrix_L$(L)_Nphi$(Nphi)_theta$(thetaL)_rp$(RootPattern_to_string(setRP[1])).jld2", "corrTensor", CorrArray)

    return CorrArray
end


function pairCorrelation(Psi, L, thetaL; xmax=20, ymax=20., meshsize=(250, 250))
    Nphi = 50
    Ly = round(Int64, 2pi*Nphi/L)


    (X2, Y2) = (Ly/2,Ly/2)
    xgrid = LinRange(X2-xmax, xmax + X2, meshsize[1])
    ygrid = LinRange(-ymax + Y2, ymax + Y2, meshsize[2])

    #set first electron at the origin 
    nameSave = "Data/Analysis/Dim4_CorrelationMatrix_L$(L)_Nphi$(Nphi)_theta$(thetaL)_rp$(RootPattern_to_string(setRP[1])).jld2"
    nameCorrMatrix = "Data/Analysis/CorrelationMatrix_L$(L)_Nphi$(Nphi)_theta$(thetaL)_rp$(RootPattern_to_string(setRP[1])).jld2"
    ψ = finite_mps(copy(Psi),  1:Nphi)
    corrMatrix = 0
    expN = 0
    if isfile(nameCorrMatrix)
        corrMatrix, expN = load(nameCorrMatrix, "correlationMatrix", "expect")
    else
        corrMatrix = correlation_matrix(copy(ψ), "N", "N";sites= 2:Nphi+1)
        expN = expect(ψ, "N"; sites= 2:Nphi+1)
        save(nameCorrMatrix, "correlationMatrix", corrMatrix, "expect", expN)
    end


    PairCorrArray = Array{Float64}(undef, Nphi, Nphi, Nphi, Nphi)
    if isfile(nameSave)
        PairCorrArray = load(nameSave, "corrTensor")
    else
        PairCorrArray = pairCorrelation(ψ, Nphi, thetaL, corrMatrix, expN)
    end

    PairMatrix = Array{Float64}(undef, meshsize)

    #run over the mesh
    for (indx, X1) in enumerate(xgrid)
        for (indy, Y1) in enumerate(ygrid)
            PairMatrix[indx, indy] = real(PairCorrelationElement((X1, X2), (Y1, Y2), 2pi/L, PairCorrArray))
        end
    end

    return xgrid, ygrid, PairMatrix
end 

function PairCorrelationElement(X::Tuple{Float64, Float64}, Y::Tuple{Float64, Float64}, aspectRatio::Float64, CorrArray)

    corr = 0
    Nphi = size(CorrArray)[1]
    for i in 1:Nphi
        for j in 1:Nphi
            i==j && continue
            for k in 1:Nphi
                for l in 1:Nphi
                    k == l && continue
                    if (i != k && i !=l) || (j!= k && j!=l)
                        continue
                    end   
                    FrontFac = conj(orbital(X[1], Y[1], i,aspectRatio))*conj(orbital(X[2], Y[2], j, aspectRatio))*orbital(X[1], Y[1], l, aspectRatio)*orbital(X[2], Y[2], k, aspectRatio)

                    corr += FrontFac*CorrArray[i, j, k, l]/(L^2*pi)
                    
                end
            end
        end
    end
    

    return corr
end


###########################################
###########################################
###########################################
###########################################

function expectationSingleMPO(psi::InfiniteCanonicalMPS, Ly::Float64, θ::Float64, χ::Int64, rp::Vector{Int64}, type::String, range::AbstractRange)
    
    dir = "Data/IMPO/"
    intType = type=="two" ? "Interaction" : "Int"
    nameMPO = "Ly$(Ly)_$(intType)$(type)_RootPattern$(RootPattern_to_string(rp)).jld2"


    H, L, R = load(dir*nameMPO, "H", "L", "R")
    s = [inds(psi[i])[2] for i=1:nsites(psi)]

    for i in 1:nsites(psi)

    Hloc = H[i] 
    @show inds(Hloc)
    @show inds(L)
    @show inds(R)
    error("")
    Hloc = Hloc*L*R
    end





end 