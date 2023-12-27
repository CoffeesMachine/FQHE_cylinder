using Revise
using ITensors
using ITensorInfiniteMPS

using FileIO
using JLD2
using KrylovKit

function replaceTranslator(ψ1::InfiniteMPS, translator)
    ψ = deepcopy(ψ1)
    return InfiniteMPS(CelledVector(ψ.data.data, translator), ψ.llim, ψ.rlim, ψ.reverse)
end

function replaceTranslator(ψ1::InfiniteCanonicalMPS, translator)
    ψ = deepcopy(ψ1)
    return InfiniteCanonicalMPS(replaceTranslator(ψ.AL, translator), replaceTranslator(ψ.C, translator), replaceTranslator(ψ.AR, translator))
end



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


function plotFidelity(setPsi, setTheta, chi, Ly; savefig=true)
    setλ = []
    setθ = [0.5*(setTheta[i+1]+setTheta[i]) for i in 1:(length(setTheta)-1)]

    for j in 1:(length(idmrgStruct)-1)
        
        ψ1 = setPsi[j]
        ψ2 = setPsi[j+1]

        ψ1 = replaceTranslator(ψ1, fermion_momentum_translater_four)
        ψ2 = replaceTranslator(ψ2, fermion_momentum_translater_four)
        replace_siteinds!(ψ2, siteinds(ψ1))

        T = TransferMatrix(ψ1.AL, ψ2.AL)
    
        
        vtest = randomITensor(dag(input_inds(T)))
        λ = []

        if norm(vtest) < 1e-10
            println("No 0 sector")
            λ = [0]
        else

            v = translatecell(translator(ψ1), vtest , -1)

        
            λ, _, _ = eigsolve(T, v, 1, :LM; tol=1e-10)
        end
        append!(setλ, abs(λ[1]))
    
    end

    return setθ, setλ
end