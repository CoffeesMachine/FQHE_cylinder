using Revise
using ITensors
using ITensorInfiniteMPS

using LaTeXStrings
using Plots

include("InfiniteCylinder.jl")

function compute_entanglement_spectrum(C::ITensor, ind::Index{Vector{Pair{QN, Int64}}}; prec = 1e-12, nb_qn = 2)
    entanglement_spectrum = Dict()
    n=1
    _, S, _ = svd(C, ind)
    temp = inds(S)[1]
    for xind in temp.space
        local_ent = Float64[]
        for y in 1:xind[2]
          if S[n, n]^2 > prec
            append!(local_ent, S[n, n]^2)
          end
          n+=1
        end
        if length(local_ent)!=0
          entanglement_spectrum[( val.(xind[1])[1:min(length(xind[1]), nb_qn)] )] = local_ent
        end
    end
  return entanglement_spectrum
end


function compute_entanglement_spectrum(ψ::InfiniteCanonicalMPS; kwargs...)
    entanglement_spectrum = Dict()
    for j in 1:nsites(ψ)
      entanglement_spectrum[j] = compute_entanglement_spectrum(ψ.C[j], only(commoninds(ψ.C[j], ψ.AL[j])), kwargs...)
    end
  return entanglement_spectrum
end



function Sector_Entanglement_Spectrum(Spec::Dict, N::Int64)
    K_= []
    eta_ = []
    for (k, v) in Spec
        if k[1] == N
            for value in v
                append!(K_, k[2])
                append!(eta_, -log(value))
            end
        end
    end

    DictRet = Dict()
    DictRet["K"] = K_
    DictRet["Eta"] = eta_

    return DictRet
end

function plot_entanglement_spectrum(EntSpec::Dict, N::Int64, Ly::Float64, chi::Int64, RootPattern::Vector{Int64})
    EntSpec = Sector_Entanglement_Spectrum(EntSpec, N)
    
    
    fig = plot()
    scatter!(fig, EntSpec["K"], EntSpec["Eta"], marker=:hline, markersize=10, linewidth=4, color="red", label = "RP : $(RootPattern_to_string(RootPattern))")
    xlabel!(L"K")
    ylabel!(L"$\xi$")
    title!("Entanglement spectrum in the N=$(N) sector for Ly=$(round(Ly, digits=3)) and χ=$(chi)", titlefont = font(8,"Computer Modern"))
    display(fig)
end
