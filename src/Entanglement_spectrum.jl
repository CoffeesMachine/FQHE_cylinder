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



function plot_entanglement_spectrum(setL::Array{Float64}, rp::Vector{Int64})
  for i in 1:4 
    l = @layout [a b ; d e]
    p = []
    for Ly in setL[2:5]
      name = "DMRG/Data/PfaffianInfinite/rp$(RootPattern_to_string(rp))_chiMax512_Ly$(Ly)_theta0.0_maxiters100_chi512_alpha0.0.jld2"
      psi = load(name, "dmrgStruct")
      D = compute_entanglement_spectrum(psi.ψ)

    
      
      setK = collect(keys(D[i]))
      newK = []
      
      for k in setK 
        append!(newK, k[1])
      end
      setK = unique(newK)
      N=0
      if i == 1
        N = 2
      elseif i == 2
        N = 4
      elseif i==3 
        N = 2
      end 

      E = Sector_Entanglement_Spectrum(D[i], N)

      push!(p, scatter(E["K"], E["Eta"], marker=:hline, markersize=5, linewidth=4, color="red", title="Ly=$Ly"))
    end
    fig = plot(p...,  layout = l, title="i=$i")
    display(fig)
  end

  # title!("Entanglement spectrum for Ly=$(round(Ly, digits=3)) and χ=$(512)", titlefont = font(8,"Computer Modern"))
end
