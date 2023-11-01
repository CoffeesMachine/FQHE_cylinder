using ITensors
using Plots
using ITensors.HDF5
using FileIO
using JLD2
using Revise

include("MatrixElement.jl")

function patternPfaff(n)
    if mod(n, 4) == 1 || mod(n, 4) == 2
        return "1"
    else
        return "0"
    end
end

function LoadMPO(N_Φ::Int64, Ly::Float64, Vs::Vector{Float64}, prec::Float64, type::String, sites; forceReload=false)
    
    MPO_ = MPO()
    nameSaving = "DMRG\\MPO\\MPO_N_phi($N_Φ)_Lx($Ly)_type($type)_Vs($Vs).jld2"
    flag = isfile(nameSaving)

    if flag && !forceReload
        MPO_, sites = load(nameSaving, "MPO", "sites")
    else
        println("Calculating new MPO")
        H = finite_Cylinder_MPO(N_Φ, Ly, Vs, prec, type)
        MPO_ = MPO(H, sites)
        save(nameSaving, "MPO", MPO_, "sites", sites)
    end

    return MPO_, sites
end;

function LoadMPS(
    Ne::Int64, Ly::Float64, type::String, sites, MPO_::MPO, Chi::Int64, GenFunction::Function; forceReload=false
    )

    ψ = MPS()
    
    DirName = "DMRG\\MPS\\Type_($type)_Ne($Ne)_Ly($Ly).h5"

    if isfile(DirName) && !forceReload
        f = h5open(DirName, "r")
        ψ = read(f,"psi", MPS)
        close(f)
        replace_siteinds!(ψ, sites)
    else

        ψ0 = productMPS(sites, GenFunction)
        _, ψ = dmrg(MPO_, ψ0; nsweeps=10, maxdim = Chi, cutoff=1e-8, noise=1e-4)

        f = h5open(DirName, "w")
        write(f,"psi", ψ)
        close(f)
    end

    return ψ
end;

##########
##########

function Laughlin(Ne::Int64, Ly::Float64, Vs::Vector{Float64})

    N_Φ = 3*(Ne-1)+1

    sites = [Index([QN(("Nf", 0), ("NfMom", 0))=>1, QN(("Nf", 1), ("NfMom", n))=>1], "Fermion,Site,n=$n", dir = ITensors.Out) for n in 1:N_Φ]

    MPO_ = LoadMPO(N_Φ, Ly, Vs, 1e-9, "two", sites, forceReload=false)
    
    ψ = Laughlin(Ne, Ly, MPO_, sites)

    @show inner(ψ', MPO_, ψ)
    return ψ
end;

function Laughlin(Ne::Int64, Ly::Float64, MPO_::MPO, sites)
        
    function pattern(n)
        if mod(n, 3) == 1
            return "1"
        else
            return "0"
        end
    end
    
    ψ = LoadMPS(Ne, Ly, "Laughlin13", sites, MPO_, 250, pattern, forceReload=false)
    return ψ
end

#################################
#################################


function Pfaffian(Ne::Int64, Ly::Float64, Vs::Vector{Float64})
    N_phi = 2*(Ne-1)

    sites = [Index([QN(("Nf", 0), ("NfMom", 0))=>1, QN(("Nf", 1), ("NfMom", n))=>1], "Fermion,Site,n=($n)", dir = ITensors.Out) for n in 1:N_phi]

    prec = 1e-7
    
    MPO_, sites = LoadMPO(N_phi, Ly, Vs, prec, "three", sites, forceReload=false)
    ψ = Pfaffian(Ne, Ly, MPO_, sites)

    @show inner(ψ', MPO_, ψ)

    return ψ, sites
end;

function Pfaffian(Ne::Int64, Ly::Float64, MPO_::MPO, sites)

    ψ = LoadMPS(Ne, Ly, "Pfaffian", sites, MPO_, 250, patternPfaff; forceReload=false)
    return ψ
end;

#################################
#################################

function Entanglement(ψ, cutbound)
    prec = 1e-12
    nb_qn = 2
    n=1
    entanglement_spectrum = Dict()
    ITensors.orthogonalize!(ψ, cutbound)
    _, S, _ = svd(ψ[cutbound], (linkind(ψ, cutbound-1), siteind(ψ,cutbound)))
    temp = inds(S)[1]
    for xind in temp.space
        local_ent = Float64[]
        for y in 1:xind[2]
          if S[n, n]^2 > prec
            append!(local_ent, -2*log(S[n, n]))
          end
          n += 1 
        end
        if length(local_ent)!=0
          entanglement_spectrum[( val.(xind[1])[1:min(length(xind[1]), nb_qn)] )] = local_ent
        end
    end
  return entanglement_spectrum
end


function phase_transition(t, χ::Int64, MPO_2body, MPO_3body, sites, Ly)
    println("Calculating ground state for t=$t")
    Ne = Int64(length(sites)/2+1)
    name = "DMRG/finite_Cylinder_MPS/t$(round(t, digits=8))_Ly$(Ly)_Ne$(Ne).jld2"

    if isfile(name)
        E, ψ = load(name, "Energy", "psi")        
        replace_siteinds!(ψ, sites)
    else
        ψ0 = productMPS(sites, patternPfaff)
        #MPO_Ham = t*deepcopy(MPO_2body) + (1-t)*deepcopy(MPO_3body)
        MPO_Ham = sin(t)*deepcopy(MPO_2body) + cos(t)*deepcopy(MPO_3body)
        E, ψ = dmrg(MPO_Ham, ψ0; nsweeps=10, maxdim = χ, cutoff=1e-8, noise=1e-6)

        save(name, "Energy", E, "psi", ψ)
    end
    return E, ψ
end


function phase_transition_linear_pos(t, χ::Int64, MPO_2body, MPO_3body, sites, Ly)
    println("Calculating ground state for t=$t")
    Ne = Int64(length(sites)/2+1)
    name = "DMRG/finite_Cylinder_MPS/Linear_pos_t$(round(t, digits=8))_Ly$(Ly)_Ne$(Ne).jld2"

    if isfile(name)
        E, ψ = load(name, "Energy", "psi")        
        replace_siteinds!(ψ, sites)
    else
        ψ0 = productMPS(sites, patternPfaff)
        #MPO_Ham = t*deepcopy(MPO_2body) + (1-t)*deepcopy(MPO_3body)
        MPO_Ham = t*deepcopy(MPO_2body) + (1-t)*deepcopy(MPO_3body)
        E, ψ = dmrg(MPO_Ham, ψ0; nsweeps=10, maxdim = χ, cutoff=1e-8, noise=1e-6)

        save(name, "Energy", E, "psi", ψ)
    end
    return E, ψ
end

function fidelity(t1::Float64, t2::Float64, Ly::Float64, sites)

    Ne = Int64(length(sites)/2+1)
    name = "DMRG/finite_Cylinder_MPS/t$(round(t1, digits=8))_Ly$(Ly)_Ne$(Ne).jld2"
    _, ψ1 = load(name, "Energy", "psi")
    
    name = "DMRG/finite_Cylinder_MPS/t$(round(t2, digits=8))_Ly$(Ly)_Ne$(Ne).jld2"
    _, ψ2 = load(name, "Energy", "psi")
    return abs(inner(ψ1', ψ2))
end


#################################
#################################


function plot_fidelity(setT, setFi, Ne, Ly)
    fig = plot()
    scatter!(fig, setT,  setFi, label= "fidelity")
    xlabel!("θ")
    ylabel!("<ψ|ψL>")
    title!("fidelity for Ne=$(Ne) and Ly = $(Ly)")
    vline!([n*pi/4 for n=1:1:8])
    display(fig)
end;

function plot_entanglement(dictE, legends::String)
    setK = []
    setE = []
    #testK = [5, 6, 7, 10, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 18, 19, 20]
    #testE = [3.878012225657725, 11.058311913174627, 15.856778588062138, 19.353915742791635, 21.705104253950854, 0.4831809513910732, 1.1599882769419096, 5.028235370270603, 6.713312555758491, 7.991096694234697, 9.513993605504876, 3.1989047730690507]
    #testE = [3.9898763254837544, 8.021546233752991, 10.156219123830288, 0.5879885113963235, 1.2634469042056207, 2.10736119965245, 6.282318175097273, 7.673290523106472, 8.884266131609342, 9.269950626936993, 14.152596453561536, 11.28119229849738, 13.974196979965907, 16.37924930661924, 17.8125895976097, 14.05536233190668, 18.510679041835644, 20.534684619232078, 20.90652276063041, 23.573093138849075, 26.946916853942806, 30.66489770269142, 3.989876325483755,8.02154623375299,10.156219123830297]

    
    for (k, val) in dictE
        for Ent in val 
            push!(setK, k[2]-k[1])
            push!(setE, Ent)
        end
    end
    fig = plot()
    scatter!(fig, setK, setE, label=legends, xlabel="K", ylabel="ξ")
    
    #scatter!(fig, testK, testE, color=:red, markershape=:diamond)
    display(fig)
end;

function plot_overlap(setT,  setψPf, Ne, Ly)

    fig = plot()
    scatter!(fig, setT,  setψPf, label= "Overlap with Pfaffian 1/2")
    xlabel!("t")
    ylabel!("<ψ|ψL>")
    title!("Overlap for Ne=$(Ne) and Ly = $(Ly)")
    display(fig)
end;

#################################
#################################

function main(Ne::Int64, Lx::Float64, Nmin::Float64, Nmax::Float64, Nsteps::Int64)

    setT = LinRange(Nmin, Nmax, Nsteps+1)
    N_Φ = 2*(Ne-1)
    sites = [Index([QN(("Nf", 0), ("NfMom", 0))=>1, QN(("Nf", 1), ("NfMom", n))=>1], "Fermion,Site,n=$n", dir = ITensors.Out) for n in 1:N_Φ]

    println("Calculating the MPO")
    MPO_2body, sites = LoadMPO(N_Φ, Lx, [1.], 1e-9, "two", sites)
    MPO_3body, sites = LoadMPO(N_Φ, Lx, [0.; 0.; 1.], 1e-9, "three", sites)


    println("Calculating the GS for the Pfaffian 1/2")
    ψPf = Pfaffian(Ne,Lx, deepcopy(MPO_3body), sites)
    
    setOPf = []
    setEPf = []
    setFi = []

    for t in setT
         
        χ = 250
        
        E, ψ = phase_transition(t, χ, MPO_2body, MPO_3body, sites, Lx)

        push!(setEPf, E)
        
    end

    for index in 1:Nsteps
        push!(setFi, fidelity(setT[index], setT[index+1], Lx, sites))
    end

    #plot_overlap(setT, setOPf, Ne, Lx)

    plot_fidelity(setT[1:Nsteps], setFi, Ne, Lx)

    fig2 = plot()
    scatter!(fig2, setT,  setEPf, label ="Energy Pfaffian", xlabel="θ", ylabel = "E")
    title!("Energy for Ne=$(Ne) and Ly = $(Lx)")
    vline!([n*pi/4 for n=1:1:8])
    display(fig2)
    #plot_entanglement(Entanglement(ψPf, Ne-1), "Entanglement for the pfaffian")

end