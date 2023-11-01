using FileIO
using JLD2
using Plots
include("MatrixElement.jl")




function loadDict(N, L, Vs, type::String)

    L = Int64(L)
    localisation = "C:\\Users\\basil\\Documents\\EPFL\\Master\\MasterProject\\Code\\DMRG\\Hamiltonians\\"
    saving_ = "Test_coeffs_Pf.jld2"

    DictReturn = Dict()
    jldopen(localisation*saving_, "r") do file
        DictReturn = file["coeffs"]
    end

    return DictReturn
end;


function compareDict(N, L, Vs, type="two")

    pro = loadDict(N, L, Vs, type)
    homeMade = Generate_Elements(N, L, Vs, "three")
    #homeMade = optimize_coefficients(homeMade; prec=1e-8, PHsym = false)
    setratio = []
    for key in keys(pro)
        ratio = pro[key]/homeMade[key]
        #=
        if !haskey(homeMade, key)
            println("the key $key does not apprear in homemade version")
        else
            
            println("########")
            println("ratio : ", pro[key]/homeMade[key])
            push!(setratio, pro[key]/homeMade[key])
        
            
        end
        =#

        if abs(1. - ratio) > 1e-2
            if pro[key] < 1e-6
                continue
            end
            
            println("###############")
            @show ratio 
            @show pro[key]
            @show homeMade[key]
        end
    end
    @show length(keys(homeMade))
    @show length(keys(pro))
    #display(histogram(setratio, xlims=(0.0, 0.25)))
    #=
    for key in keys(homeMade)

        if !haskey(pro, key)
            println("the key $key does not apprear in pro version")
        else
            @show pro[key]/homeMade[key]
        end
    end
    =#
end