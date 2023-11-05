using Revise
using ITensors
using ITensorInfiniteMPS
using Plots

include("InfiniteCylinder.jl");

#=
function mainLaugh(Ly)
   
    
	model_params = (Ly=Ly, Vs=[1.])
    maxIter = 2
	save_every = 2
    

    setχ = 200:200

    saving_prefix = "DMRG\\Data\\LaughlinInfinite\\"

    for 🥵 in setχ
        dmrgStruc = Laughlin_struct(Ly, [1.])
        println("Starting 🥵 = $🥵")
        all_ener = Float64[]
        all_err = Float64[]
        all_entr = Float64[]
        #for alpha in alphas
        for x in 1:maxIter÷save_every
            ener, err, entr = idmrg(dmrgStruc; mixer = true, alpha = 1e-6, maxdim = 🥵, nb_iterations = save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-12, output_level = 1)
            ener, err, entr = idmrg(dmrgStruc; mixer = true, alpha = 0., maxdim = 🥵, nb_iterations = save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-12, output_level = 1)
            append!(all_ener, ener); append!(all_err, err); append!(all_entr, entr)
            save(saving_prefix*"Ly_$(Ly)_chi-$(🥵).jld2", "dmrgStruc", dmrgStruc, "energies", all_ener, "errors", all_err, "entropies", all_entr, "params", model_params)
            length(all_ener)<10 && continue
            save_every*x < 6 && continue
            if std(all_ener[end-5:end]) < ener_tol && std(all_entr[end-5:end])< ent_tol
                println("Finishing early")
                break
            end
        end
    end

end;
=#
# mainLaugh(8.)

######################################
######################################

function mainPfaff(Ly)

    #dmrgstruct = Pfaff_struct(8., [0., 0., 1.])
    dmrgStruct =  FQHE_idmrg([2, 1, 2, 1], Ly, [1.], [0., 0., 1.], 0.)
	model_params = (Ly=Ly, Vs=[0., 0., 1.])
    maxIter = 4
	save_every = 2
    

    # setχ = 200:50:200
    setχ = 300:300
    saving_prefix = "DMRG\\Data\\PfaffianInfinite\\"

    for χ in setχ
        
        println("Starting χ = $χ")
        all_ener = Float64[]
        all_err = Float64[]
        all_entr = Float64[]
        #for alpha in alphas
        for x in 1:maxIter÷save_every

            ener, err, entr = idmrg(dmrgStruct; mixer = true, alpha = 1e-6, maxdim = χ, nb_iterations = save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-12, output_level = 1)
            ener, err, entr = idmrg(dmrgStruct; mixer = true, alpha = 0, maxdim = χ, nb_iterations = save_every, measure_entropies = true, ener_tol = 1e-14, cutoff = 1e-12, output_level = 1)
            
            append!(all_ener, ener); append!(all_err, err); append!(all_entr, entr)
            
            save(saving_prefix*"Ly_$(Ly)_chi-$(χ).jld2", "dmrgStruc", dmrgStruct, "energies", all_ener, "errors", all_err, "entropies", all_entr, "params", model_params)
            length(all_ener)<10 && continue
            save_every*x < 6 && continue
            if std(all_ener[end-5:end]) < ener_tol && std(all_entr[end-5:end])< ent_tol
                println("Finishing early")
                break
            end
        end
    end
end



mainPfaff(8.)