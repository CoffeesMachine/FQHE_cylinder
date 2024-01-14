using DelimitedFiles

function writefiles(Ly, tmin, tmax, N; Fourb=false)

    path = "C:/Users/basil/Documents/EPFL/Master/MasterProject/Code/DMRG/PrepareFilesCluster/ClusterFiles/"
    name = "$(round(tmin,digits=2))_$(round(tmax,digits=2))_$(N)_$(Ly)_F$(Fourb).txt"
    L = Ly*ones(N)
    theta = collect(LinRange(tmin, tmax, N))
    if Fourb
        newtheta = []
        count = 0
        for el in theta
            abs(cos(el)) < 0.1 && continue
            append!(newtheta, el)
            count += 1
        end

        theta = newtheta
        L = L[1:count]
    end

    open(path*name, "w") do io
        writedlm(io, [L theta])
    end

    println("Files generated")
end


writefiles(11., -1.5, 1.5, 40; Fourb=false)