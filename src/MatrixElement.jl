using FileIO
using JLD2
using ITensors


function projector_local(n, κ, m::Int64)
    #=
    This function generates the coefficients of the matrix element for the cylinder for two body interaction
    Inputs : 
                n : quantum number that defines the position in the cylinder
                κ : (2π/L) the aspect ratio in the periodic direction (taken to be x here)
                m : relative angular momentum
    Return :
                The coefficient associated to these parameters according to 
                the reference "PHYSICAL REVIEW B 88, 165303 (2013)"
    =#
    return (2/pi)^(0.25)*sqrt(κ)*exp(-(κ*n)^2)*HermitePolynomial(sqrt(2)*κ*n, m)/sqrt(2^m*factorial(m))
end


function projector_local(n1, n2, κ, m::Int64)
    #=
    This function generates the coefficients of the matrix element for the cylinder for three body interaction
    Inputs : 
                n1 : first quantum number linked to momentum
                n2 : second quantum number linked to momentum
                κ : (2π/L) the aspect ratio in the periodic direction (taken to be x here)
                m : relative angular momentum
    Return :
                The coefficient associated to these parameters according to 
                the reference "https://arxiv.org/pdf/1606.05353.pdf"
    =#
    common_factor = κ*(3/pi^2)^0.25*exp(-(n1^2 + n2^2 + n1*n2)*κ*κ)
    if m==0 
        return common_factor
    elseif m == 1
        return 0
    elseif m==2
        return 0
        #return common_factor*0.25*(HermitePolynomial(κ*(n1 - n2)/sqrt(2), 2) 
         #                       + HermitePolynomial(sqrt(3)*κ*(n1 + n2)/sqrt(2), 2))
    elseif m==3
        return common_factor*(
                            sqrt(3)*HermitePolynomial(κ*(n1 - n2)/sqrt(2), 1)
                            * HermitePolynomial(sqrt(3)*κ*(n1 + n2)/sqrt(2), 2)
                            - HermitePolynomial(κ*(n1 - n2)/sqrt(2), 3)/sqrt(3)
                            )/8
    else 
        println("Error, the projector on this momentum subspace is not implemented yet")
        return nothing
    end


end;


function all_projector_two_body(N_Φ, L_x, max_angular)
    #=
    This function genertes all the coefficient away from the "center of mass" j. 
    A state if fully determied by the triplet (j,k,l). Since the index has to be an integer
    the difference betweem j and k (or l) must be an integer. Theefore, half integers values are allowed
    These coefficient are independant from the center of mass

    Inputs : 
                N_Φ : Total number of flux quanta
                L_x : Length of the cylinder in the periodic direction (taken to be x)
                max_angular : maximal anglar momentum allowed
    Output :
                # The dictionary associated to each possible elements

    =#
    coeff = Dict{Float64,Array{Float64,1}}()
    
    # For fermions, only odd values matter
    set_m = 1:1:max_angular

    #moving away from the center of mass
    for ell in 0.5:0.5:(N_Φ/2-0.5)
        coeff[ell] = zeros(Float64, length(set_m))
        for (index_m, m) in enumerate(set_m)
            # add to the dictionary the element
            coeff[ell][index_m] += (projector_local(ell, 2*pi/L_x, m) - projector_local(-ell, 2*pi/L_x, m))/2
          end
    end


    return coeff
end; 

function all_projector_three_bodies(N_Φ, L_x, max_angular)
    coeff = Dict{Tuple{Float64, Float64}, Array{Float64,1}}()
    
    # For fermions, only odd values matter
    set_m = 1:max_angular
    for r1 in -N_Φ:1/3:(N_Φ)
        for r2 in -(N_Φ):1/3:(N_Φ)

            r1 = round(r1, digits=3)
            r2 = round(r2, digits=3)

            coeff[(r1, r2)] = zeros(Float64, length(set_m))

            for (index_m, m) in enumerate(set_m)
                coeff[(r1, r2)][index_m] += (projector_local(r1, r2, 2*pi/L_x, m) - projector_local(r2, r1, 2*pi/L_x, m))/2
            end
        end
    end
    return coeff
end;

function all_projectors(N_Φ, L_x, max_angular, type)
    if type=="two"
        return all_projector_two_body(N_Φ, L_x, max_angular)
    elseif type == "three"
        return all_projector_three_bodies(N_Φ, L_x, max_angular)
    else
        println("Interaction unknown")
        return nothing
    end
end

function projectors_four_body_parallel(Lx, Ly, N_Φ; prec=1e-12)
    prefactor = sqrt(Lx / Ly) / sqrt(N_Φ) * (2 * pi / Ly)^4
    coefficients = [Dict{Tuple{Int64, Int64, Int64}, Float64}() for n in 1:Threads.nthreads()]
    for g = 0:3
        expFactor = zeros(length(collect(-4*N_Φ-g:4:4*N_Φ-1)))
        for (xk, k) in enumerate(-4*N_Φ-g:4:4*N_Φ-1)
            expFactor[xk] = exp(-2*pi^2/Ly^2*(k/4)^2)
        end
        Threads.@threads for (xk, k) in collect(enumerate(-4*N_Φ-g:4:4*N_Φ-1))
            for (xq, q) in enumerate(-4*N_Φ-g:4:4*N_Φ-1)
                if k==q
                    continue
                end
                for (xr, r) in enumerate(-4*N_Φ-g:4:4*N_Φ-1)
                    if k==r || q==r
                        continue
                    end
                    temp = prefactor*W_polynomial(k/4, q/4, r/4, -k/4-q/4-r/4)*expFactor[xk]*expFactor[xq]*
                                    expFactor[xr]*exp(-2*pi^2/Ly^2*(k/4+ q/4 + r/4)^2)
                    coefficients[Threads.threadid()][(k, q, r)]=temp
                end
            end
        end
    end
    return merge(coefficients...)
end

function streamline_four_body(DictE, N_Φ)
    Coeff = Dict()

    for n1 in 0:(N_Φ -4)
        for n2 in (n1 + 1):(N_Φ - 3)
            for n3 in (n2 + 1):(N_Φ -2)
                for n4 in (n3 + 1):(N_Φ - 1)
                    R = n1 + n2 + n3 + n4
                    Coeff[(n1, n2, n3, n4)] = DictE[(R - 4*n1, R - 4*n2, R - 4*n3)]
                end
            end
        end
    end

    return Coeff
end
#################################################################

function two_body_elements(N_Φ, coeff, PseudoPot::Array{Float64})
    #=



    =#

    Coefficients = Dict()

    #loop on all possiples values of the "center of mass"
    for j in 0.5:0.5:N_Φ-1.5
        
        # Loop on the distance from the center of mass. Choose to start on integer 
        # or half-integer depending of the j. 
        # Starting from the right of c†_(j+k)c†_(j-k)c_(j-l)c_(j+l)
        
        for l in mod1(j, 1):1:min(j, N_Φ-1-j)
            antisym_1 = 1                  
            n2 = round(Int64, j + l)
            n1 = round(Int64, j - l)
            # We want a well ordered product of operators, therefore n1 > n2. 
            # If not the case, we swap them and changes the sign because they are fermions
            if n2 > n1
                antisym_1 = -1
                n1, n2 = n2, n1
                @assert n2 < n1
            
            # If both are the same, ignore the case due to Pauli exclusion principle 
            elseif n1 == n2
                continue
            end

            # Loop on the k-variable now and apply same reasonning as for the l case

            for k in mod1(j, 1):1:min(j, N_Φ-1-j)
                antisym_2 = 1
                m1 = round(Int64, j + k)
                m2 = round(Int64, j - k)

                if m1 > m2
                    antisym_2 = -1
                    tmp = m1
                    m1 = m2
                    m2 = tmp
                    @assert m1 < m2
                elseif m1 == m2 
                    continue
                end

                # We check if the dictionnary has already the keys in it 
                if haskey(Coefficients, [m1, m2, n1, n2])
                    Coefficients[[m1, m2, n1, n2]] +=
                    antisym_1*antisym_2*sum([coeff[l][x] * PseudoPot[x] * coeff[k][x] for x in 1:length(PseudoPot)])
                else
                    Coefficients[[m1, m2, n1, n2]] =
                    antisym_1*antisym_2*sum([coeff[l][x] * PseudoPot[x] * coeff[k][x] for x in 1:length(PseudoPot)])
                end
            end
        end
    end

    return Coefficients
end;


function three_body_elements(N_Φ, coeff, PseudoPot::Array{Float64})
    Coefficients = Dict()

    for j in 1/3:1/3:(N_Φ-1-1/3)
        for r3 in -N_Φ+j:1:N_Φ-j 
            for r4 in -N_Φ+j:1:N_Φ-j 
                
                m1 = round(Int64, j+r3)
                m2 = round(Int64, j+r4)
                m3 = round(Int64, j-r4-r3)

                if m1 == m2 || m2 == m3 || m3 == m1 || m3 < 0  || m2 < 0  || m1 < 0 || m1 > N_Φ-1  || m2 > N_Φ-1 || m3 > N_Φ-1 
                    continue
                end
                
                m1, m2, m3, antisym_1 = wellOrderdedSet(m1, m2, m3)
                m1, m2, m3 = m3, m2, m1
                for r1 in -N_Φ+j:1:N_Φ-j 
                    for r2 in -N_Φ+j:1:N_Φ-j 
    
                        n1 = round(Int64, j+r1)
                        n2 = round(Int64, j+r2)
                        n3 = round(Int64, j-r2-r1)
                       
        
                        if n1 == n2 || n2 == n3 || n3 == n1 || n3 < 0 || n2 <0 || n1 < 0 || n1 > N_Φ-1  || n2 > N_Φ-1 || n3 > N_Φ-1 
                            continue
                        end
        
                        n1, n2, n3, antisym_2 = wellOrderdedSet(n1, n2, n3)

                        r1t = round(r1, digits=3)
                        r2t = round(r2, digits=3)
                        r3t = round(r3, digits=3)
                        r4t = round(r4, digits=3)
                        
                        
                        if haskey(Coefficients, [n1, n2, n3, m1, m2, m3])
                            Coefficients[[n1, n2, n3, m1, m2, m3]] += antisym_1*antisym_2*sum([(coeff[(r1t, r2t)][x] * PseudoPot[x] * coeff[(r3t, r4t)][x])/6 for x in 1:length(PseudoPot)])
                        else
                            Coefficients[[n1, n2, n3, m1, m2, m3]] = antisym_1*antisym_2*sum([(coeff[(r1t, r2t)][x] * PseudoPot[x] * coeff[(r3t, r4t)][x])/6 for x in 1:length(PseudoPot)])
                        end
                    end
                end
            end
        end
    end
    return Coefficients
end;

function Neutralize_Backgroud(Coeff::Dict, N_Φ, v)
    Background_coeff = Dict()
    for n in 0:(N_Φ-2)
        for m in n:N_Φ-1
            if haskey(Coeff, [n, m, m, n])
                if !haskey(Background_coeff, [n,n])
                    Background_coeff[[n, n]] = -v*Coeff[[n, m, m, n]]
                else
                    Background_coeff[[n, n]] += -v*Coeff[[n, m, m, n]]
                end
            end
        end
    end

    return Background_coeff
end


function four_body_elements_parallel(N_Φ, Coeff::Dict; prec=1e-12)

    Coeff4B = Dict()

    #Center of mass

    for R in 6:(4*N_Φ-10)
        for n1 in max(0, R-3*N_Φ + 6):min(N_Φ -4, (R - 6)÷4)
            for n2 in max(n1 + 1, R - n1 - 2*N_Φ + 3):min(N_Φ - 3, (R - n1 - 3)÷3)
                for n3 in max(n2 + 1, R - n1 - n2 -N_Φ + 1):min(N_Φ - 2, (R - n1 - n2 -1)÷2)
                    n4 = R - n1 - n2 - n3
                    for m1 in max(0, R - 3*N_Φ + 6):min(N_Φ - 4, (R - 6)÷4)
                        for m2 in max(m1 + 1, R - m1 - 2*N_Φ + 3):min(N_Φ - 3, (R - m1 - 3)÷3)
                            for m3 in max(m2 + 1, R - m1 - m2 - N_Φ + 1):min(N_Φ - 2, (R - m1 - m2 - 1)÷2)
                                m4 = R - m1 - m2 - m3

                                elements = Coeff[(n1, n2, n3, n4)]' * Coeff[(m1, m2, m3, m4)]
                                if abs(elements) > prec
                                    if haskey(Coeff4B, [m1, m2, m3, m4, n4, n3, n2, n1])
                                        Coeff4B[[m1, m2, m3, m4, n4, n3, n2, n1]] += elements
                                    else
                                        Coeff4B[[m1, m2, m3, m4, n4, n3, n2, n1]] = elements
                                
                                    end    
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    return Coeff4B
end

###################################################################

function Generate_Elements(N_Φ, L_x, PseudoPot::Array{Float64}, type="two")
    println("Parameters of the Hamiltonians are")
    @show N_Φ
    @show L_x
    @show PseudoPot


    max_angular = length(PseudoPot)


    coeff = all_projectors(N_Φ, L_x, max_angular, type)


    if type=="two"
        @time Ham = two_body_elements(N_Φ, coeff, PseudoPot)
        return Ham
    elseif type == "three"
        println("Creating The Hamiltonian for 3 body interaction")
        @time Ham = three_body_elements(N_Φ, coeff, PseudoPot)
        return Ham
    else
        println("Unkown type of interaction choose between <<two>> or <<three>>")
        return nothing
    end
end;

function Generate_4Body(; r::Float64=1.0, Lx::Float64=-1.0, Ly::Float64=-1.0, N_phi::Int64=10, prec=1e-12)
    if Lx != -1
        println("Generating 4body pseudopotential coefficients from Lx")
        Ly = 2 * pi * N_phi / Lx
        r = Lx / Ly
    elseif Ly != -1
        println("Generating 4body pseudopotential coefficients from Ly")
        Lx = 2 * pi * N_phi / Ly
        r = Lx / Ly
    else
        println("Generating 4body pseudopotential coefficients from r")
        Lx = sqrt(2 * pi * N_phi * r)
        Ly = sqrt(2 * pi * N_phi / r)
    end
    println(
        string(
          "Parameters are N_phi=",
          N_phi,
          ", r=",
          round(r; digits=3),
          ", Lx =",
          round(Lx; digits=3),
          " and Ly =",
          round(Ly; digits=3),
        ),
    )

    coeff = projectors_four_body_parallel(Lx, Ly, N_phi)
    coeff = streamline_four_body(coeff, N_phi)

    return four_body_elements_parallel(N_phi, coeff; prec=prec)
end


function Generate_Idmrg4body(Coeffs; prec=1e-10)
    AllCoeff = [Dict{Vector{Int64}, Float64}() for i=1:length(collect(keys(Coeffs)))]
    i = 1
    for (k, v) in Coeffs
        AllCoeff[i] = v
    end
    
    Coeffs = merge(AllCoeff...)
    @show typeof(Coeffs)
    opt = optimize_coefficients(Coeffs;prec=prec)
    opt = filter_optimized_Hamiltonian_by_first_site(opt)

    return opt
end




function Generate_IdmrgCoeff(Ly::Float64, Vs::Array{Float64};prec=1e-8, PHsym=false)
    rough_N = round(Int64, 2*Ly)-2
    test = round(Int64, 2*Ly)-2
    while rough_N <= test
        rough_N = test + 2
        coeff = Generate_Elements(rough_N, Ly, Vs, "three")

        opt = optimize_coefficients(coeff; prec=prec, PHsym=PHsym)
        opt = filter_optimized_Hamiltonian_by_first_site(opt)
        
        test = check_max_range_optimized_Hamiltonian(opt)
        if rough_N > test
          return opt
        end
    end
end
####################################
####################################
####################################
 

function HermitePolynomial(x, n::Int64)

    if n == 0
        return 1
    elseif n == 1
        return 2*x
    elseif n == 2
        return 4*x^2 - 2
    end
        
    Hm2 = 1
    Hm1 = 2*x
    Hm = 4*x^2 - 2

    for j in 3:n
        Hm2 = Hm1
        Hm1 = Hm
        Hm = 2*x*Hm1 - 2*(j - 1) *Hm2
    end

    return Hm

end;

function W_polynomial(ns...)
    N = length(ns)
    
    if N == 2
        return ns[1] - ns[2]
    end
    
    if N == 1
        return 0
    end
    
    res = 1
    
    for j in 2:N
        res *= (ns[1] - ns[j])
    end
    
    return res * W_polynomial(ns[2:end]...)
end
  
function W_polynomial(n1, n2)
    return (n1 - n2)
end

function W_polynomial(n1, n2, n3)
    return (n1 - n2) * (n1 - n3) * (n2 - n3)
end
  
function W_polynomial(n1, n2, n3, n4)
    return (n1 - n2) * (n1 - n3) * (n1 - n4) * (n2 - n3) * (n2 - n4) * (n3 - n4)
end

function wellOrderdedSet(m1, m2, m3)
    gloablsign= 1
    M = (m1, m2, m3)
    M1 = min(M[1], M[2], M[3])
    if M1 == m2
        gloablsign *= -1
        M = (m2, m1, m3)
    elseif M1 == m3
        M = (m3, m1, m2)
    end

    M2 = max(M[2], M[3])
    if M2 == M[2]
        gloablsign *= -1
        M = (M[1], M[3], M[2])
    end
    @assert M[1] < M[2] && M[2] < M[3]
    return M[1], M[2], M[3], gloablsign
end;

function get_perm!(lis, name)
    for j in 1:(length(lis) - 1)
      if lis[j] > lis[j + 1]
        c = lis[j]
        lis[j] = lis[j + 1]
        lis[j + 1] = c
        c = name[j]
        name[j] = name[j + 1]
        name[j + 1] = c
        sg = get_perm!(lis, name)
        return -sg
      end
    end
    return 1
end;

function filter_op!(lis, name)
    x = 1
    while x <= length(lis) - 1
      if lis[x] == lis[x + 1]
        if name[x] == "Cdag" && name[x + 1] == "C"
          popat!(lis, x + 1)
          popat!(name, x + 1)
          name[x] = "N"
        elseif name[x] == "C" && name[x + 1] == "Cdag"
          popat!(lis, x + 1)
          popat!(name, x + 1)
          name[x] = "Nbar"
        else
          print("Wrong order in filter_op")
        end
      end
      x += 1
    end
end;

function optimize_coefficients(coeff::Dict; prec=1e-12, PHsym = false)
    optimized_dic = Dict()
    for (ke, v) in coeff
      if abs(v) < prec
        continue
      end
      if mod(length(ke), 2) == 1
        error("Odd number of operators is not implemented")
      end
      name = PHsym ? vcat(fill("C", length(ke)÷2), fill("Cdag", length(ke)÷2)) :  vcat(fill("Cdag", length(ke)÷2), fill("C", length(ke)÷2))
      k = Base.copy(ke)
      sg = get_perm!(k, name)
      filter_op!(k, name)

      new_k = [isodd(n) ? name[n ÷ 2 + 1] : k[n ÷ 2] + 1 for n in 1:(2 * length(name))]
      optimized_dic[new_k] = sg * v
    end
    return optimized_dic
end;
  
function filter_optimized_Hamiltonian_by_first_site(coeff::Dict; n=1)
    res = Dict()
    for (k, v) in coeff
      if k[2] == n
        res[k] = v
      end
    end
    return res
end;

function filter_optimized_Hamiltonian_by_first_siteOld(coeff::Dict; n=0)
    res = Dict()
    for (k, v) in coeff
      if k[1] == n
        res[k] = v
      end
    end
    return res
end;
  
function check_max_range_optimized_Hamiltonian(coeff::Dict; check=2)
    temp = 0
    for k in keys(coeff)
      temp = max(temp, k[end] - k[check])
    end
    return temp
end

#############################
#############################
#############################


function finite_Cylinder_MPO(N_Φ::Int64, L_x::Float64, Vs::Array{Float64,1}, prec::Float64, type::String; NeutralizBackGround::Bool=false, filling::Float64=1/2)
    if !NeutralizBackGround ||  type=="three"
        rough_N = N_Φ-2
        test = rough_N
            while rough_N <= test
            rough_N = test + 2
            coeff = Generate_Elements(rough_N, L_x, Vs, type)
            opt = optimize_coefficients(coeff; prec=prec)
            test = check_max_range_optimized_Hamiltonian(opt)
            if rough_N > test
                return  generate_Hamiltonian(opt)
            end
        end
    else
        println("Neutralizing the background for the $(type) body term")
        rough_N = N_Φ-2
        test = rough_N
            while rough_N <= test
            rough_N = test + 2
            coeff = Generate_Elements(rough_N, L_x, Vs, type)
            coeff_bck = Neutralize_Backgroud(coeff, N_Φ, filling)
            opt = optimize_coefficients(coeff; prec=prec)
            coeff_bck = optimize_coefficients(coeff_bck; prec=prec)
            test = check_max_range_optimized_Hamiltonian(opt)
            if rough_N > test
                return  (generate_Hamiltonian(opt) + generate_Hamiltonian(coeff_bck))
            end
        end
    end
    #sorted_opt = sort_by_configuration(opt);
end;

function generate_Hamiltonian(mpo::OpSum, coeff::Dict; global_factor=1, prec=1e-12)
    for (k, v) in coeff
      if abs(v) > prec
        add!(mpo, global_factor * v, k...)
      end
    end
    return mpo
end;
  
function generate_Hamiltonian(coeff::Dict; global_factor=1, prec=1e-12)
    mpo = OpSum()
    return generate_Hamiltonian(mpo, coeff; global_factor=global_factor, prec=prec)
end;