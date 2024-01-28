using FileIO
using JLD2
using ITensors


include("BuildingHamiltoniansAuxiliary.jl")

##############
#  two body  #
##############

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


function all_projectors(N_Φ, L_x, max_angular)
    return all_projector_two_body(N_Φ, L_x, max_angular)
end

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
    else
        println("Unkown type of interaction choose between <<two>> or <<three>>")
        return nothing
    end
end;


################
#  three body  #
################

function build_three_body_coefficient_factorized_cylinder(Lx, Ly, N_phi; prec=1e-12, gap=false)
    renorm = gap ? sqrt(0.07216878367598695) : 1.
    coefficients = Dict{Tuple{Int64,Int64},Float64}()
    for g in 0:2
      for (k, q) in Iterators.product((-3N_phi - g):3:(3N_phi - 1), (-3N_phi - g):3:(3N_phi - 1))
        coefficients[(k, q)] =
          sqrt(Lx / Ly) / sqrt(N_phi) *
          (2 * pi / Ly)^3 *
          W_polynomial(k / 3, q / 3, -k / 3 - q / 3) *
          exp(-2 * pi^2 / Ly^2 * ((k / 3)^2 + (q / 3)^2 + (-k / 3 - q / 3)^2))/renorm
      end
    end
    return coefficients
end


function build_three_body_coefficient_factorized_cylinder_5(Lx, Ly, N_phi; prec=1e-12)
    coefficients = Dict{Tuple{Int64,Int64},Float64}()
    for g in 0:2
        for (k, q) in Iterators.product((-3N_phi - g):3:(3N_phi - 1), (-3N_phi - g):3:(3N_phi - 1))

            W2 = 2*3* pi^2 / Ly^2 * ((k / 3)^2 + (q / 3)^2 + (-k / 3 - q / 3)^2)
            coefficients[(k, q)] =
                sqrt(Lx / Ly) / sqrt(N_phi) *
                (2 * pi / Ly)^3 *
                W_polynomial(k / 3, q / 3, -k / 3 - q / 3) * 
                exp(-W2/3)
        end
    end
    return coefficients
end 

function build_three_body_coefficient_factorized_cylinder_6(Lx, Ly, N_phi; prec=1e-12)
    coefficients = Dict{Tuple{Int64,Int64},Float64}()
    for g in 0:2
        for (k, q) in Iterators.product((-3N_phi - g):3:(3N_phi - 1), (-3N_phi - g):3:(3N_phi - 1))

            W2 = 2*3* pi^2 / Ly^2 * ((k / 3)^2 + (q / 3)^2 + (-k / 3 - q / 3)^2)
            coefficients[(k, q)] =
                sqrt(Lx / Ly) / sqrt(N_phi) *
                (2 * pi / Ly)^6 *
                W_polynomial(k / 3, q / 3, -k / 3 - q / 3) *
                (k/3)*(q/3)*(-k/3-q/3)*
                exp(-W2/3)
        end
    end
    return coefficients
end 

function streamline_three_body_dictionnary_cylinder(dic, N_phi)
    coefficients = Dict{Tuple{Int64,Int64,Int64},Float64}()
    for n_1 in 0:(N_phi - 3)
      for n_2 in (n_1 + 1):(N_phi - 2)
        for n_3 in (n_2 + 1):(N_phi - 1)
          R3 = n_1 + n_2 + n_3
          coefficients[(n_1, n_2, n_3)] = dic[(R3 - 3*n_1, R3 - 3*n_2)]
        end
      end
    end
    return coefficients
end

function build_hamiltonian_from_three_body_factorized_streamlined_dictionary(
    coeff, N_phi; global_sign=1, prec=1e-12
  )
    full_coeff = Dict{Array{Int64,1},Float64}()
    for R3 in 3:(3*N_phi - 3)  #Barycenter
      for n_1 in max(0, R3 - 2N_phi + 3):min(N_phi - 3, R3 ÷ 3 - 1)
        for n_2 in max(n_1 + 1, R3 - n_1 - N_phi + 1):min(N_phi - 2, (R3 - n_1 - 1) ÷ 2)
          n_3 = R3 - n_1 - n_2
          for m_1 in max(0, R3 - 2N_phi + 3):min(N_phi - 3, R3 ÷ 3 - 1)
            for m_2 in max(m_1 + 1, R3 - m_1 - N_phi + 1):min(N_phi - 2, (R3 - m_1 - 1) ÷ 2)
              m_3 = mod(R3 - m_1 - m_2, N_phi)
              temp = global_sign * (coeff[n_1, n_2, n_3]' * coeff[m_1, m_2, m_3])
              if abs(temp) > prec
                full_coeff[[m_1, m_2, m_3, n_3, n_2, n_1]] = temp
              end
            end
          end
        end
      end
    end
    return full_coeff
end
  
function build_hamiltonian_from_three_body_factorized_streamlined_dictionary(
    coeff::Vector{Dict}, N_phi; global_sign=1, prec=1e-12
  )
    full_coeff = Dict{Array{Int64,1},Float64}()
    for R3 in 3:(3*N_phi - 3)  #Barycenter
      for n_1 in max(0, R3 - 2N_phi + 3):min(N_phi - 3, R3 ÷ 3 - 1)
        for n_2 in max(n_1 + 1, R3 - n_1 - N_phi + 1):min(N_phi - 2, (R3 - n_1 - 1) ÷ 2)
          n_3 = R3 - n_1 - n_2
          for m_1 in max(0, R3 - 2N_phi + 3):min(N_phi - 3, R3 ÷ 3 - 1)
            for m_2 in max(m_1 + 1, R3 - m_1 - N_phi + 1):min(N_phi - 2, (R3 - m_1 - 1) ÷ 2)
              m_3 = mod(R3 - m_1 - m_2, N_phi)
              temp = 0 
              for dictCoeff in coeff
                temp += global_sign * (dictCoeff[n_1, n_2, n_3]' * dictCoeff[m_1, m_2, m_3])
              end
              if abs(temp) > prec
                full_coeff[[m_1, m_2, m_3, n_3, n_2, n_1]] = temp
              end
            end
          end
        end
      end
    end
    return full_coeff
end


function build_three_body_pseudopotentials(;
    r::Float64=1.0,
    Lx::Float64=-1.0,
    Ly::Float64=-1.0,
    N_phi::Int64=10,
    prec=1e-12,
    global_sign=1,
    gap = false, 
    Haffnian = false
  )
    if Lx != -1
      println("Generating 3 body pseudopotential coefficients from Lx")
      flush(stdout)
      Ly = 2 * pi * N_phi / Lx
      r = Lx / Ly
    elseif Ly != -1
      println("Generating 3 body pseudopotential coefficients from Ly")
      flush(stdout)
      Lx = 2 * pi * N_phi / Ly
      r = Lx / Ly
    else
      println("Generating 3 body pseudopotential coefficients from r")
      flush(stdout)
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
    flush(stdout)

    coeff = 0 
    if Haffnian 
        println("Generating Haffnian Hamiltonian")
        flush(stdout)
        coeff = Vector{Dict}(undef, 3)
        coeff[1] = build_three_body_coefficient_factorized_cylinder(Lx, Ly, N_phi; prec=prec, gap=false)
        coeff[2] = build_three_body_coefficient_factorized_cylinder_5(Lx, Ly, N_phi; prec=prec)
        coeff[3] = build_three_body_coefficient_factorized_cylinder_6(Lx, Ly, N_phi; prec=prec)

        coeff[1] = streamline_three_body_dictionnary_cylinder(coeff[1], N_phi)
        coeff[2] = streamline_three_body_dictionnary_cylinder(coeff[2], N_phi)
        coeff[3] = streamline_three_body_dictionnary_cylinder(coeff[3], N_phi)
    else 
        coeff = build_three_body_coefficient_factorized_cylinder(Lx, Ly, N_phi; prec=prec, gap=gap)
        coeff = streamline_three_body_dictionnary_cylinder(coeff, N_phi)
    end 

    coeff = build_hamiltonian_from_three_body_factorized_streamlined_dictionary(coeff, N_phi; global_sign=global_sign, prec=prec)
    return coeff
end


function Generate_OptIdmrg(Ly; prec=1e-10)
    rough_N = round(Int64, 2*Ly)-2
    test = round(Int64, 2*Ly)-2
    while rough_N <= test
        rough_N = test + 2
        coeff = build_three_body_pseudopotentials(;N_phi=rough_N, Ly=Ly, prec=prec)

        opt = filter_dict(coeff; prec=prec)
        opt = filter_optimized_Hamiltonian_by_first_siteGen(opt; pos=1, n=0)
        
        test = check_max_range_optimized_Hamiltonian(opt; check=1)

        if rough_N > test
          return opt
        end
    end

end


function Generate_IdmrgCoeff(Ly::Float64, Vs::Array{Float64};prec=1e-8, PHsym=false)
    rough_N = round(Int64, 2*Ly)-2
    test = round(Int64, 2*Ly)-2
    while rough_N <= test
        rough_N = test + 2
        coeff = build_three_body_pseudopotentials(;N_phi=rough_N, Ly=Ly, prec=prec)

        opt = optimize_coefficients(coeff; prec=prec, PHsym=PHsym)
        opt = filter_optimized_Hamiltonian_by_first_site(opt)
        
        test = check_max_range_optimized_Hamiltonian(opt)
        if rough_N > test
          return opt
        end
    end
end

################
#  four body  #
################


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

function projectors_four_body_parallel(Lx, Ly, N_Φ; prec=1e-12)
    prefactor = sqrt(Lx/Ly)*N_Φ*(2*pi/Ly)^5/sqrt(204.42908051834982)
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



####################################
####################################
####################################
 

function Generate_Idmrg(Coeffs; prec=1e-8)
    AllCoeff = [Dict{Vector{Any}, Float64}() for i=1:length(collect(keys(Coeffs)))]
    i = 1
    CoeffsNew = Dict()
    println("Merging the coeffs for length $(length(Coeffs))")
    flush(stdout)
    for (k, v) in Coeffs
        AllCoeff[i] = v
        i+= 1
    end
    AllCoeff = reduce(merge, AllCoeff)
    opt = optimize_coefficients(AllCoeff; prec=prec)
    opt = filter_optimized_Hamiltonian_by_first_siteGen(opt; pos=2, n=1)
    println("Coefficents generated")
    flush(stdout)
    return opt
end
#############################
#############################
#############################

function generate_Hamiltonian(mpo::OpSum, coeff::Dict; global_factor=1, prec=1e-7)
    for (k, v) in coeff
      if abs(v) > prec
        add!(mpo, global_factor * v, k...)
      end
    end
    return mpo
end;
  
function generate_Hamiltonian(coeff::Dict; global_factor=1, prec=1e-7)
    mpo = OpSum()
    return generate_Hamiltonian(mpo, coeff; global_factor=global_factor, prec=prec)
end;