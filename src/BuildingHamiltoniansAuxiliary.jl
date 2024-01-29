using Revise
using ITensors
using ITensorInfiniteMPS



#### Polynomials ####

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

### Dictionary manipulations


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
  
function filter_optimized_Hamiltonian_by_first_siteGen(coeff::Dict; pos=2, n=1)
    res = Dict()
    for (k, v) in coeff
      if k[pos] == n
        res[k] = v
      end
    end
    return res
end

filter_optimized_Hamiltonian_by_first_site(coeff::Dict; n=1) = filter_optimized_Hamiltonian_by_first_siteGen(coeff; pos=2, n=n)

function filter_dict(coeff::Dict; prec=1e-12)
    res = Dict()
    for (k, v) in coeff
        if abs(v) > prec
            res[k] = v
        end
    end
    return res 
end 

  
function check_max_range_optimized_Hamiltonian(coeff::Dict; check=2)
    temp = 0
    for k in keys(coeff)
      temp = max(temp, k[end] - k[check])
    end
    return temp
end
