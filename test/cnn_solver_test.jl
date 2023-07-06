include("test_intro.jl")

include("../src/unet/model.jl")
include("test_utils.jl")
include("../src/gpu_krylov.jl")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/data.jl")

include("../src/solvers/cnn_helmholtz_solver.jl")
ENV["JULIA_CUDA_MEMORY_POOL"] = "none"

function get_rhs(n, m, h; blocks=2)
    rhs = zeros(ComplexF64,n+1,m+1,1,1)
    rhs[floor(Int32,n / 2.0),floor(Int32,m / 2.0),1,1] = r_type(1.0 ./minimum(h)^2)
    rhs = vec(rhs)
    if blocks == 1
        return reshape(rhs, (length(rhs),1))
    
    end
    for i = 2:blocks
        rhs1 = zeros(ComplexF64,n+1,m+1,1,1)
        rhs1[floor(Int32,(n / blocks)*(i-1)),floor(Int32,(m / blocks)*(i-1)),1,1] = r_type(1.0 ./minimum(h)^2)
        rhs = cat(rhs, vec(rhs1), dims=2)
    end
    return rhs
end

function plot_results(filename, res, n, m)
    heatmap(real(reshape(res[:,1],n+1, m+1)), color=:blues)
    savefig(filename)
end



# setup
# maybe try 272x192 with f=2.7
n = 240 #240#352# 272 #352 #560 #352
m = 160 #160#240# 192 #240 #304 #240

domain = [0, 13.5, 0, 4.2]
h = r_type.([(domain[2]-domain[1])./ n, (domain[4]-domain[3])./ m])

# n += 32
# m += 16

# generating kappa
kappa_i, c = get2DSlownessLinearModel(n,m;normalized=false)
medium = kappa_i.^2
c = maximum(kappa_i)

omega_exact = r_type((0.1*2*pi) / (c*maximum(h)))
f_fwi = 2.7#3.9# 2.7 #3.9 # 6.2 # 3.9
omega_fwi = r_type(2*pi*f_fwi)

println("c=$(c) - h=$(h)")
println("omega_exact = $(omega_exact) f_exact = $(omega_exact/2pi)")
println("omega = $(omega_fwi) f_fwi = $(f_fwi)")


kappa = (kappa_i .* (omega_fwi/(omega_exact*c)))
omega = omega_exact * c
println("final omega = $(omega)")

ABLpad = 20
ABLamp = omega
gamma = r_type.(getABL([n+1,m+1],true,ones(Int64,2)*ABLpad,Float64(ABLamp)))
attenuation = r_type(0.01*4*pi);
gamma .+= attenuation

# generating rhs
rhs = get_rhs(n,m,h; blocks=4)
println("size of rhs $(size(rhs))")


M = getRegularMesh(domain,[n;m])
M.h = h
useSommerfeldBC = true
Helmholtz_param = HelmholtzParam(M,Float64.(gamma),Float64.(medium),Float64(omega_fwi),true,useSommerfeldBC)

solver_type = "VU"

solver = getCnnHelmholtzSolver(solver_type; solver_tol=1e-4)
solver = setMediumParameters(solver, Helmholtz_param)


println(solver_type)
result, param = solveLinearSystem(sparse(ones(size(rhs))), rhs, solver,0)|>cpu
exit()
plot_results("test_16_cnn_solver_point_source_result_$(solver_type)", result, n ,m)


# new_medium = readdlm("FWI_(384, 256)_FC1_GN10.dat", '\t', Float64);
# new_medium = new_medium[1:n+1,1:m+1]
# println(size(new_medium))
# Helmholtz_param = HelmholtzParam(M,Float64.(gamma),Float64.(new_medium),Float64(omega_fwi),true,useSommerfeldBC)

# solver = setMediumParameters(solver, Helmholtz_param)
solver = retrain(1,1,solver;iterations=10, initial_set_size=128, lr=1e-6)

result, param = solveLinearSystem(sparse(ones(size(rhs))), rhs, solver,0)|>cpu
plot_results("test_16_cnn_solver_point_source_result_$(solver_type)_after_retrain", result, n ,m)

