include("test_intro.jl")

include("../src/unet/model.jl")
include("test_utils.jl")
include("../src/gpu_krylov.jl")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/data.jl")

include("../src/solvers/cnn_helmholtz_solver.jl")
ENV["JULIA_CUDA_MEMORY_POOL"] = "none"

useSommerfeldBC = true

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

function get_setup(n,m,domain; blocks=4)
    h = r_type.([(domain[2]-domain[1])./ n, (domain[4]-domain[3])./ m])
    kappa_i, c = get2DSlownessLinearModel(n,m;normalized=false)
    medium = kappa_i.^2
    c = maximum(kappa_i)
    
    omega_exact = r_type((0.1*2*pi) / (c*maximum(h)))
    # omega_fwi = r_type(2*pi*f_fwi)
    kappa = kappa_i ./ c # (kappa_i .* (omega_fwi/(omega_exact*c)))
    omega = omega_exact * c
    
    ABLpad = 20
    ABLamp = omega
    gamma = r_type.(getABL([n+1,m+1],true,ones(Int64,2)*ABLpad,Float64(ABLamp)))
    attenuation = r_type(0.01*4*pi);
    gamma .+= attenuation

    M = getRegularMesh(domain,[n;m])
    M.h = h

    rhs = get_rhs(n,m,h; blocks=blocks)
    return HelmholtzParam(M,Float64.(gamma),Float64.(medium),Float64(omega_exact),true,useSommerfeldBC), rhs
end

# setup
# maybe try 272x192 with f=2.7
# n = 240 #352 #560 #352
# m = 160 #240 #304 #240

# domain = [0, 13.5, 0, 4.2]
# h = r_type.([(domain[2]-domain[1])./ n, (domain[4]-domain[3])./ m])

# n += 32
# m += 16

# generating kappa
# kappa_i, c = get2DSlownessLinearModel(n,m;normalized=false)
# medium = readdlm("FWI_(384, 256)_FC1_GN10.dat", '\t', r_type);
# medium = medium[1:n+1,1:m+1]
# kappa_i = sqrt.(medium)
# medium = kappa_i.^2
# c = maximum(kappa_i)

# omega_exact = r_type((0.1*2*pi) / (c*maximum(h)))
# f_fwi = 2.6 #3.9 # 6.2 # 3.9
# omega_fwi = r_type(2*pi*f_fwi)

# println("c=$(c) - h=$(h)")
# println("omega_exact = $(omega_exact) f_exact = $(omega_exact/2pi)")
# println("omega = $(omega_fwi) f_fwi = $(f_fwi)")


# kappa = (kappa_i .* (omega_fwi/(omega_exact*c)))
# omega = omega_exact * c
# println("final omega = $(omega)")

# ABLpad = 20
# ABLamp = omega
# gamma = r_type.(getABL([n+1,m+1],true,ones(Int64,2)*ABLpad,Float64(ABLamp)))
# attenuation = r_type(0.01*4*pi);
# gamma .+= attenuation

# generating rhs


# M = getRegularMesh(domain,[n;m])
# M.h = h
# Helmholtz_param = HelmholtzParam(M,Float64.(gamma),Float64.(medium),Float64(omega_fwi),true,useSommerfeldBC)
domain = [0, 13.5, 0, 4.2]

solver_type = "VU"

solver_2_6 = getCnnHelmholtzSolver(solver_type; solver_tol=1e-4)
n = 240
m = 160
helmholtz_param, rhs_2_6 = get_setup(n,m,domain; blocks=1)
solver_2_6 = setMediumParameters(solver_2_6, helmholtz_param)


solver_3_9 = copySolver(solver_2_6)
n = 352
m = 240
helmholtz_param, rhs_3_9 = get_setup(n,m,domain)
solver_3_9 = setMediumParameters(solver_3_9, helmholtz_param)

println("solver for 2.6")
result, solver_2_6 = solveLinearSystem(sparse(ones(size(rhs_2_6))), rhs_2_6, solver_2_6,0)|>cpu

println("solver for 3.9")
result, solver_3_9 = solveLinearSystem(sparse(ones(size(rhs_3_9))), rhs_3_9, solver_3_9,0)|>cpu
# plot_results("test_16_cnn_solver_point_source_result_$(solver_type)", result, n ,m)
exit()
solver_2_6 = retrain(1,1, solver_2_6;iterations=10, batch_size=16, initial_set_size=256, lr=1e-6)
solver_3_9.model = solver_2_6.model

println("solver for 2.6 - after retraining")
result, solver_2_6 = solveLinearSystem(sparse(ones(size(rhs_2_6))), rhs_2_6, solver_2_6,0)|>cpu


println("solver for 3.9 - after retraining")
result, solver_3_9 = solveLinearSystem(sparse(ones(size(rhs_3_9))), rhs_3_9, solver_3_9,0)|>cpu

# new_medium = readdlm("FWI_(384, 256)_FC1_GN10.dat", '\t', Float64);
# new_medium = new_medium[1:n+1,1:m+1]
# println(size(new_medium))
# Helmholtz_param = HelmholtzParam(M,Float64.(gamma),Float64.(new_medium),Float64(omega_fwi),true,useSommerfeldBC)
