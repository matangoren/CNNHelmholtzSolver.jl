include("test_intro.jl")

include("../src/unet/model.jl")
include("test_utils.jl")
include("../src/gpu_krylov.jl")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/data.jl")

include("../src/solvers/cnn_helmholtz_solver.jl")


# setup
n = 352
m = 240
domain = [0, 13.5, 0, 4.2]
h = r_type.([(domain[2]-domain[1])./ n, (domain[4]-domain[3])./ m])

# generating kappa
kappa_i, c = get2DSlownessLinearModel(n,m;normalized=false)
medium = kappa_i.^2
c = maximum(kappa_i)

omega_exact = r_type((0.1*2*pi) / (c*maximum(h)))
f_fwi = 3.9 
omega_fwi = r_type(2*pi*f_fwi)

println("c=$(c) - h=$(h)")
println("omega_exact = $(omega_exact) f_exact = $(omega_exact/2pi)")
println("omega = $(omega_fwi) f_fwi = $(f_fwi)")


kappa = (kappa_i .* (omega_fwi/(omega_exact*c)))
omega = omega_exact * c

ABLpad = 20
ABLamp = omega
gamma = r_type.(getABL([n+1,m+1],true,ones(Int64,2)*ABLpad,Float64(ABLamp)))
attenuation = r_type(0.01*4*pi);
gamma .+= attenuation

# set_size = augment_size = 48
# rs, es = generate_retrain_random_data(set_size, augment_size, n, m, h, kappa|>gpu, omega, gamma|>gpu; gmres_restrt=-1, blocks=16);

M = getRegularMesh(domain,[n;m])
M.h = h
useSommerfeldBC = true
Helmholtz_param = HelmholtzParam(M,Float64.(gamma),Float64.(medium),Float64(omega_fwi),true,useSommerfeldBC)

solver_type = "VU"

solver = getCnnHelmholtzSolver(solver_type; solver_tol=1e-4)
solver = setMediumParameters(solver, Helmholtz_param)

solver = retrain(1,1,solver)

