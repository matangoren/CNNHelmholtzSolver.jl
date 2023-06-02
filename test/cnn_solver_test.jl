include("test_intro.jl")

include("../src/unet/model.jl")
include("test_utils.jl")
include("../src/gpu_krylov.jl")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/data.jl")

include("../src/solvers/solver_utils.jl")
include("../src/solvers/cnn_helmholtz_solver.jl")


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
n = 352
m = 240
domain = [0, 13.5, 0, 4.2]
h = r_type.([(domain[2]-domain[1])./ n, (domain[4]-domain[3])./ m])
n += 32
m += 16

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

# generating rhs
rhs = get_rhs(n,m,h; blocks=1)
println("size of rhs $(size(rhs))")


M = getRegularMesh(domain,[n;m])
M.h = h
useSommerfeldBC = true
Helmholtz_param = HelmholtzParam(M,Float64.(gamma),Float64.(medium),Float64(omega_fwi),true,useSommerfeldBC)


solver = getCnnHelmholtzSolver("JU")
solver = setMediumParameters(solver, Helmholtz_param)


println("JU")
result_3_9, param = solveLinearSystem(sparse(ones(size(rhs))), rhs, solver,1)|>cpu
plot_results("test_16_cnn_solver_point_source_result_JU", result_3_9, n ,m)