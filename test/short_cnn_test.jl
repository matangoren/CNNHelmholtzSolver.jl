include("test_intro.jl")

# include("../src/solvers/solver_utils.jl")
include("../src/unet/model.jl")
include("test_utils.jl")
include("../src/gpu_krylov.jl")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/data.jl")

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

n = 352
m = 240
domain = [0, 13.5, 0, 4.2]
h = r_type.([(domain[2]-domain[1])./ n, (domain[4]-domain[3])./ m])
n += 32
m += 16

M = getRegularMesh(domain,[n;m])
M.h = h
Ainv = getCnnHelmholtzSolver("VU")
	
Helmholtz_param = HelmholtzParam(M,ones(5,5),ones(5,5),3.9*2*pi,true,true)
Ainv = setMediumParameters(Ainv, Helmholtz_param)

rhs = get_rhs(M.n[1], M.n[2], M.h; blocks=16)
U, Ainv = solveLinearSystem(sparse(ones(size(rhs))), rhs, Ainv)