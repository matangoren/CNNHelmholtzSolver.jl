using Statistics
using Flux
using Flux: @functor
using LaTeXStrings
using KrylovMethods
using Distributions: Normal
using BSON: @load
using Plots
using Dates
using CSV, DataFrames
using Random
using CUDA

use_gpu = true
if use_gpu == true
    CUDA.allowscalar(true)
    cgpu = gpu
else
    cgpu = cpu
    pyplot()
end

pu = cpu # gpu
r_type = Float64
c_type = ComplexF64
u_type = Float32
gmres_type = ComplexF64
a_type = Array{gmres_type}

println("before includes")
include("../src/flux_components.jl")
include("../src/kappa_models.jl")
include("../src/data.jl")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/unet/utils.jl")
include("../src/unet/train.jl")
println("after includes")

fgmres_func = KrylovMethods.fgmres

gamma_val = 0.00001
pad_cells = [10;10]
blocks = 64

kappa_type = 4 # 0 - uniform, 1 - CIFAR10, 2 - STL10
kappa_threshold = 25 # kappa ∈ [0.01*threshold, 1]
smooth = true
k_kernel = 5
f = 10.0
n = m = 128
h = 2.0./(m+n)

println("before kappa")
kappa = r_type.(generate_kappa!(n, m; type=kappa_type, smooth=smooth, threshold=kappa_threshold, kernel=k_kernel)|>pu)
println("after kappa")
omega = r_type(2*pi*f);
gamma = gamma_val*2*pi * ones(r_type,size(kappa))
# gamma = r_type.(absorbing_layer!(gamma, pad_cells, omega))|>pu

x_true = randn(c_type,n+1,m+1, 1, 1)
r_vcycle, _ = generate_r_vcycle!(n, m, h, kappa, omega, gamma, reshape(x_true,n+1,m+1,1,1))
r_vcycle = vec(r_vcycle)
for i = 2:blocks
    global r_vcycle, x_true
    x_true = randn(c_type,n+1,m+1, 1, 1)
    r_vcycle1, _ = generate_r_vcycle!(n, m, h, kappa, omega, gamma, reshape(x_true,n+1,m+1,1,1))
    r_vcycle = cat(r_vcycle, vec(r_vcycle1), dims=2)
end

println("r_vcycle size: ",size(r_vcycle))
println("r_vcycle type: ",typeof(r_vcycle))

_, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
println(typeof(helmholtz_matrix))
A(v) = vec(helmholtz_chain!(reshape(v, n+1, m+1, 1, 1), helmholtz_matrix; h=h))

function As(v)
    res = vec(A(v[:,1]))
    for i = 2:blocks
        res = cat(res, vec(A(v[:,i])), dims=2)
    end

    return res
end

start = time_ns()
result_vector = As(r_vcycle)
println("time for vectorize calculation $((start-time_ns())/1e+9)")
println("As(r_vcycle) size: ", size(result_vector))
println("As(r_vcycle) norm: ", norm(result_vector))

A_chain(vs) = reshape(helmholtz_chain!(reshape(vs, n+1, m+1, 1, blocks), helmholtz_matrix; h=h),(n+1)*(m+1),blocks)

start = time_ns()
result_chain = A_chain(r_vcycle)
println("time for matrix calculation $((start-time_ns())/1e+9)")
println("As(r_vcycle) size: ", size(result_chain))
println("As(r_vcycle) norm: ", norm(result_chain))








