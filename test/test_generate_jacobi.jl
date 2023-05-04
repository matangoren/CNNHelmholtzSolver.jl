using Statistics
using LinearAlgebra
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
# a_type = CuArray{gmres_type}
a_type = Array{gmres_type}
run_title = "64bit_cpu"

println("before includes")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/unet/model.jl")
include("../src/data.jl")
include("../src/unet/train.jl")
include("../src/kappa_models.jl")
include("../src/gpu_krylov.jl")
include("test_utils.jl")
println("after includes")

fgmres_func = KrylovMethods.fgmres # gpu_flexible_gmres #

# Test Parameters

level = 3
v2_iter = 10
gamma_val = 0.00001
pad_cells = [10;10]
point_sorce_results = true
check_unet_as_preconditioner = true
dataset_size = 1
blocks = 10

# Model Parameters

model_type = FFSDNUnet
k_type = FFKappa
resnet_type = SResidualBlock
arch = 2
k_chs = 10
indexes = 3
σ = elu
kernel = (3,3)
e_vcycle_input = false
kappa_type = 4 # 0 - uniform, 1 - CIFAR10, 2 - STL10
kappa_threshold = 25 # kappa ∈ [0.01*threshold, 1]
kappa_input = true
gamma_input = true
axb = false
norm_input = false
smooth = true
k_kernel = 5
before_jacobi = false
unet_in_vcycle = false

n = m = 128
h = r_type(2.0 / (n+m))
f = 10.0
restrt = 10
max_iter = 30
println("before kappa")
kappa = r_type.(generate_kappa!(n, m; type=kappa_type, smooth=smooth, threshold=kappa_threshold, kernel=k_kernel)|>pu)
println("after kappa")
omega = r_type(2*pi*f);
gamma = gamma_val*2*pi * ones(r_type,size(kappa))
gamma = r_type.(absorbing_layer!(gamma, pad_cells, omega))|>pu

_, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma)
x_true = randn(c_type, n+1,m+1,1,1)
b = helmholtz_chain!(x_true, helmholtz_matrix; h=h)

x_vcycle, x_vcycle_channels = generate_jacobi!(n, m, h, kappa, omega, gamma, b)

