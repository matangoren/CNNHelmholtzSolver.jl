using Statistics
using LinearAlgebra
using Flux
using Flux: @functor
using Flux.Data: DataLoader
using LaTeXStrings
using KrylovMethods
using Distributions: Normal
using BSON: @load
using Plots
using CSV, DataFrames
using Dates
using Random
using CUDA

use_gpu = true
if use_gpu == true
    # CUDA.allowscalar(false)
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

include("../src/multigrid/helmholtz_methods.jl")
include("../src/unet/model.jl")
include("../src/data.jl")
include("../src/unet/train.jl")
include("../src/kappa_models.jl")
include("test_utils.jl")

fgmres_func = KrylovMethods.fgmres # gpu_flexible_gmres #

function test_train_unet!(n, f; is_save=false, data_augmentetion=false, e_vcycle_input=false,
                                    kappa_type=1, threshold=50, kappa_input=true, kappa_smooth=false, k_kernel=3,
                                    gamma_input=true, kernel=(3,3), smaller_lr=10, v2_iter=10, level=3,
                                    axb=false, norm_input=false, model_type=SUnet, k_type=NaN, resnet_type=SResidualBlock, k_chs=-1, indexes=3, data_path="", full_loss=false, residual_loss=false, gmres_restrt=1, σ=elu, arch=1)

    h = 1.0./n;
    gamma_val = 0.00001
    pad_cells = [10;10]
    kappa = r_type.(ones(r_type,n-1,n-1)|>pu)
    omega = r_type(2*pi*f);
    gamma = gamma_val*2*pi * ones(r_type,size(kappa))
    gamma = r_type.(absorbing_layer!(gamma, pad_cells, omega))|>pu
    
    dataset_size = 10000
    dataset = generate_random_data!(dataset_size, n, n, kappa, omega, gamma; e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level,
                                                          kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, axb=axb, norm_input=norm_input, gmres_restrt=gmres_restrt)
    # println("dataset size $(size(dataset[:][0]))")
    
    rs = vcat(reshape.(getfield.(dataset, 1), (n-1)*(n-1), 3)...)
    es = vcat(reshape.(getfield.(dataset, 2), (n-1)*(n-1), 2)...)
    println("dataset rs typeof $(typeof(rs))")
    println("dataset es typeof $(typeof(es))")

    df = DataFrame(hcat(rs,es), [:RR, :RI, :KAPPA, :ER, EI])
    
    CSV.write("./data/dataset_1.csv", df)
    
end

init_lr = 0.0001
opt = RADAM(init_lr)
train_size = 10000
test_size = 100
batch_size = 20
iterations = 120
full_loss = false
gmres_restrt = -1 # 1 -Default, 5 - 5GMRES, -1 Random
blocks = 10

test_train_unet!(128, 10.0;
                    data_augmentetion = false,
                    e_vcycle_input = false,
                    kappa_type = 1,
                    kappa_input = true,
                    threshold = 25,
                    kappa_smooth = true,
                    k_kernel = 5,
                    gamma_input = true,
                    kernel = (3,3),
                    smaller_lr = 48,
                    v2_iter = 10,
                    level = 3,
                    axb = false,
                    norm_input = false,
                    model_type = FFSDNUnet,
                    k_type = TFFKappa,
                    resnet_type = TSResidualBlockI,
                    k_chs = 10,
                    arch = 2,
                    indexes = 3,
                    full_loss = full_loss,
                    residual_loss = false,
                    data_path = "",
                    gmres_restrt = gmres_restrt,
                    σ = elu)
