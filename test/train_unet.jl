# using Statistics
# using LinearAlgebra
# using Flux
# using Flux: @functor
# using Flux.Data: DataLoader
# using LaTeXStrings
# using KrylovMethods
# using Distributions: Normal
# using BSON: @load
# using Plots
# using CSV, DataFrames
# using Dates
# using Random

# using CUDA
# using CUDA: CuIterator
# using MAT
# using Helmholtz

# # pyplot()

# use_gpu = true
# if use_gpu == true
#     CUDA.allowscalar(true)
#     cgpu = gpu
# else
#     cgpu = cpu
# end

# pu = cpu # gpu
# r_type = Float32
# c_type = ComplexF32
# u_type = Float32
# gmres_type = ComplexF32
# # a_type = CuArray{gmres_type}
# a_type = Array{gmres_type}
include("test_intro.jl")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/unet/model.jl")
include("../src/data.jl")
include("../src/unet/train.jl")
include("../src/gpu_krylov.jl")
include("test_utils.jl")
# # ENV["CUDA_VISIBLE_DEVICES"] = "0,1"
# # ENV["JULIA_CUDA_MEMORY_POOL"] = "none"

if use_gpu == true
    fgmres_func = gpu_flexible_gmres
else
    fgmres_func = KrylovMethods.fgmres
end

function test_train_unet!(n, m, h, opt, init_lr, train_size, test_size, batch_size, iterations;
                                    is_save=false, data_augmentetion=false, e_vcycle_input=false,
                                    kappa_type=1, threshold=50, kappa_input=true, kappa_smooth=false, k_kernel=3,
                                    gamma_input=true, kernel=(3,3), smaller_lr=10, v2_iter=10, level=3,
                                    axb=false, norm_input=false, model_type=SUnet, k_type=NaN, resnet_type=SResidualBlock, k_chs=-1, indexes=3, data_path="", full_loss=false, residual_loss=false, gmres_restrt=1, σ=elu, arch=1)

    kappa, c = get2DSlownessLinearModel(n,m;normalized=true)|>cgpu
    omega = r_type((0.1*2*pi) / maximum(h)) # maxmial effective omega (we absorb c into omega) - hwk = hwc = hw'= 2pi/10

    ABLpad = [20;20]
    gamma = r_type.(getABL([n+1,m+1], true, ABLpad, Float64(omega)))|>cgpu
    attenuation = r_type(0.01*4*pi);
    gamma .+= attenuation

    # test_name = "test_16_FFSDNUnet_TFFKappa_TSResidualBlockI n=352 m=240 Neummann=true ABLpad=[20;20] gamma=attenuation norm_input=false mirror-padding same_kappa=false kappa=slowness squared=normalized linear (FWI format) h=with domain"
    test_name = "Encoder-Solver"

    mkpath("models/$(test_name)")
    mkpath("models/$(test_name)/train_log")                                                                                                                                         
    kappa_file = matopen("models/$(test_name)/kappa.mat", "w");
    write(kappa_file, "kappa", kappa|>cpu)
    close(kappa_file);                                                                      
    
    file = matopen("models/$(test_name)/model_parameters", "w");
    write(file,"e_vcycle_input",e_vcycle_input)
    write(file,"kappa_input",kappa_input)
    write(file,"gamma_input",gamma_input)
    write(file,"kernel",collect(kernel))
    write(file,"model_type",string(model_type))
    write(file,"k_type",string(k_type))
    write(file,"resnet_type",string(resnet_type))
    write(file,"k_chs",k_chs)
    write(file,"indexes",indexes)
    write(file,"sigma",string(σ))
    write(file,"arch",arch)
    close(file);
    
    println("Before model gpu status $(CUDA.memory_status())")
    model = create_model!(e_vcycle_input, kappa_input, gamma_input; kernel=kernel, type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)|>cgpu
    println("After model gpu status $(CUDA.memory_status())")
    model, train_loss, test_loss = train_residual_unet!(model, test_name, n, m, h, kappa, omega, gamma,
                                                        train_size, test_size, batch_size, iterations, init_lr;
                                                        e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level, data_augmentetion=data_augmentetion,
                                                        kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, k_kernel=k_kernel,
                                                        gamma_input=gamma_input, kernel=kernel, smaller_lr=smaller_lr, axb=axb, jac=false, norm_input=norm_input, model_type=model_type, k_type=k_type, k_chs=k_chs, indexes=indexes,
                                                        data_train_path=data_path, full_loss=full_loss, residual_loss=residual_loss, gmres_restrt=gmres_restrt,σ=σ, same_kappa=false, linear_kappa=true)

    iter = range(1, length=Int64(iterations/4))
    p = plot(iter, train_loss, label="Train loss")
    plot!(iter, test_loss, label="Test loss")
    yaxis!("Loss", :log10)
    xlabel!("Iterations")
    savefig("models/$(test_name)/train_log/loss")
end


init_lr = 0.0001
opt = RADAM(init_lr)
train_size = 100 # 20000
test_size = 100 # 1000
batch_size = 8
iterations = 120
full_loss = false
gmres_restrt = -1 # 1 -Default, 5 - 5GMRES, -1 Random
blocks = 10
n = 352 # 224 # 288
m = 240 # 112 # 176

domain = [0, 13.5, 0, 4.2]
h = r_type.([(domain[2]-domain[1])./ n, (domain[4]-domain[3])./ m])

test_train_unet!(n, m, h, opt, init_lr, train_size, test_size, batch_size, iterations;
                    data_augmentetion = false,
                    e_vcycle_input = false,
                    kappa_type = 1,
                    kappa_input = true,
                    threshold = 25,
                    kappa_smooth = true,
                    k_kernel = 5,
                    gamma_input = true,
                    kernel = (3,3),
                    smaller_lr = 60,
                    v2_iter = 10,
                    level = 3,
                    axb = false,
                    norm_input = false,
                    model_type = Solver, #FFSDNUnet,
                    k_type = Encoder, #TFFKappa,
                    resnet_type = TSResidualBlockI,
                    k_chs = 10,
                    arch = 2,
                    indexes = 3,
                    full_loss = full_loss,
                    residual_loss = false,
                    data_path = "",
                    gmres_restrt = gmres_restrt,
                    σ = elu)
