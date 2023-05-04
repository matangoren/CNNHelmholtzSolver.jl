# all imports and global variables
include("test_intro.jl")

run_title = "64bit_cpu"


println("before includes")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/unet/model.jl")
include("../src/data.jl")
include("../src/unet/train.jl")
# include("../src/kappa_models.jl")
include("../src/gpu_krylov.jl")
include("test_utils.jl")
println("after includes")

fgmres_func = KrylovMethods.fgmres # gpu_flexible_gmres #

# Test Parameters

level = 3
v2_iter = 10
gamma_val = 0.00001
pad_cells = [20;20]
point_sorce_results = false
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
axb = true
norm_input = false
smooth = true
k_kernel = 5
before_jacobi = false
unet_in_vcycle = false


# test_name = "01_24_40_RADAM_ND_FFSDNUnet_TFFKappa_TSResidualBlockI"
# test_name = "12_02_41_RADAM_ND_FFSDNUnet_TFFKappa_TSResidualBlockI_normalized_linear_kappe"
# test_name = "22_31_07_RADAM_ND_FFSDNUnet_TFFKappa_TSResidualBlockI_normalized_linear_kappa_test"
# test_name = "test_4_FFSDNUnet_TFFKappa_TSResidualBlockI n=128 m=128 Neummann=false padding=[16;16] norm_input=true zero-padding normalized_kappa h=[4.2/n, 4.2/m]"
# test_name = "test_5_FFSDNUnet_TFFKappa_TSResidualBlockI n=128 m=128 Neummann=false padding=[16;16] norm_input=false zero-padding normalized_kappa h=[4.2/n, 4.2/m]"
# test_name = "test_6_FFSDNUnet_TFFKappa_TSResidualBlockI n=128 m=128 Neummann=false padding=[10;10] norm_input=false zero-padding same_kappa=ones h=[1/n, 1/m]"
# test_name = "test_8_FFSDNUnet_TFFKappa_TSResidualBlockI n=256 m=128 Neummann=false padding=[10;10] norm_input=false zero-padding same_kappe=ones h=uniform"
# test_name = "test_9_FFSDNUnet_TFFKappa_TSResidualBlockI n=256 m=128 Neummann=false padding=[10;10] norm_input=false zero-padding kappa linear h=uniform"
# test_name = "test_10_FFSDNUnet_TFFKappa_TSResidualBlockI n=256 m=128 Neummann=false padding=[16;16] norm_input=false zero-padding same_kappa=false kappa=normalized linear h=uniform"
# test_name = "test_11_FFSDNUnet_TFFKappa_TSResidualBlockI n=288 m=176 Neummann=false padding=[16;16] norm_input=false zero-padding same_kappa=false kappa=normalized linear (FWI format) h=with domain"
test_name = "test_13_FFSDNUnet_TFFKappa_TSResidualBlockI n=224 m=112 Neummann=false ABLpad=[20;20] gamma=attenuation norm_input=false zero-padding same_kappa=false kappa=slowness squared=normalized linear (FWI format) h=with domain"

n = 224 # 288
m = 112 # 176
domain = [0, 13.5, 0, 4.2]
h = r_type.([(domain[2]-domain[1])./ n, (domain[4]-domain[3])./ m])
kappa, c = get2DSlownessLinearModel(n,m;normalized=true)|>cpu
# already omega effective - uses normalized kappa (c)
omega = r_type((0.1*2*pi) / maximum(h)) # maxmial effective omega (we absorb c into omega) - hwk = hwc = hw'= 2pi/10


ABLpad = [20;20] # [16;16]
gamma = r_type.(getABL([n+1,m+1], false, ABLpad, Float64(omega)))|>cpu
attenuation = r_type(0.01*4*pi);
gamma .+= attenuation


# f = (0.6*3)/(2*2*pi*maximum(h))
restrt = 10
max_iter = 30
# println("before kappa")
# # kappa = r_type.(generate_kappa!(n, m; type=1, smooth=smooth, threshold=50, kernel=k_kernel)|>pu)
# kappa, c = get2DSlownessLinearModel(n,m; normalized=true)|>pu
# # kappa = r_type.((ones(n+1,m+1))|>pu)
# # file = matopen("models/$(test_name)/kappa.mat", "r"); d = read(file); close(file);
# # kappa = r_type.(d["kappa"]|>pu)
# # kappa_file = matopen("models/$(test_name)/kappa.mat", "r");
# # kappa  = r_type.(read(kappa_file, "kappa"))
# # close(kappa_file);
# println("after kappa")
# omega = r_type(2*pi*f*c); # 2*pi*1.5 / (10*h[1])
# gamma = gamma_val*2*pi * ones(r_type,size(kappa))
# gamma = r_type.(absorbing_layer!(gamma, pad_cells, omega))|>pu
bs = 10

retrain_size = 300
iter = 30

# sm_test_name = "23_48_23_$(run_title)_b$(blocks)_m$(kappa_type)_f$(Int32(f))_$(retrain_size)_$(iter)"
# sm_test_name_r = "$(sm_test_name)_retrain"
# sm_test_name = "28_8_model_random_cifar10_cr106_axb=false"
# sm_test_name = "test_7_linear_model_cr106_axb=true_n_128_m_128_kappa_linear"
sm_test_name = "test13"

println("before model")
model = load_model!("models/$(test_name)/model.bson", e_vcycle_input, kappa_input, gamma_input;kernel = kernel,model_type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)
println("after model")
# model_r128 = load_model!("../models/23_48_23 128 100 10 30 5 f f -1 r=f.bson", e_vcycle_input, kappa_input, gamma_input;kernel = kernel,model_type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)
# model_r256 = load_model!("../models/23_48_23 10 blocks 256 300 10 20 3 f f -1 r=f.bson", e_vcycle_input, kappa_input, gamma_input;kernel = kernel,model_type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)
# model_r512 = load_model!("../models/23_48_23 10 blocks 512 500 20 30 3 f f -1 r=f.bson", e_vcycle_input, kappa_input, gamma_input;kernel = kernel,model_type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)

# model_r128, _ = model_tuning!(model1, sm_test_name, kappa, omega, gamma, n, m, f, retrain_size, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1,kappa_type;residual_loss=false)

if point_sorce_results == false
    check_point_source_problem!("$(Dates.format(now(), "HH_MM_SS")) $(sm_test_name)", model, n, m, h, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
end
if check_unet_as_preconditioner == true #gmres_alternatively_
    # check_model_times!(sm_test_name, model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
    check_model!(sm_test_name, model, n, m, h, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
    # check_model!(sm_test_name_r, model_r128, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
end

# n = m = 256
# f = 20.0
# restrt = 15
# max_iter = 40
# k_kernel = 3
# kappa = r_type.((generate_kappa!(n, m; type=kappa_type, smooth=true, threshold=kappa_threshold, kernel=k_kernel))|>pu)
# omega = r_type(2*pi*f)
# gamma = gamma_val*2*pi * ones(r_type,size(kappa))
# gamma = r_type.((absorbing_layer!(gamma, pad_cells, omega))|>pu)
# bs = 10
# iter = 40
# # model, _ = model_tuning!(model, sm_test_name, kappa, omega, gamma, n, f, 50, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1)
# # model, sm_test_name = model_tuning!(model, sm_test_name, kappa, omega, gamma, n, f,500, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1;residual_loss=false)

# if point_sorce_results == false
#     check_point_source_problem!("$(Dates.format(now(), "HH_MM_SS")) $(sm_test_name)", model, n, m, h, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
# end
# if check_unet_as_preconditioner == true
#     # check_model_times!("$(sm_test_name)", model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
#     check_model!(sm_test_name, model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
#     # check_model!(sm_test_name_r, model_r256, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
# end

# n = m = 512
# f = 40.0
# restrt = 20
# max_iter = 50
# k_kernel = 3
# kappa = r_type.(generate_kappa!(n, m; type=kappa_type, smooth=false, threshold=kappa_threshold, kernel=k_kernel))|>pu
# omega = r_type(2*pi*f)
# gamma = gamma_val*2*pi * ones(r_type,size(kappa))
# gamma = r_type.(absorbing_layer!(gamma, pad_cells, omega))|>pu
# bs = 20
# iter = 40
# # model, _ = model_tuning!(model, sm_test_name, kappa, omega, gamma, n, f, 50, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1)
# # model, sm_test_name = model_tuning!(model, sm_test_name, kappa, omega, gamma, n, f, 500, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1;residual_loss=false)

# if point_sorce_results == false
#     check_point_source_problem!(test_name, model, n, m, h, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
# end
# if check_unet_as_preconditioner == true
#     # check_model_times!("$(sm_test_name)", model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
#     check_model!(sm_test_name, model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
#     # check_model!(sm_test_name_r, model_r512, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
# end
