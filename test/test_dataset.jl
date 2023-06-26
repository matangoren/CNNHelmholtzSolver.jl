include("test_intro.jl")
include("../src/unet/CustomDatasets.jl")
include("../src/gpu_krylov.jl")
if use_gpu == true
    fgmres_func = gpu_flexible_gmres
else
    fgmres_func = KrylovMethods.fgmres
end
include("../src/data.jl")
include("../src/multigrid/helmholtz_methods.jl")
# model = Dense(2=>1, leakyrelu; init=ones);

# function loss3(m, x, y)
#     println(typeof(x))
#     println(typeof(y))
#     norm(m(x) .- y)
# end

# loss!(x, y) = loss3(model, x, y)
# loss!(tuple) = loss!(tuple[1], tuple[2])

# x = ones(560,304,1,1)
# y = ones(560,304,1,1)

# data = Tuple[]
# for i=1:10
#     append!(data,[(x,y)])
# end
# println(2⋅2)
kappa = gamma = r_type.(ones(257,257))|>cgpu
x, y = generate_retrain_random_data(100,100,256,256,r_type.([1/256, 1/256]), kappa, r_type(3.9), gamma)
ds = UnetDatasetFromArray(x,y, kappa, gamma)
loader = Flux.Data.DataLoader(ds, batchsize=16, shuffle=true)
for (x,y) in loader
    println(size(x))
    println(typeof(x))
end
