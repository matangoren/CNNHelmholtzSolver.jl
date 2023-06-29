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
model = Dense(200=>100, leakyrelu; init=ones);

function loss3(m, x, y)
    println(size(x))
    norm(m(x) .- y)
end

loss!(x, y) = loss3(model, x, y)
loss!(tuple) = loss!(tuple[1], tuple[2])

x = ones(200,200,1,1)
y = ones(100,200,1,1)

data = Tuple[]
for i=1:20
    append!(data,[(x,y)])
end

batch = data[1:16]
Flux.train!(loss!, Flux.params(model), batch, RADAM(1e-4))
# loader = Flux.Data.DataLoader(ds, batchsize=16, shuffle=true)
# for (x,y) in loader
#     println(size(x))
#     println(typeof(x))
# end
