include("../src/unet/UnetDataset.jl")
include("test_intro.jl")

n = m = 512

x = ones(n+1,m+1,3,1)
y = ones(n+1,m+1,2,1)
gamma = ones(n+1,m+1,1,1)

data = [(x,y),(x,y),(x,y),(x,y),(x,y)]

dataset = Dataset{Float32}(data, gamma)
loader = Flux.Data.DataLoader(dataset, batchsize=4, shuffle=true)
println("Loader length: $(length(loader))")
for (i, (x, y)) in enumerate(loader)
    i == 10 && break
    println("$i: $(size(x)) $(size(y))")
end