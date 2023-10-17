include("test_intro.jl")

include("../src/unet/model.jl")
include("../src/gpu_krylov.jl")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/data.jl")

include("../src/solvers/cnn_helmholtz_solver.jl")
include("utils.jl")

files = readdir(joinpath(@__DIR__, "files/"))
sort!(files, by=x->parse(Int, "$(match(r".*[0-9]+", split(split(x, "Cyc")[2],"_")[1]).match)$(match(r".*[0-9]+", split(split(x, "FC")[2],"_")[1]).match)"))

useSommerfeldBC = true
domain = [0, 13.5, 0, 4.2]
domain += [0, 64*(13.5/608), 0, 32*(4.2/304)]

f_max = 6.7555556;
size_max = 42
sizes = [i for i=16:2:size_max]
f = (sizes./42)*f_max

solver_name = "VU"
solver = getCnnHelmholtzSolver(solver_name; solver_tol=1e-4)

for file in files
    
    freqIndex = parse(Int64,split(split(file, "FC")[2], "_")[1])
    model_name = split(file,".")[1]
    println(joinpath(@__DIR__, "models/$(model_name).bson"))
    if isfile(joinpath(@__DIR__, "models/$(model_name).bson"))
        println("exists")
        continue
    end
    n = sizes[freqIndex]*16
    m = Int64(n/2)
    f_i = f[freqIndex]

    kappa_file = joinpath(@__DIR__, "files/$(file)")
    seg_model = get_seg_model(kappa_file, n,m; doTranspose=false) # velocity model
    figure()
    imshow(seg_model', clim = [1.5,4.5],cmap = "jet"); colorbar();
    savefig(joinpath(@__DIR__,"kappas/$(model_name).png"))
    close()

    helmholtz_param, _ = get_setup(n,m,domain, f_i; blocks=1, kappa_file=kappa_file)
    global solver = setMediumParameters(solver, helmholtz_param)

    start_time = time_ns()
    global solver = retrain(1,1, solver;iterations=30, batch_size=16, initial_set_size=128, lr=1e-4, data_epochs=4)
    end_time = time_ns()
    println("time took for retrain: $((end_time-start_time)/1e9)")

    model = solver.model|>cpu
    @save joinpath(@__DIR__, "models/$(model_name).bson") model
end



