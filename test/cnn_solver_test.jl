include("test_intro.jl")

include("../src/unet/model.jl")
include("../src/gpu_krylov.jl")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/data.jl")

include("../src/solvers/cnn_helmholtz_solver.jl")
include("utils.jl")

# ENV["JULIA_CUDA_MEMORY_POOL"] = "none"

useSommerfeldBC = true


# (608,304) --- 6.7555555411700405 -> 14
# (576,288) --- 6.400000178096225 -> 13
# (544,272) --- 6.04444481502241 -> 15
# (512,256) --- 5.688888844820669 -> 14
# (480,240) --- 5.3333334817468545 -> 14
# (448,224) --- 4.977777815109077 -> 13
# (416,208) --- 4.622222148471298 -> 12
# (384,192) --- 4.266666785397484 -> 16
# (352,176) --- 3.9111111187597056 -> 22
# (320,160) --- 3.5555554521219275 -> 17
# (288,144) --- 3.2000000890481126 -> 20
# (256,128) --- 2.8444444224103345 -> 15

domain = [0, 13.5, 0, 4.2]
domain += [0, 64*(13.5/608), 0, 32*(4.2/304)]

# original_h = r_type.([13.5 / 608, 4.2/ 304])
f_initial_grid = 6.7555555411700405

kappa_file = "FWI_(672, 336)_Cyc2_FC7_GN15"

n = 28*16
m = Int64(n//2)
f_fwi = (28/42)*f_initial_grid
helmholtz_param, rhs = get_setup(n,m,domain, f_fwi; blocks=4, kappa_file="$(kappa_file).dat") # default linear velocity model

solver_name = "VU"
solver_vu = getCnnHelmholtzSolver(solver_name; solver_tol=1e-4, inTesting=true)
solver_vu = setMediumParameters(solver_vu, helmholtz_param)
# solve
println("===== solving for $(solver_name) =====")
result_vu, vu_history = solveLinearSystem(sparse(ones(size(rhs))), rhs, solver_vu,0)
exit()

solver_name = "V"
solver_v = copySolver(solver_vu)
solver_v = setSolverType(solver_name, solver_v)
# solve
println("===== solving for $(solver_name) =====")
result_v, v_history = solveLinearSystem(sparse(ones(size(rhs))), rhs, solver_v,0)


start_time = time_ns()
solver_vu = retrain(1,1, solver_vu;iterations=30, batch_size=16, initial_set_size=64, lr=1e-6, data_epochs=5)
end_time = time_ns()
println("time took for retrain: $((end_time-start_time)/1e9)")


println("===== solving for $(solver_name) - after retraining =====")
result_vu, vu_history_retraining = solveLinearSystem(sparse(ones(size(rhs))), rhs, solver_vu,0)

test_name = "$(kappa_file) convergence initial test"
VU_V_graph("$(test_name) n=$(n) m=$(m) axb=true",[vu_history', v_history', vu_history_retraining'], ["VU", "V", "VU-retraining"], ["blue", "green", "orange"])
