include("test_intro.jl")

include("../src/unet/model.jl")
include("../src/gpu_krylov.jl")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/data.jl")

include("../src/solvers/cnn_helmholtz_solver.jl")
# ENV["JULIA_CUDA_MEMORY_POOL"] = "none"

useSommerfeldBC = true

function expandModelNearest(m,n,ntarget)
    if length(size(m))==2
        mnew = zeros(Float64,ntarget[1],ntarget[2]);
        for j=1:ntarget[2]
            for i=1:ntarget[1]
                jorig = convert(Int64,ceil((j/ntarget[2])*n[2]));
                iorig = convert(Int64,ceil((i/ntarget[1])*n[1]));
                mnew[i,j] = m[iorig,jorig];
            end
        end
    elseif length(size(m))==3
        mnew = zeros(Float64,ntarget[1],ntarget[2],ntarget[3]);
        for k=1:ntarget[3]
            for j=1:ntarget[2]
                for i=1:ntarget[1]
                    korig = convert(Int64,floor((k/ntarget[3])*n[3]));
                    jorig = convert(Int64,floor((j/ntarget[2])*n[2]));
                    iorig = convert(Int64,floor((i/ntarget[1])*n[1]));
                    mnew[i,j,k] = m[iorig,jorig,korig];
                end
            end
        end
    end
    return mnew
end

function get_seg_model(kappa_file, n,m; doTranspose=false)
    newSize = [n+1, m+1]
    medium = readdlm(kappa_file);
    # medium = medium*1e-3;
    if doTranspose
        medium = medium';
    end
    medium = expandModelNearest(medium,   collect(size(medium)),newSize);

    return r_type.(medium) # m is velocity model
end

function get_rhs(n, m, h; blocks=2)
    rhs = zeros(ComplexF64,n+1,m+1,1,1)
    rhs[floor(Int32,n / 2.0),floor(Int32,m / 2.0),1,1] = r_type(1.0 ./minimum(h)^2)
    rhs = vec(rhs)
    if blocks == 1
        return reshape(rhs, (length(rhs),1))
    end
    for i = 2:blocks
        rhs1 = zeros(ComplexF64,n+1,m+1,1,1)
        rhs1[floor(Int32,(n / blocks)*(i-1)),floor(Int32,(m / blocks)*(i-1)),1,1] = r_type(1.0 ./minimum(h)^2)
        rhs = cat(rhs, vec(rhs1), dims=2)
    end
    return rhs
end

function plot_results(filename, res, n, m)
    heatmap(real(reshape(res[:,1],n+1, m+1)), color=:blues)
    savefig(filename)
end

function get_setup(n,m,domain, original_h, f_fwi; blocks=4, kappa_file="")
    h = r_type.([(domain[2]-domain[1])./ n, (domain[4]-domain[3])./ m])
    # ratio = f_fwi / f_initial_grid
    # h = original_h ./ ratio    

    if kappa_file != ""
        seg_model = get_seg_model(kappa_file, n,m; doTranspose=false) # velocity model
        kappa_i = velocityToSlowness(seg_model)
        println(minimum(kappa_i))
        println(maximum(kappa_i))
        medium = kappa_i.^2

        figure()
        imshow(kappa_i', clim = [0.0, 1.0],cmap = "Blues"); colorbar();
        savefig("m.png")
        close()
    else # default linear kappa
        kappa_i, _ = get2DSlownessLinearModel(n,m;normalized=false)
        medium = kappa_i.^2
    end
    
    c = maximum(kappa_i)
    omega_exact = r_type((0.1*2*pi) / (c*maximum(h)))
    omega_fwi = r_type(2*pi*f_fwi)
    omega = omega_exact * c
    
    ABLpad = 16
    ABLamp = omega_exact
    gamma = r_type.(getABL([n+1,m+1],true,ones(Int64,2)*ABLpad,Float64(ABLamp)))
    attenuation = r_type(0.01*4*pi);
    gamma .+= attenuation

    M = getRegularMesh(domain,[n;m])

    rhs = get_rhs(n,m,h; blocks=blocks)
    return HelmholtzParam(M,Float64.(gamma),Float64.(medium),Float64(omega_fwi),true,useSommerfeldBC), rhs
end

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

original_h = r_type.([13.5 / 608, 4.2/ 304])
f_initial_grid = 6.7555555411700405


solver_type = "VU"

# solver_2_6 = getCnnHelmholtzSolver(solver_type; solver_tol=1e-4)
# n = 256
# m = 128
# f_fwi = (16/42)*f_initial_grid
# helmholtz_param, rhs_2_6 = get_setup(n,m,domain, original_h, f_fwi; blocks=1)
# solver_2_6 = setMediumParameters(solver_2_6, helmholtz_param)


solver_3_9 = getCnnHelmholtzSolver(solver_type; solver_tol=1e-8) # copySolver(solver_2_6)
n = 256
m = 128
f_fwi = (16/42)*f_initial_grid
helmholtz_param, rhs_3_9 = get_setup(n,m,domain, original_h, f_fwi; blocks=1, kappa_file="FWI_(672, 336)_FC1_GN20.dat")
solver_3_9 = setMediumParameters(solver_3_9, helmholtz_param)

# solver_3_9.solver_tol = 1e-8
# println("solver for 2.6")
# result, solver_2_6 = solveLinearSystem(sparse(ones(size(rhs_2_6))), rhs_2_6, solver_2_6,0)|>cpu

println("solver for 3.9")
result, solver_3_9 = @time solveLinearSystem(sparse(ones(size(rhs_3_9))), rhs_3_9, solver_3_9,1)|>cpu

print("SECOND v")
result, solver_3_9 = @time solveLinearSystem(sparse(ones(size(rhs_3_9))), rhs_3_9, solver_3_9,1)|>cpu

# plot_results("test_16_cnn_solver_point_source_result_$(solver_type)", result, n ,m)
exit()
start_time = time_ns()
solver_3_9 = retrain(1,1, solver_3_9;iterations=30, batch_size=16, initial_set_size=64, lr=1e-4, data_epochs=5)
# solver_3_9.model = solver_2_6.model
end_time = time_ns()
println("time took for retrain: $((end_time-start_time)/1e9)")
# println("solver for 2.6 - after retraining")
# result, solver_2_6 = solveLinearSystem(sparse(ones(size(rhs_2_6))), rhs_2_6, solver_2_6,0)|>cpu


println("solver for 3.9 - after retraining")
result, solver_3_9 = solveLinearSystem(sparse(ones(size(rhs_3_9))), rhs_3_9, solver_3_9,0)|>cpu