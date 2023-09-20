include("../multigrid/helmholtz_methods.jl")
include("../gpu_krylov.jl")
include("../unet/losses.jl")
include("../unet/CustomDatasets.jl")

if use_gpu == true
    fgmres_func = gpu_flexible_gmres
else
    fgmres_func = KrylovMethods.fgmres
end

function get_solver_type(solver_name)
    if solver_name == "JU"
        return Dict("before_jacobi"=>true, "unet"=>true, "after_jacobi"=>true, "after_vcycle"=>false)
    elseif solver_name == "VU"
        return Dict("before_jacobi"=>false, "unet"=>true, "after_jacobi"=>false, "after_vcycle"=>true)
    elseif solver_name == "U"
        return Dict("before_jacobi"=>false, "unet"=>true, "after_jacobi"=>false, "after_vcycle"=>false)
    end
    return Dict("before_jacobi"=>false, "unet"=>false, "after_jacobi"=>false, "after_vcycle"=>true)
end

function setupSolver()
    file = matopen(joinpath(@__DIR__, "../../models/$(model_name)/model_parameters"), "r"); DICT = read(file); close(file);
    e_vcycle_input = DICT["e_vcycle_input"]
    kappa_input = DICT["kappa_input"]
    gamma_input = DICT["gamma_input"]
    kernel = Tuple(DICT["kernel"])
    model_type = @eval $(Symbol(DICT["model_type"])) # FFSDNUnet
    k_type = @eval $(Symbol(DICT["k_type"])) # TFFKappa
    resnet_type = @eval $(Symbol(DICT["resnet_type"])) # TSResidualBlockI
    k_chs = DICT["k_chs"]
    indexes = DICT["indexes"]
    σ = @eval $(Symbol(DICT["sigma"])) # elu
    arch = DICT["arch"]

    model = create_model!(e_vcycle_input, kappa_input, gamma_input; kernel=kernel, type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)
    model = model|>cpu
    println("after create")
    @load joinpath(@__DIR__, "../../models/$(model_name)/model.bson") model
    @info "$(Dates.format(now(), "HH:MM:SS.sss")) - Load Model"
    
    return model|>cgpu, DICT
end

function get_kappa_features(param::CnnHelmholtzSolver)
    model = param.model
    n = param.n
    m = param.m
    kappa = param.kappa
    gamma = param.gamma
    arch = (param.model_parameters)["arch"]
    indexes = (param.model_parameters)["indexes"]

    kappa_features = NaN
    if arch != 0
        kappa_input = reshape(kappa.^2, n+1, m+1, 1, 1)
        if indexes != 3
            kappa_input = cat(kappa_input, reshape(gamma, n+1, m+1, 1, 1), dims=3)
        end
        kappa_features = model.kappa_subnet(kappa_input|>cgpu)
    end
    return kappa_features
end

function solve(param::CnnHelmholtzSolver, r_vcycle, restrt, max_iter; v2_iter=10, level=3, axb=false)

    solver_type = param.solver_type
    model = param.model
    n = param.n
    m = param.m
    h = param.h
    kappa = param.kappa
    kappa_features = param.kappa_features
    omega = param.omega
    gamma = param.gamma
    arch = (param.model_parameters)["arch"]
    solver_tol = param.solver_tol
    relaxation_tol = param.relaxation_tol

    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    blocks = size(r_vcycle,2)
    coefficient = r_type(minimum(h)^2)

    A(v) = vec(helmholtz_chain!(reshape(v, n+1, m+1, 1, blocks), helmholtz_matrix; h=h))
    
    function M_Unet(r)
        r = reshape(r, n+1, m+1, 1, blocks)
        rj = reshape(r, n+1, m+1, 1, blocks)
        e = a_type(zeros(c_type, n+1, m+1, 1, blocks))
        # ej = zeros(c_type, n+1, m+1, 1, blocks)|>cgpu

        if solver_type["before_jacobi"] == true
            e = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
            # ej = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
            rj = r - reshape(A(ej), n+1, m+1, 1, blocks) # I think the reshape is redundant here
        end
        
        ABLpad = [16;16]
        gamma_net = r_type.(getABL([n+1,m+1], true, ABLpad, Float64(1.0)))|>cgpu
        attenuation = r_type(0.01*4*pi);
        gamma_net .+= attenuation
        
        if solver_type["unet"] == true
            for i=1:blocks  
                input = complex_grid_to_channels!(reshape(rj[:,:,1,i],n+1,m+1,1,1); blocks=1)
                if arch == 1
                    input = cat(input, reshape(kappa.^2, n+1, m+1, 1, 1), reshape(gamma_net, n+1, m+1, 1, 1), kappa_features, dims=3)
                    e_unet = model.solve_subnet(input)
                elseif arch == 2
                    input = cat(input, reshape(kappa.^2, n+1, m+1, 1, 1), reshape(gamma_net, n+1, m+1, 1, 1), dims=3)
                    e_unet = model.solve_subnet(input, kappa_features)
                else
                    input = cat(input, reshape(kappa.^2, n+1, m+1, 1, 1), reshape(gamma_net, n+1, m+1, 1, 1), dims=3)
                    e_unet = model(input)
                end
                e[:,:,1,i] .+= (e_unet[:,:,1,1] + im*e_unet[:,:,2,1]) .* coefficient
            end
        end
        
        if solver_type["after_jacobi"] == true
            e = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
        elseif solver_type["after_vcycle"] == true
            e, = v_cycle_helmholtz!(n, m, h, e, r, kappa, omega, gamma; v2_iter = v2_iter, level = level, blocks=blocks)
        end

        return vec(e)
    end

    function SM(r)
        e_vcycle = a_type(zeros(c_type,n+1,m+1,1,blocks))
        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r,n+1,m+1,1,blocks), kappa, omega, gamma; v2_iter = v2_iter, level=3, blocks=blocks, tol=relaxation_tol)
        return vec(e_vcycle)
    end

    x_init = a_type(zeros(c_type,(n+1)*(m+1),blocks))
    ##### just for run-time compilation #####
    # @info "$(Dates.format(now(), "HH:MM:SS")) - before warm-up" 
    # x3,flag3,err3,iter3,resvec3 = fgmres_func(A, vec(r_vcycle), 3, tol=1e-5, maxIter=1,
    #                                                 M=SM, x=vec(x_init), out=-1,flexible=true)
    # x1,flag1,err1,iter1,resvec1 = fgmres_func(A, vec(r_vcycle), 1, tol=1e-5, maxIter=1,
    #                                                 M=M_Unet, x=vec(x3), out=1,flexible=true)
    # @info "$(Dates.format(now(), "HH:MM:SS")) - after warm-up" 
    #########################################
    
    println("before gmres $(typeof(r_vcycle)) $(norm(r_vcycle)) $(norm(x_init))")
    x3,flag3,err3,iter3,resvec3 =@time fgmres_func(A, vec(r_vcycle), 3, tol=solver_tol, maxIter=1,
                                                    M=SM, x=vec(x_init), out=-1,flexible=true)
    println("In CNN solve - number of iterations=$(iter3) err1=$(err3)")

    x1,flag1,err1,iter1,resvec1 =@time fgmres_func(A, vec(r_vcycle), restrt, tol=solver_tol, maxIter=max_iter,
                                                            M=M_Unet, x=vec(x3), out=1,flexible=true)
    
    CSV.write(file_path, DataFrame(Cycle=[param.cycle], FreqIndex=[param.freqIndex], Omega=[omega], Iterations=[iter1], Error=[err1]), delim=';', append=true) 
    
    println("In CNN solve - number of iterations=$(iter1) err1=$(err1)")
    return reshape(x1,(n+1)*(m+1),blocks)|>pu
    
end


function retrain_model(model, base_model_folder, new_model_name, n, m, h, kappa, omega, gamma,
                            set_size, batch_size, iterations, lr;
                            e_vcycle_input=false, v2_iter=10, level=3, threshold=50,
                            axb=false, jac=false, norm_input=false,
                            gmres_restrt=1, σ=elu, blocks=8, relaxation_tol=1e-4)

    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    coefficient = r_type(minimum(h)^2)
    @info "$(Dates.format(now(), "HH:MM:SS")) - Start Re-Train for $(base_model_folder)\\$(new_model_name)"

    X, Y = generate_retrain_random_data(set_size, n, m, h, kappa, omega, gamma;
                                        e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level, threshold=threshold,
                                        axb=axb, jac=jac, norm_input=norm_input, gmres_restrt=gmres_restrt, blocks=blocks)
    
    
    dataset = UnetDatasetFromArray(X,Y)
    @info "$(Dates.format(now(), "HH:MM:SS")) - Generated Data"


    loss!(x, y) = error_loss!(model, x, y)
    loss!(tuple) = loss!(tuple[1], tuple[2])
    
    A(v) = vec(helmholtz_chain!(reshape(v, n+1, m+1, 1, Int64(prod(size(v)) / ((n+1)*(m+1)))), helmholtz_matrix; h=h))
    function SM(r)
        bs = Int64(prod(size(r)) / ((n+1)*(m+1)))
        e_vcycle = a_type(zeros(n+1,m+1,1,bs))
        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r,n+1,m+1,1,bs), kappa, omega, gamma; v2_iter = v2_iter, level=3, blocks=bs, tol=relaxation_tol)
        return vec(e_vcycle)
    end

    opt = RADAM(lr)

    for iteration in 1:iterations
        @info "$(Dates.format(now(), "HH:MM:SS")) - iteration #$(iteration)/$(iterations))"
        println("X size = $(size(dataset.X)) --- Y size = $(size(dataset.Y))")
        data_loader = DataLoader(dataset, batchsize=batch_size, shuffle=true)

        Flux.train!(loss!, Flux.params(model), data_loader, opt)
        loss = dataset_loss!(data_loader, loss!) / size(dataset.X,4)
        println("loss: $(loss)")

        if iteration == iterations
            break
        end

        @info "$(Dates.format(now(), "HH:MM:SS")) - Creating Data - iteration #$(iteration)/$(iterations))"

        rs_vector = a_float_type[]
        es_vector = a_float_type[]
        for (batch_r, batch_e) in data_loader
            num_samples = size(batch_e,4)

            e_model = model(batch_r)
            e_model = (e_model[:,:,1,:] + im*e_model[:,:,2,:]) .* coefficient

            e_model = reshape(e_model, n+1, m+1, 1, num_samples)
            r_model = helmholtz_chain!(e_model, helmholtz_matrix; h=h) 
            # println("batch_r norm = $(norm(batch_r[:,:,1:2,:]))")
            # println("batch_e norm = $(norm(batch_e[:,:,1:2,:]))")
            # println("r_model norm = $(norm(r_model))")
            # println("e_model norm = $(norm(e_model))")
            r_residual = reshape((batch_r[:,:,1,:] + im*batch_r[:,:,2,:]), n+1, m+1, 1, num_samples) - r_model
            e_tilde,flag,err,counter,resvec = fgmres_func(A, vec(r_residual), 3, tol=1e-10, maxIter=1,
                                                    M=SM, x=vec(zeros(c_type, size(e_model))|>gpu), out=-1,flexible=true)
            
            e_tilde = reshape(e_tilde, n+1, m+1, 1, num_samples)
            # println("r_residual norm = $(norm(r_residual))")
            # println("e_tilde norm = $(norm(e_tilde))")
            # Ae_tilde = helmholtz_chain!(e_tilde, helmholtz_matrix; h=h) #.* coefficient 
            # rs = copy(batch_r)
            # rs[:,:,1:2,:] -= complex_grid_to_channels!(Ae_tilde; blocks=num_samples)
            e_tilde = complex_grid_to_channels!(e_tilde; blocks=num_samples) ./ coefficient
            e_model = complex_grid_to_channels!(e_model; blocks=num_samples) ./ coefficient
            
            
            es = e_model .+ e_tilde
            rs = copy(batch_r)
            rs[:,:,1:2,:] += complex_grid_to_channels!(r_residual; blocks=num_samples)
            append!(rs_vector, [rs])
            append!(es_vector, [es])
        end
        # dataset.X, dataset.Y = cat(rs_vector..., dims=4), cat(es_vector..., dims=4)
        dataset.X, dataset.Y = cat(dataset.X, rs_vector..., dims=4), cat(dataset.Y, es_vector..., dims=4)

    end

    return model
end