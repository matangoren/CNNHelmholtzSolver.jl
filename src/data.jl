using CSV, DataFrames
using Distributions: Uniform
using Plots
using DelimitedFiles

# x ← FGMRES(A=Helmholtz, M=V-Cycle, b, x = 0, maxIter = 1)
function generate_vcycle!(n, m, h, kappa, omega, gamma::a_float_type, b::a_type; v2_iter=10, level=3, restrt=1, blocks=1)

    sl_matrix_level3, h_matrix_level3 = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    
    kappa_coarse = down(reshape(kappa, n+1, m+1, 1, 1))[:,:,1,1]
    gamma_coarse = down(reshape(gamma, n+1, m+1, 1, 1))[:,:,1,1]
    sl_matrix_level2, h_matrix_level2 = get_helmholtz_matrices!(kappa_coarse, omega, gamma_coarse; alpha=r_type(0.5))
    
    kappa_coarse = down(reshape(kappa_coarse, Int64((n/2)+1),  Int64((m/2)+1), 1, 1))[:,:,1,1]
    gamma_coarse = down(reshape(gamma_coarse,  Int64((n/2)+1),  Int64((m/2)+1), 1, 1))[:,:,1,1]
    sl_matrix_level1, h_matrix_level1 = get_helmholtz_matrices!(kappa_coarse, omega, gamma_coarse; alpha=r_type(0.5))

    A(v::a_type) = vec(helmholtz_chain!(reshape(v, n+1, m+1, 1, blocks), h_matrix_level3; h=h))
    function M(v)
        v = reshape(v, n+1, m+1,1,blocks)
        x = a_type(zeros(n+1,m+1,1,blocks))
        x, = v_cycle_helmholtz!(n, m, h, x, v, h_matrix_level1, sl_matrix_level1, h_matrix_level2, sl_matrix_level2, h_matrix_level3, sl_matrix_level3;  v2_iter=v2_iter, level=level, blocks=blocks)        
        return vec(x)
    end

    x0 = a_type(zeros(n+1,m+1,1,blocks))
    if restrt == -1
        restrt = rand(1:10)
    end
    x_vcycle, = fgmres_func(A, vec(b), restrt, tol=1e-10, maxIter=1, M=M, x=vec(x0), out=-1, flexible=true)
    x_vcycle_channels = complex_grid_to_channels!(reshape(x_vcycle,n+1,m+1,1,blocks), blocks=blocks)
    return x_vcycle, x_vcycle_channels
end

# x ← FGMRES(A=Helmholtz, M=Jacobi, b, x = 0, maxIter = 1)
function generate_jacobi!(n, m, h, kappa, omega, gamma, b; v2_iter=10, level=3, restrt=1, blocks=1)

    sl_matrix, h_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))

    A(v) = vec(helmholtz_chain!(reshape(v, n+1, m+1, 1, blocks), h_matrix; h=h))
    function M(v)
        v = reshape(v,n+1,m+1,1,blocks)
        x = a_type(zeros(n+1,m+1,1,blocks))
        x = jacobi_helmholtz_method!(n, m, h, x, v, sl_matrix)
        return vec(x)
    end

    x0 = a_type(zeros(c_type,n+1,m+1,1,blocks))
    if restrt == -1
        restrt = rand(1:10)
    end

    x_vcycle, = fgmres_func(A, vec(b), restrt, tol=1e-10, maxIter=1, M=M, x=vec(x0), out=-1, flexible=true)
    x_vcycle_channels = complex_grid_to_channels!(reshape(x_vcycle,n+1,m+1,1,blocks); blocks=blocks)
    return x_vcycle, x_vcycle_channels
end

# r ← Ax - A(FGMRES(A=Helmholtz, M=V-Cycle, b, x = 0, maxIter = 1))
function generate_r_vcycle!(n, m, h, kappa, omega, gamma::a_float_type, x_true::a_type; v2_iter=10, level=3, restrt=1, jac=false, blocks=1)
    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    b_true = helmholtz_chain!(x_true, helmholtz_matrix; h=h)

    if jac == true
        x_vcycle, _ = generate_jacobi!(n, m, h, kappa, omega, gamma, b_true; v2_iter=v2_iter, level=level, restrt=restrt, blocks=blocks)
    else
        x_vcycle, _ = generate_vcycle!(n, m, h, kappa, omega, gamma, b_true; v2_iter=v2_iter, level=level, restrt=restrt, blocks=blocks)
    end

    x_vcycle = reshape(x_vcycle,n+1,m+1,1,blocks)
    e_true = x_true .- x_vcycle
    r_vcycle = b_true .- helmholtz_chain!(x_vcycle, helmholtz_matrix; h=h)

    return r_vcycle, e_true
end


function generate_r_e_batch(n, m, h, kappa, omega, gamma; 
    e_vcycle_input=false, norm_input=false, v2_iter=10, level=3, axb=false, jac=false, gmres_restrt=1, blocks=1)

    x_true = a_type(randn(c_type,n+1,m+1, 1, blocks)) # FIXED!!!

    if axb == true
        # Generate b
        _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
        r_vcycle = helmholtz_chain!(x_true, helmholtz_matrix; h=h)
        e_true = x_true
    else
        # Generate r,e
        r_vcycle, e_true = generate_r_vcycle!(n, m, h, kappa, omega, gamma, x_true;restrt=gmres_restrt, jac=jac, blocks=blocks)
    end

    r_vcycle = r_type(minimum(h)^2) .* r_vcycle
    # check with normalization
    if norm_input == true
        norms_r = mapslices(norm, r_vcycle, dims=[1,2,3])
        r_vcycle = r_vcycle ./ norms_r
        e_true = e_true ./ norms_r
    end

    r_vcycle_channels = complex_grid_to_channels!(r_vcycle; blocks=blocks)
    e_true_channels = complex_grid_to_channels!(e_true; blocks=blocks)

    # Generate e-vcycle
    if e_vcycle_input == true
        e_vcycle, e_vcycle_channels = generate_vcycle!(n, m, h, kappa, omega, gamma, r_vcycle; v2_iter=v2_iter, level=level, blocks=blocks)
        input = cat(e_vcycle_channels, r_vcycle_channels, dims=3)
    else
        input = r_vcycle_channels
    end

    return input, e_true_channels

end

# not needed
function augment_data(data_r, data_e, num_augmented)
    data_size = size(data_r,4)
    augmented_data_r_vector = a_float_type[]
    augmented_data_e_vector = a_float_type[]
    for i=1:num_augmented
        rand1 = rand(1:data_size)
        rand2 = rand(1:data_size)
        r1, e1 = data_r[:,:,:,rand1], data_e[:,:,:,rand2]
        r2, e2 = data_r[:,:,:,rand2], data_e[:,:,:,rand2]
        α = abs(rand(r_type))
        new_r = α*r1 + (1-α)*r2
        new_e = α*e1 + (1-α)*e2
        append!(augmented_data_r_vector, [new_r])
        append!(augmented_data_e_vector, [new_e])
    end

    return cat(data_r, augmented_data_r_vector..., dims=4), cat(data_e, augmented_data_e_vector..., dims=4)
end

function generate_retrain_random_data(data_set_m, n, m, h, kappa, omega, gamma; 
    e_vcycle_input=false, v2_iter=10, level=3, threshold=50,
    axb=false, jac=false, norm_input=false, gmres_restrt=1, blocks=8)

    data_r_vector = a_float_type[]
    data_e_vector = a_float_type[]

    batches_partitions = collect(Iterators.partition(1:data_set_m, blocks))
    kappa_repeated = repeat(kappa.^2,1,1,1,blocks)
    gamma_repeated = repeat(gamma,1,1,1,blocks)
    for part in batches_partitions
        batch_size = length(part)

        r_vcycle_channels, e_true_channels = generate_r_e_batch(n, m, h, kappa, omega, gamma; e_vcycle_input=e_vcycle_input,
                                                    v2_iter=v2_iter, level=level, axb=axb, jac=jac, blocks=batch_size)
        
        if batch_size != blocks
            kappa_repeated = repeat(kappa.^2,1,1,1,batch_size)
            gamma_repeated = repeat(gamma,1,1,1,batch_size)
        end

        append!(data_r_vector, [cat(r_vcycle_channels, kappa_repeated, gamma_repeated, dims=3)])
        append!(data_e_vector, [e_true_channels])
    end
    
    return cat(data_r_vector..., dims=4), cat(data_e_vector..., dims=4)
end

function generate_random_data!(test_name, data_set_m, n, m, h, kappa, omega, gamma; e_vcycle_input=true, v2_iter=10, level=3, data_augmentetion=false,
                                                          kappa_type=1, threshold=50, kappa_input=true, kappa_smooth=false, k_kernel=3, axb=false, 
                                                          jac=false, norm_input=false, gmres_restrt=1, same_kappa=false, data_folder_type="train")
    
    data_dirname = "datasets/$(test_name)/$(data_folder_type)"
    println("$(data_folder_type) data_set_m = $(data_set_m)")
    
    if isdir(data_dirname)
        println("using previous data :)")
        return data_dirname
    end
    mkpath(data_dirname)

    for i = 1:data_set_m

        if same_kappa == false
            kappa, c = get2DSlownessLinearModel(n,m; normalized=true)|>cgpu
        end

        # blocks has to be 1 - because kappa is different for each sample in the batch
        input, e_true_channels = generate_r_e_batch(n, m, h, kappa, omega, gamma; e_vcycle_input=e_vcycle_input,
                                                    v2_iter=v2_iter, level=level, axb=axb, jac=jac, blocks=1)

        
        input = kappa_input == true ? cat(input, reshape(kappa.^2, n+1, m+1, 1, 1), dims=3) : input
        
        
        data_file = matopen("$(data_dirname)/sample_$(i).mat", "w");
        write(data_file, "x", input|>cpu)
        write(data_file, "y", e_true_channels|>cpu)
        close(data_file);

        # Data Augmentetion
        # if data_augmentetion == true && mod(i,3) == 0
        #     (input_2,e_2) = dataset[rand(1:size(dataset,1))]
        #     r_index = e_vcycle_input == true ? 3 : 1
        #     r_2 = input_2[:,:,r_index:r_index+1,:]

        #     scalar = abs(rand(r_type))
        #     r_t = scalar*r_vcycle_channels+(1-scalar)*r_2
        #     scale = (scalar*norm(r_vcycle_channels) + (1-scalar)*norm(r_2))/norm(r_t)

        #     input_t = (scalar*input+(1-scalar)*input_2)*scale
        #     e_t = (scalar*e_true_channels+(1-scalar)*e_2)*scale
        #     append!(dataset,[(input_t |> pu, e_t |> pu)])
        # end
    end
    
    return data_dirname
end

function get_csv_set!(path, data_set_m)
    println("In get_csv_set $(path)")
    df = CSV.File(path)|> DataFrame(x = Array{r_type, 4}[], y = Array{r_type, 4}[])
    dataset = Tuple[]
    println(df.x[1])
    for i = 1:data_set_m
        append!(dataset,[(df.x[i], df.y[i])])
    end
    return dataset
end



function get2DVelocityLinearModel(n::Int, m::Int; top_lb=1.65, top_ub=1.75, bottom_lb=2.5, bottom_ub=3.5, absorbing_val=1.5)
    velocity_model = zeros(r_type, n+1, m+1)
    # adding sea layers
    num_layers = rand(2:7)
    velocity_model[:,1:num_layers] .= absorbing_val


    top_val = rand(Uniform(top_lb, top_ub))
    bottom_val = rand(Uniform(bottom_lb, bottom_ub))
    linear_model = (range(top_val,stop=bottom_val,length=m+1-num_layers)) * ((ones(n+1)'))

    velocity_model[:,num_layers+1:end] = linear_model'
    
    return velocity_model
end

function get2DSlownessLinearModel(n::Int, m::Int; mref_file="", top_lb=1.65, top_ub=1.75, bottom_lb=2.5, bottom_ub=3.5, absorbing_val=1.5, normalized=false)
    velocity_model = get2DVelocityLinearModel(n,m)
    slowness_model = r_type.(1.0./(velocity_model.+1e-16))

    c = r_type(1.0)
    if normalized
        c = maximum(slowness_model)
        slowness_model = slowness_model ./ c
    end
    return slowness_model, c
end

velocityToSlowSquared(v::Array) = (1.0./(v.+1e-16)).^2

