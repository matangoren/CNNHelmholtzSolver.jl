using CSV, DataFrames
using Distributions: Uniform
using Plots
using DelimitedFiles

# x ← FGMRES(A=Helmholtz, M=V-Cycle, b, x = 0, maxIter = 1)
function generate_vcycle!(n, m, h, kappa, omega, gamma, b; v2_iter=10, level=3, restrt=1)

    sl_matrix_level3, h_matrix_level3 = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    kappa_coarse = down(reshape(kappa, n+1, m+1, 1, 1)|>pu)[:,:,1,1]
    gamma_coarse = down(reshape(gamma, n+1, m+1, 1, 1)|>pu)[:,:,1,1]
    sl_matrix_level2, h_matrix_level2 = get_helmholtz_matrices!(kappa_coarse, omega, gamma_coarse; alpha=r_type(0.5))
    kappa_coarse = down(reshape(kappa_coarse, Int64((n/2)+1),  Int64((m/2)+1), 1, 1)|>pu)[:,:,1,1]
    gamma_coarse = down(reshape(gamma_coarse,  Int64((n/2)+1),  Int64((m/2)+1), 1, 1)|>pu)[:,:,1,1]

    sl_matrix_level1, h_matrix_level1 = get_helmholtz_matrices!(kappa_coarse, omega, gamma_coarse; alpha=r_type(0.5))

    A(v::a_type) = vec(helmholtz_chain!(reshape(v, n+1, m+1, 1, 1), h_matrix_level3; h=h))
    function M(v::a_type)
        v = reshape(v, n+1, m+1)
        x = zeros(c_type,n+1,m+1)|>pu
        x, = v_cycle_helmholtz!(n, m, h, x, v, h_matrix_level1, sl_matrix_level1, h_matrix_level2, sl_matrix_level2, h_matrix_level3, sl_matrix_level3;  v2_iter=v2_iter, level=level)
        return vec(x)
    end

    x0 = zeros(c_type,n+1,m+1,1,1)|>pu
    if restrt == -1
        restrt = rand(1:10)
    end
    x_vcycle, = fgmres_func(A, vec(b), restrt, tol=1e-10, maxIter=1, M=M, x=vec(x0), out=-1, flexible=true)
    x_vcycle_channels = complex_grid_to_channels!(x_vcycle)
    return x_vcycle, x_vcycle_channels
end

# x ← FGMRES(A=Helmholtz, M=Jacobi, b, x = 0, maxIter = 1)
function generate_jacobi!(n, m, h, kappa, omega, gamma, b; v2_iter=10, level=3, restrt=1)

    sl_matrix, h_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))

    A(v::a_type) = vec(helmholtz_chain!(reshape(v, n+1, m+1, 1, 1), h_matrix; h=h))
    function M(v::a_type)
        v = reshape(v, n+1, m+1)
        x = zeros(c_type,n+1,m+1)|>pu
        x = jacobi_helmholtz_method!(n, m, h, x, v, sl_matrix)
        return vec(x)
    end

    x0 = zeros(c_type,n+1,m+1,1,1)|>pu
    if restrt == -1
        restrt = rand(1:10)
    end

    x_vcycle, = fgmres_func(A, vec(b), restrt, tol=1e-10, maxIter=1,
                                                    M=M, x=vec(x0), out=-1, flexible=true)
    x_vcycle_channels = complex_grid_to_channels!(x_vcycle)
    return x_vcycle, x_vcycle_channels
end

# r ← Ax - A(FGMRES(A=Helmholtz, M=V-Cycle, b, x = 0, maxIter = 1))
function generate_r_vcycle!(n, m, h, kappa, omega, gamma, x_true; v2_iter=10, level=3, restrt=1, jac=false)

    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    b_true = helmholtz_chain!(x_true, helmholtz_matrix; h=h)

    if jac == true
        x_vcycle, _ = generate_jacobi!(n, m, h, kappa, omega, gamma, b_true; v2_iter=v2_iter, level=level, restrt=restrt)
    else
        x_vcycle, _ = generate_vcycle!(n, m, h, kappa, omega, gamma, b_true; v2_iter=v2_iter, level=level, restrt=restrt)
    end
    x_vcycle = reshape(x_vcycle, n+1, m+1, 1, 1)
    e_true = x_true .- x_vcycle
    r_vcycle = b_true .- helmholtz_chain!(x_vcycle, helmholtz_matrix; h=h)

    return r_vcycle, e_true
end

function generate_random_data!(test_name, data_set_m, n, m, h, kappa, omega, gamma; e_vcycle_input=true, v2_iter=10, level=3, data_augmentetion=false,
                                                          kappa_type=1, threshold=50, kappa_input=true, kappa_smooth=false, k_kernel=3, axb=false, jac=false, norm_input=false, gmres_restrt=1, same_kappa=false, linear_kappa=true, data_folder_type="train")
    
    data_dirname = "datasets/$(test_name)/$(data_folder_type)"
    println("$(data_folder_type) data_set_m = $(data_set_m)")
    
    if isdir(data_dirname)
        if data_folder_type == "retrain"
            rm(data_dirname)
        else
            println("using previous data :)")
            return "$(data_dirname)/"
        end
    end
    mkpath(data_dirname)

    # dataset = Tuple[]
    # data_set_m = data_augmentetion == true ? floor(Int32,0.75*data_set_m) : data_set_m
    for i = 1:data_set_m

        if same_kappa == false
            kappa, c = a_type(get2DSlownessLinearModel(n,m; normalized=true))
        end
        
        # Generate Random Sample
        x_true = a_type(randn(c_type,n+1,m+1, 1, 1))

        if axb == true
            # Generate b
            _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
            r_vcycle = helmholtz_chain!(x_true, helmholtz_matrix; h=h)
            e_true = x_true
        else
            # Generate r,e
            r_vcycle, e_true = generate_r_vcycle!(n, m, h, kappa, omega, gamma, x_true;restrt=gmres_restrt, jac=jac)
        end

        r_vcycle = r_type(minimum(h)^2) .* r_vcycle
        if norm_input == true
            norm_r = r_type(norm(r_vcycle))
            r_vcycle = r_vcycle ./ norm_r
            e_true = e_true ./ norm_r
        end

        # Print scale information
        if mod(i,1000) == 0
            if axb == true
                @info "$(Dates.format(now(), "HH:MM:SS")) - i = $(i) norm b = $(norm(r_vcycle)) norm x = $(norm(e_true))"
            else
                @info "$(Dates.format(now(), "HH:MM:SS")) - i = $(i) norm r = $(norm(r_vcycle)) norm e = $(norm(e_true))"
            end
        end

        r_vcycle_channels = complex_grid_to_channels!(r_vcycle)
        e_true_channels = complex_grid_to_channels!(e_true)

        # Generate e-vcycle
        if e_vcycle_input == true
            e_vcycle, e_vcycle_channels = generate_vcycle!(n, m, h, kappa, omega, gamma, r_vcycle; v2_iter=v2_iter, level=level)
            input = cat(e_vcycle_channels, r_vcycle_channels, dims=3)
        else
            input = r_vcycle_channels
        end

        input = kappa_input == true ? cat(input, reshape(kappa.^2, n+1, m+1, 1, 1), dims=3) : input
        
        # append!(dataset,[(input, e_true_channels)])


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

    return "$(data_dirname)/"
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
    #adding sea layers
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

