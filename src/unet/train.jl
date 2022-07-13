using BSON: @save

include("losses.jl")

function get_data_x_y(dataset, n, m, gamma)
    x = zeros(r_type, n+1, m+1, 4, size(dataset,1)) |> pu
    y = zeros(r_type, n+1, m+1, 2, size(dataset,1)) |> pu

    for i=1:size(dataset,1)
        x[:,:,1:3,i] = dataset[i][1]
        x[:,:,4,i] = gamma # Eran said to train without gamma
        y[:,:,:,i] = dataset[i][2]
    end

    return x, y
end


function train_residual_unet!(model, test_name, n, m, h, f, kappa, omega, gamma,
                            train_size, test_size, batch_size, iterations, init_lr;
                            e_vcycle_input=true, v2_iter=10, level=3, data_augmentetion=true, kappa_type=1, threshold=50,
                            kappa_input=true, kappa_smooth=false, k_kernel=3, gamma_input=false, kernel=(3,3), smaller_lr=10, axb=false, jac=false, norm_input=false,
                            model_type=SUnet, k_type=NaN, k_chs=-1, indexes=3, data_train_path="", data_test_path="", full_loss=false, residual_loss=false, error_details=false, gmres_restrt=1, σ=elu, in_tuning=false, linear_kappa=true) #, model=NaN)

    @info "$(Dates.format(now(), "HH:MM:SS")) - Start Train $(test_name)"

    if data_train_path != ""
        train_set = get_csv_set!(data_train_path, train_size, n, m, h)
    else
        train_set = generate_random_data!(train_size, n, m, h, kappa, omega, gamma;
                                                e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level, data_augmentetion =data_augmentetion,
                                                kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, k_kernel=k_kernel, axb=axb, jac=jac, norm_input=norm_input, gmres_restrt=gmres_restrt, same_kappa=in_tuning, linear_kappa=linear_kappa)
    end
    if data_test_path != ""
        test_set = get_csv_set!(data_test_path, test_size, n, m, h)
    else
        test_set = generate_random_data!(test_size, n, m, h, kappa, omega, gamma;
                                                e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level,
                                                kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, k_kernel=k_kernel, axb=axb, jac=jac, norm_input=norm_input, gmres_restrt=gmres_restrt, same_kappa=in_tuning, linear_kappa=linear_kappa)
    end
    @info "$(Dates.format(now(), "HH:MM:SS")) - Generated Data"
    mkpath("models")

    if use_gpu == true
        println("AFTER DATA GENERATION $(CUDA.available_memory() / 1e9)")
    end
    train_set_x, train_set_y = get_data_x_y(train_set, n, m, gamma)
    test_set_x, test_set_y = get_data_x_y(train_set, n, m, gamma)
    if use_gpu == true
        println("AFTER DATA x_y $(CUDA.available_memory() / 1e9)")
    end
    batchs = floor(Int64,train_size / batch_size) # (batch_size*10))
    test_loss = zeros(iterations)
    train_loss = zeros(iterations) 

    CSV.write("$(test_name) loss.csv", DataFrame(Train=[], Test=[]), delim = ';')
    
    errors_count = 4
    if errors_count > 1 && full_loss == true
        CSV.write("test/unet/results/$(test_name) loss.csv", DataFrame(Train=[],Train_U1=[],Train_U2=[],Train_J2=[],Test=[]), delim = ';')

    end
    if residual_loss == true
        CSV.write("$(test_name) loss.csv", DataFrame(Train=[], Residual=[], Error=[], Test=[]), delim = ';')
    end

    loss!(x, y) = error_loss!(model, x, y; in_tuning=in_tuning)
    loss!(tuple) = loss!(tuple[1], tuple[2])
    
    # Start model training
    lr = init_lr
    opt = RADAM(lr)

    train_data_loader = DataLoader((train_set_x, train_set_y), batchsize=batch_size, shuffle=true)
    test_data_loader = DataLoader((test_set_x, test_set_y), batchsize=batch_size, shuffle=false)

    for iteration in 1:iterations
        println("===== iteration #$(iteration)/$(iterations) =====")
        if use_gpu == true
            println("GPU usage $(CUDA.available_memory() / 1e9)")
        end
        if mod(iteration,smaller_lr) == 0
            lr = lr / 2
            opt = RADAM(lr)
            batch_size = min(batch_size * 2,512)
            smaller_lr = ceil(Int64,smaller_lr / 2)
            @info "$(Dates.format(now(), "HH:MM:SS")) - Update Learning Rate $(lr) Batch Size $(batch_size)"
        end
        

        Flux.train!(loss!, Flux.params(model), train_data_loader, opt)
        
        train_loss[iteration] = dataset_loss!(train_data_loader, loss!) / size(train_set,1)
        test_loss[iteration] = dataset_loss!(test_data_loader, loss!) / size(test_set,1)

        CSV.write("$(test_name) loss.csv", DataFrame(Train=[train_loss[iteration]], Test=[test_loss[iteration]]), delim = ';',append=true)
        @info "$(Dates.format(now(), "HH:MM:SS")) - $(iteration)) Train loss value = $(train_loss[iteration]) , Test loss value = $(test_loss[iteration])"

        if mod(iteration,30) == 0
            if use_gpu == true
                println("GPU usage BEFORE saving $(CUDA.available_memory() / 1e9)")
            end
            model = model|>cpu
            @save "models/$(test_name).bson" model
            @info "$(Dates.format(now(), "HH:MM:SS")) - Save Model $(test_name).bson"

            model = model|>cgpu
            if use_gpu==true
                println("GPU usage AFTER saving $(CUDA.available_memory() / 1e9)")
            end
        end
    end

    model = model|>cpu
    @save "models/$(test_name).bson" model
    @info "$(Dates.format(now(), "HH:MM:SS")) - Save Model $(test_name).bson"

    model = model|>cgpu
    return model, train_loss, test_loss
end