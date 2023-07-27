using BSON: @save

include("training_utils.jl")
include("losses.jl")
include("CustomDatasets.jl")


function train_residual_unet!(model, test_name, n, m, h, kappa, omega, gamma,
                            train_size, test_size, batch_size, iterations, init_lr;
                            e_vcycle_input=true, v2_iter=10, level=3, data_augmentetion=true, kappa_type=1, threshold=50,
                            kappa_input=true, kappa_smooth=false, k_kernel=3, gamma_input=false, kernel=(3,3), smaller_lr=10, axb=false, jac=false, norm_input=false,
                            model_type=SUnet, k_type=NaN, k_chs=-1, indexes=3, data_train_path="", data_test_path="", full_loss=false, residual_loss=false, error_details=false, gmres_restrt=1, σ=elu, same_kappa=false, linear_kappa=true) #, model=NaN)

    @info "$(Dates.format(now(), "HH:MM:SS")) - Start Train $(test_name)"

    train_set_path = generate_random_data!(test_name, train_size, n, m, h, kappa, omega, gamma;
                                                e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level, data_augmentetion =data_augmentetion,
                                                kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, 
                                                k_kernel=k_kernel, axb=axb, jac=jac, norm_input=norm_input, gmres_restrt=gmres_restrt, same_kappa=same_kappa, data_folder_type="train")

    test_set_path = generate_random_data!(test_name, test_size, n, m, h, kappa, omega, gamma;
                                                e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level,
                                                kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, 
                                                kappa_smooth=kappa_smooth, k_kernel=k_kernel, axb=axb, jac=jac, norm_input=norm_input, gmres_restrt=gmres_restrt, same_kappa=same_kappa, data_folder_type="test")                                           
    @info "$(Dates.format(now(), "HH:MM:SS")) - Generated Data"
    
    if use_gpu == true
        println("after data generation GPU memory status $(CUDA.memory_status())")
    end

    # train_set_x, train_set_y = get_data_x_y(train_set_path, train_size, n, m, gamma|>cpu)
    # test_set_x, test_set_y = get_data_x_y(test_set_path, test_size, n, m, gamma|>cpu)
    train_dataset = UnetDatasetFromDirectory(train_set_path, train_size, gamma|>cpu)
    test_dataset = UnetDatasetFromDirectory(test_set_path, test_size, gamma|>cpu)

    if use_gpu == true
        println("after data x_y GPU memory status $(CUDA.memory_status())")
    end

    train_loss = zeros(Int64(iterations/3))|>cpu # not used for now
    test_loss = zeros(Int64(iterations/3))|>cpu

    CSV.write("models/$(test_name)/train_log/loss.csv", DataFrame(Train=[], Test=[]), delim = ';')
    
    # train_data_loader = DataLoader((train_set_x, train_set_y), batchsize=batch_size, shuffle=true)
    train_data_loader = DataLoader(train_dataset, batchsize=batch_size, shuffle=true)
    test_data_loader = DataLoader(test_dataset, batchsize=batch_size, shuffle=false)

    
    loss!(x, y) = error_loss!(model, x, y)
    loss!(tuple) = loss!(tuple[1], tuple[2])

    # Start model training
    lr = init_lr
    opt = RADAM(lr)

    for iteration in 1:iterations
        println("===== iteration #$(iteration)/$(iterations) =====")
        if use_gpu == true
            println("GPU memory status $(CUDA.memory_status())")
        end
        if mod(iteration,smaller_lr) == 0
            # lr = max(lr / 10, 1e-6)
            lr = lr / 2
            opt = RADAM(lr)
            smaller_lr = ceil(Int64,smaller_lr / 2)
            @info "$(Dates.format(now(), "HH:MM:SS")) - Update Learning Rate $(lr)"
        end
        # if iteration > 140
        #  lr = 1e-6
        #  # opt = RADAM(lr) # forgot to use this lr
         
        # end
        if iteration > 0
            println("Training")
            Flux.train!(loss!, Flux.params(model), train_data_loader, opt)

            @info "$(Dates.format(now(), "HH:MM:SS")) - $(iteration))"    

            if mod(iteration, 3) == 0 # just to save some run-time :)
                test_loss[Int64(iteration/3)] = dataset_loss!(test_data_loader, loss!) / test_size
                CSV.write("models/$(test_name)/train_log/loss.csv", DataFrame(Test=[test_loss[Int64(iteration/3)]]), delim = ';',append=true)
                @info "$(Dates.format(now(), "HH:MM:SS")) - $(iteration)) Test loss value = $(test_loss[Int64(iteration/3)])"       
            end
            
        
            if mod(iteration,10) == 0
                model = model|>cpu
                @save "models/$(test_name)/model.bson" model
                @info "$(Dates.format(now(), "HH:MM:SS")) - Save Model $(test_name).bson"
                model = model|>cgpu
            end
        end
             
    end

    model = model|>cpu
    @save "models/$(test_name)/model.bson" model
    @info "$(Dates.format(now(), "HH:MM:SS")) - Save Model $(test_name).bson"

    model = model|>cgpu
    return model, train_loss, test_loss
end