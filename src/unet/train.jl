using BSON: @save
using MappedArrays
using DelimitedFiles
using MAT

include("losses.jl")

function get_data_x_y(dir_name, set_size, n, m, gamma)

    println("In get_data_x_y set_size = $(set_size)")

    function loadDataFromFile(filename::String)
        file = matopen(filename, "r"); file_data = read(file); close(file);
        return file_data["x"], file_data["y"]

    end
    datadir = dirname(dir_name)
    data = mappedarray(loadDataFromFile, readdir(datadir, join=true))

    x = zeros(r_type, n+1, m+1, 4, set_size)
    y = zeros(r_type, n+1, m+1, 2, set_size)

    for i=1:set_size
        if mod(i,1000) == 0
            @info "$(Dates.format(now(), "HH:MM:SS")) In data point #$(i)/$(set_size)"
        end
        data_i = data[i]
        x[:,:,1:3,i] = data_i[1]
        x[:,:,4,i] = gamma
        y[:,:,:,i] = data_i[2]
    end

    return x, y
end


function train_residual_unet!(model, test_name, n, m, h, kappa, omega, gamma,
                            train_size, test_size, batch_size, iterations, init_lr;
                            e_vcycle_input=true, v2_iter=10, level=3, data_augmentetion=true, kappa_type=1, threshold=50,
                            kappa_input=true, kappa_smooth=false, k_kernel=3, gamma_input=false, kernel=(3,3), smaller_lr=10, axb=false, jac=false, norm_input=false,
                            model_type=SUnet, k_type=NaN, k_chs=-1, indexes=3, data_train_path="", data_test_path="", full_loss=false, residual_loss=false, error_details=false, gmres_restrt=1, σ=elu, same_kappa=false, linear_kappa=true) #, model=NaN)

    @info "$(Dates.format(now(), "HH:MM:SS")) - Start Train $(test_name)"

    train_set_path = generate_random_data!(test_name, train_size, n, m, h, kappa, omega, gamma;
                                                e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level, data_augmentetion =data_augmentetion,
                                                kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, 
                                                k_kernel=k_kernel, axb=axb, jac=jac, norm_input=norm_input, gmres_restrt=gmres_restrt, same_kappa=same_kappa, linear_kappa=linear_kappa, data_folder_type="train")

    test_set_path = generate_random_data!(test_name, test_size, n, m, h, kappa, omega, gamma;
                                                e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level,
                                                kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, 
                                                kappa_smooth=kappa_smooth, k_kernel=k_kernel, axb=axb, jac=jac, norm_input=norm_input, gmres_restrt=gmres_restrt, same_kappa=same_kappa, linear_kappa=linear_kappa, data_folder_type="test")                                           
    
    @info "$(Dates.format(now(), "HH:MM:SS")) - Generated Data"
    # train_size = 16
    # test_size = 16
    if use_gpu == true
        println("after data generation GPU memory status $(CUDA.memory_status())")
    end

    train_set_x, train_set_y = get_data_x_y(train_set_path, train_size, n, m, gamma)
    test_set_x, test_set_y = get_data_x_y(test_set_path, test_size, n, m, gamma)

    if use_gpu == true
        println("after data x_y GPU memory status $(CUDA.memory_status())")
    end

    test_loss = zeros(Int64(iterations/4))|>cpu
    train_loss = zeros(Int64(iterations/4))|>cpu

    CSV.write("models/$(test_name)/train_log/loss.csv", DataFrame(Train=[], Test=[]), delim = ';')
    
    train_data_loader = DataLoader((train_set_x, train_set_y), batchsize=batch_size, shuffle=true)
    test_data_loader = DataLoader((test_set_x, test_set_y), batchsize=batch_size, shuffle=false)

    # boundary_gamma = r_type.(getABL([n+1,m+1], false, [10,10], Float64(1.0)))|>cgpu

    loss!(x, y) = error_loss!(model, x, y; in_tuning=same_kappa)
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
            lr = lr / 2
            opt = RADAM(lr)
            smaller_lr = ceil(Int64,smaller_lr / 2)
            @info "$(Dates.format(now(), "HH:MM:SS")) - Update Learning Rate $(lr) Batch Size $(batch_size)"
        end
        # println("before train alpha = $(model.solve_subnet.alpha)")
        Flux.train!(loss!, Flux.params(model), train_data_loader, opt)
        # println("after train alpha = $(model.solve_subnet.alpha)")
        @info "$(Dates.format(now(), "HH:MM:SS")) - $(iteration))"    

        if mod(iteration,4) == 0
            train_loss[iteration] = dataset_loss!(train_data_loader, loss!) / train_size
            test_loss[iteration] = dataset_loss!(test_data_loader, loss!) / test_size
            CSV.write("models/$(test_name)/train_log/loss.csv", DataFrame(Train=[train_loss[iteration]], Test=[test_loss[iteration]]), delim = ';',append=true)
            @info "$(Dates.format(now(), "HH:MM:SS")) - $(iteration)) Train loss value = $(train_loss[iteration]) , Test loss value = $(test_loss[iteration])"    
        end

        
      
        if mod(iteration,30) == 0
            if use_gpu == true
                println("GPU usage BEFORE saving $(CUDA.available_memory() / 1e9)")
            end
            model = model|>cpu
            @save "models/$(test_name)/model.bson" model
            @info "$(Dates.format(now(), "HH:MM:SS")) - Save Model $(test_name).bson"

            model = model|>cgpu
            if use_gpu==true
                println("GPU usage AFTER saving $(CUDA.available_memory() / 1e9)")
            end
        end
    end

    model = model|>cpu
    @save "models/$(test_name)/model.bson" model
    @info "$(Dates.format(now(), "HH:MM:SS")) - Save Model $(test_name).bson"

    model = model|>cgpu
    return model, train_loss, test_loss
end