function norm_diff!(x,y)
    return sqrt(sum((x - y).^2) / sum(y.^2))
end

function error_loss!(model, input::CuArray, output::CuArray)
    return norm_diff!(model(input; in_tuning=false), output)|>cpu
end

function error_loss!(model, input::Array, output::Array)
    return norm_diff!(model(input|>cgpu; in_tuning=false), output|>cgpu)|>cpu
end

function dataset_loss!(dataloader, loss_func)
    loss = 0
    for (x, y) in dataloader
        loss += loss_func(x,y)
    end
    return loss
end