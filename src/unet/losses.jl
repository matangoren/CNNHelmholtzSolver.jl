function norm_diff!(x,y)
    return sqrt(sum((x - y).^2) / sum(y.^2))
end

function error_loss!(model, input, output; in_tuning=false,boundary_gamma=nothing)
    return norm_diff!(model(input|>cgpu; in_tuning=false), output|>cgpu)|>cpu
end

function dataset_loss!(dataloader, loss_func)
    loss = 0
    for (x, y) in dataloader
        loss += loss_func(x,y)
    end
    return loss
end