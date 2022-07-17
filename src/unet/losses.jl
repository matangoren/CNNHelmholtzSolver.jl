function norm_diff!(x,y)
    return sqrt(sum((x - y).^2) / sum(y.^2))
end

function error_loss!(model, input, output; in_tuning=false)
    model_result = model(input |> cgpu; in_tuning=in_tuning)
    
    return norm_diff!(model_result, output |> cgpu)
end

function dataset_loss!(dataloader, loss_func)
    loss = 0
    for (x, y) in dataloader
        loss += loss_func(x,y)
    end

    return loss
end