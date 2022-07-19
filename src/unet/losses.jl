function norm_diff!(x,y)
    return sqrt(sum((x - y).^2) / sum(y.^2))
end

# function error_loss!(model, input, output; in_tuning=false)

#     input = input|>cgpu
#     output = output|>cgpu
#     model_result = model(input; in_tuning=in_tuning)
#     diff = norm_diff!(model_result, output)
#     input = input|>pu
#     output = output|>pu

#     return diff
# end
function error_loss!(model, input, output; in_tuning=false)
    model_result = model(input|>cgpu; in_tuning=false)
    return norm_diff!(model_result, output|>cgpu)
end

function dataset_loss!(dataloader, loss_func)
    loss = 0
    for (x, y) in dataloader
        loss += loss_func(x,y)
    end

    return loss
end