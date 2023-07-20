struct UnetDatasetFromDirectory
    data_path::String
    dataset_size::Int
end

function Base.getindex(d::UnetDatasetFromDirectory, i::Int)
    file = matopen("$(d.data_path)/sample_$(i).mat", "r"); data = read(file); close(file);
    return data["x"], data["y"]
end

function Base.getindex(d::UnetDatasetFromDirectory, ids::Array)
    x, y = d[ids[1]]

    xs = Array{r_type}(undef, size(x,1), size(x,2), size(x,3), length(ids))
    ys = Array{r_type}(undef, size(y,1), size(y,2), size(y,3), length(ids))

    xs[:,:,:, 1] .= x
    ys[:,:,:, 1] .= y

    for (i, id) in enumerate(ids[2:end])
        x, y = d[id]
        xs[:,:,:, i + 1] .= x
        ys[:,:,:, i + 1] .= y
    end
    xs, ys
end

Base.IndexStyle(::Type{UnetDatasetFromDirectory}) = IndexLinear()
Base.size(d::UnetDatasetFromDirectory) = (d.dataset_size,)
Base.length(d::UnetDatasetFromDirectory) = d.dataset_size


# this dataset is used for re-training
mutable struct UnetDatasetFromArray
    X
    Y
end

function Base.getindex(d::UnetDatasetFromArray, i::Int)
    return d.X[:,:,:,i], d.Y[:,:,:,i]
end

function Base.getindex(d::UnetDatasetFromArray, ids::Array)
    batch_size = length(ids)
    paired_ids = rand(1:size(d.X,4), batch_size)
    # alphas = rand(r_type, batch_size)
    # alphas = ones(r_type, batch_size)

    xs = a_float_type[]
    ys = a_float_type[]

    for (i, id) in enumerate(ids[1:end])
        x, y = d[id]
        # x_p, y_p = d[paired_ids[i]]
        # append!(xs, [alphas[i]*x + (1-alphas[i])*x_p])
        append!(xs, [x])
        # append!(ys, [alphas[i]*y + (1-alphas[i])*y_p])
        append!(ys, [y])
    end
    cat(xs...,dims=4), cat(ys...,dims=4)
end

Base.IndexStyle(::Type{UnetDatasetFromArray}) = IndexLinear()
Base.size(d::UnetDatasetFromArray) = (size(d.X,4),)
Base.length(d::UnetDatasetFromArray) = size(d.X,4)