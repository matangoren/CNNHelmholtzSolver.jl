mutable struct UnetDatasetFromDirectory
    data_path::String
    dataset_size::Int
    gamma::Array # make sure it's on CPU
end

function Base.getindex(d::UnetDatasetFromDirectory, i::Int)
    file = matopen("$(d.data_path)/sample_$(i).mat", "r"); data = read(file); close(file);
    return cat(data["x"], d.gamma, dims=3), data["y"]
end

function Base.getindex(d::UnetDatasetFromDirectory, ids::Union{Array,UnitRange})
    xs = a_float_type[]
    ys = a_float_type[]

    for id in ids
        x, y = d[id]
        append!(xs, [x|>cgpu])
        append!(ys, [y|>cgpu])
    end
    cat(xs...,dims=4), cat(ys...,dims=4)
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

function Base.getindex(d::UnetDatasetFromArray, ids::Union{Array,UnitRange})
    batch_size = length(ids)
    paired_ids = rand(1:size(d.X,4), batch_size)
    # alphas = rand(r_type, batch_size)

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