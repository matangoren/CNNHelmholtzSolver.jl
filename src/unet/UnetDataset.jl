# using MappedArrays
# using DelimitedFiles
# using MAT

# struct UnetDataset
#     data::ReadonlyMappedArray
#     gamma
# end

# # function loadDataFromFile(filename::String)
# #     file = matopen(filename, "r"); file_data = read(file); close(file);
# #     return file_data["x"], file_data["y"]
# # end

# function getUnetDataset(dir_name::String, gamma; entry_type="x")
#     datadir = dirname(dir_name)

#     function loadDataFromFile(filename::String)
#         file = matopen(filename, "r"); file_data = read(file); close(file);
#         # return entry_type == "x" ? cat(file_data["x"],gamma|>cpu; dims=3) : file_data["y"]
#     end

#     data = mappedarray(loadDataFromFile, readdir(datadir, join=true))
#     return UnetDataset(data, gamma)
# end

# function Base.getindex(ud::UnetDataset, k)
#     return cat(ud.data[k]...;dims=4)
# end

# function Base.length(ud::UnetDataset)
#     return length(ud.data)
# end


######################## NEW UNET DATASET ########################
# struct Dataset{T, N} <: AbstractArray{T, N}
#     data::Array{Tuple{Array,Array}}
#     gamma::Union{Array,Nothing}
# end

# Dataset{T}(data, gamma) where {T} =
#     Dataset{T, 1}(data, gamma)


# function Base.getindex(d::Dataset{T}, i::Int) where {T}
#     data_i = d.data[i]
#     x = data_i[1]
#     y = data_i[2]
#     if d.gamma !== nothing
#         x = cat(x, d.gamma, dims=3)
#     end
#     return x, y
# end

# function Base.getindex(d::Dataset{T}, ids::Array) where {T}
#     x, y = d[ids[1]]
#     xs_last_dim = ntuple(i -> Colon(), ndims(x)-1)
#     ys_last_dim = ntuple(i -> Colon(), ndims(y)-1)

#     xs = Array{T}(undef, size(x,1), size(x,2), size(x,3), length(ids))
#     ys = Array{T}(undef, size(y,1), size(y,2), size(y,3), length(ids))

#     xs[xs_last_dim..., 1] .= x
#     ys[ys_last_dim..., 1] .= y

#     for (i, id) in enumerate(ids[2:end])
#         x, y = d[id]
#         xs[xs_last_dim..., i + 1] .= x
#         ys[ys_last_dim..., i + 1] .= y
#     end
#     xs, ys
# end

# Base.IndexStyle(::Type{Dataset}) = IndexLinear()
# Base.size(d::Dataset) = (length(d.data),)
# Base.length(d::Dataset) = length(d.data)