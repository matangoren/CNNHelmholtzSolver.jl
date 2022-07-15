import DataLoaders.LearnBase: getobs, nobs

struct UnetDataset
    data::Vector{Tuple}
    gamma::Array{Float64}
end

# @functor UnetDataset
# UnetDataset(dataset, gamma::Array{Float64}) = UnetDataset(dataset,gamma)

length(dataset::UnetDataset) = length(dataset.data)
getobs(dataset::UnetDataset, i::Int) = cat(dims=3, dataset.data[i][1], gamma), dataset.data[i][2]