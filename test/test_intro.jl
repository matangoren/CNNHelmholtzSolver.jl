using Statistics
using DelimitedFiles
using LinearAlgebra
using Flux
using Flux: @functor
using Flux.Data: DataLoader
using LaTeXStrings
using KrylovMethods
using Distributions: Normal
using BSON
using BSON: @load
using BSON: @save
# using Plots
using PyPlot
using Dates
using CSV, DataFrames
using Random
using MAT
using SparseArrays
using jInv.LinearSolvers
using Helmholtz
using jInv.Mesh


pu = cpu # pu is the processing unit of the software calling the solver (in FWI case - cpu)
r_type = Float32
c_type = ComplexF32
u_type = Float32
gmres_type = ComplexF32

use_gpu = true

if use_gpu == true
    using CUDA
    CUDA.allowscalar(true)
    cgpu = gpu
    a_type = CuArray{c_type}
    a_float_type = CuArray{r_type}
else
    cgpu = cpu
    a_type = Array{c_type}
    a_float_type = Array{r_type}
    pyplot()
end
