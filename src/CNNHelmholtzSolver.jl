module CNNHelmholtzSolver
# ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
include("../test/test_intro.jl")
include("solvers/cnn_helmholtz_solver.jl")

# export up, down
export UNetUpBlock, FFSDNUnet, FFKappa, TFFKappa, TSResidualBlockI, SResidualBlock, FeaturesUNet 
end
