module CNNHelmholtzSolver

include("../test/test_intro.jl")
include("flux_components.jl")
include("./solvers/cnn_helmholtz_solver.jl")

export up, down
export UNetUpBlock, FFSDNUnet, FFKappa, TFFKappa, TSResidualBlockI, SResidualBlock, FeaturesUNet 
end
