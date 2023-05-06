module CNNHelmholtzSolver

include("../test/test_intro.jl")
include("./solvers/cnn_helmholtz_solver.jl")

export UNetUpBlock, FFSDNUnet, FFKappa, TFFKappa, TSResidualBlockI, SResidualBlock, FeaturesUNet 
end
