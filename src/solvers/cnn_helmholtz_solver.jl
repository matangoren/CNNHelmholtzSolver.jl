export CnnHelmholtzSolver,getCnnHelmholtzSolver,copySolver,solveLinearSystem,solveLinearSystem!,setupSolver

using jInv.LinearSolvers

# need to add preconditioner type parameters - JU, VU...
mutable struct CnnHelmholtzSolver <: AbstractSolver
    n::Int
    m::Int
    kappa
    omega
    gamma
    cnn_model
    tuning_size::Int
    tuning_iterations::Int
    jacobi_iterations::Int
    nSolve::Int
    solveTime::Real
end

function getCnnHelmholtzSolver(; n=32, m=32, kappa=[], omega=[], gamma=[], model=[], tuning_size=100, tuning_iterations=100, jacobi_iterations=2)
    return CnnHelmholtzSolver(n, m, kappa, omega, gamma, model, tuning_size, tuning_iterations, jacobi_iterations, 0, 0)
end

solveLinearSystem(A,B,param::CnnHelmholtzSolver,doTranspose::Int=0) = solveLinearSystem!(A,B,[],param,doTranspose)

function setupSolver(param::CnnHelmholtzSolver)
    param.kappa = [] # create new initial kappa model
    param.cnn_model = [] # load trained model
    # tune the model before continuing
    return param
end

function solveLinearSystem!(A::SparseMatrixCSC,B,X,param::JuliaSolver,doTranspose=0)
    println("in solveLinearSystem")
end


function clear!(param::CnnHelmholtzSolver)
    param.model = []
    param.kappa = []
end

function copySolver(param::CnnHelmholtzSolver)
    return getCnnHelmholtzSolver(kappa=param.kappa, model=param.model)
end