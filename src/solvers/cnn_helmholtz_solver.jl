export CnnHelmholtzSolver,getCnnHelmholtzSolver,solveLinearSystem,copySolver,setMediumParameters,setSolverType

model_name = "without_alpha"
println(joinpath(@__DIR__, "results/$(model_name)"))
println(joinpath(pwd(), "results/$(model_name)"))

mkpath(joinpath(pwd(), "results/$(model_name)"))

mutable struct CnnHelmholtzSolver<: AbstractSolver
    solver_type::Dict
    n
    m
    h
    kappa
    omega
    gamma
    model
    model_parameters
    kappa_features
    tuning_size
    tuning_iterations
    doClear
    solver_tol
    relaxation_tol
    cycle
    freqIndex
end

include("../unet/model.jl")
include("../data.jl")
include("solver_utils.jl")

function getCnnHelmholtzSolver(solver_name; n=128, m=128,h=[], kappa=[], omega=[], gamma=[], model=[], model_parameters=Dict(), kappa_features=[], tuning_size=100, tuning_iterations=100, solver_tol=1e-8, relaxation_tol=1e-4)
    if model == []
        model, model_parameters = setupSolver()
    end
    return CnnHelmholtzSolver(get_solver_type(solver_name), n, m, h, kappa, omega, gamma, model, model_parameters, kappa_features, tuning_size, tuning_iterations, 0, solver_tol, relaxation_tol,0,0)
end

function getCnnHelmholtzSolver(solver_type::Dict; n=128, m=128,h=[], kappa=[], omega=[], gamma=[], model=[], model_parameters=Dict(), kappa_features=[], tuning_size=100, tuning_iterations=100, solver_tol=1e-8, relaxation_tol=1e-4)
    if model == []
        model, model_parameters = setupSolver()
    end
    return CnnHelmholtzSolver(solver_type, n, m, h, kappa, omega, gamma, model, model_parameters, kappa_features, tuning_size, tuning_iterations, 0, solver_tol, relaxation_tol,0,0)
end

# need only B - the rhs of the linear equation. The rest of the computations is done by the CNN model.
import jInv.LinearSolvers.solveLinearSystem;
function solveLinearSystem(A,B,param::CnnHelmholtzSolver,doTranspose::Int=0)
    return solveLinearSystem!(A,B,[],param,doTranspose)
end

function setMediumParameters(param::CnnHelmholtzSolver, Helmholtz_param::HelmholtzParam)    
    param.n, param.m = Helmholtz_param.Mesh.n
    param.h = (Helmholtz_param.Mesh.h)|>cgpu
    param.gamma = a_float_type(reshape(Helmholtz_param.gamma,param.n+1,param.m+1))
    
    slowness = r_type.(reshape(sqrt.(Helmholtz_param.m),param.n+1,param.m+1)) # slowness (m from FWI is slowness squared)
    c = r_type(maximum(slowness))
    omega_exact = r_type((0.1*2*pi) / (c*maximum(param.h)))
    
    param.omega = omega_exact * c
    param.kappa = a_float_type(slowness .* (Helmholtz_param.omega/(omega_exact*c))) # normalized slowness * w_fwi/w_exact

    param.kappa_features = Base.invokelatest(get_kappa_features, param)

    return param
end

function setSolverType(solver_name::String, param::CnnHelmholtzSolver)
    param.solver_type = get_solver_type(solver_name)
    return param
end

import jInv.LinearSolvers.solveLinearSystem!;
function solveLinearSystem!(A::SparseMatrixCSC,B,X,param::CnnHelmholtzSolver,doTranspose=0)
    println("in solveLinearSystem")
    if param.model == []
        param = setupSolver!(param)
    end

    if doTranspose == 1
        B = real(B) - im*imag(B) # negate the imaginary part of B (rhs)
    end

    res = Base.invokelatest(solve, param, B|>cgpu, 10, 30)
    
    if doTranspose == 1
        # negate the imaginary part of res
        res = real(res) - im*imag(res)
    end
    return res, param
end

import jInv.LinearSolvers.clear!;
function clear!(param::CnnHelmholtzSolver)
    param.model = []
    param.model_parameters = Dict()
end

import jInv.LinearSolvers.copySolver;
function copySolver(param::CnnHelmholtzSolver)
    return getCnnHelmholtzSolver(param.solver_type; model=param.model, model_parameters=param.model_parameters, solver_tol=param.solver_tol, relaxation_tol=param.relaxation_tol) 
end

function setModel(model, param::CnnHelmholtzSolver)
    param.model = model
    return param
end

# cycle and index - just for identifying the current retraining phase
function retrain(cycle::Int, index::Int, param::CnnHelmholtzSolver; iterations=4, batch_size=16, initial_set_size=32, lr=1e-5)
    param.cycle = cycle
    param.freqIndex = index
    new_model_name = "retrain_model_cycle=$(cycle)_freqIndex=$(index)"
    
    param.model, X = retrain_model(param.model, model_name, new_model_name, param.n, param.m, param.h,
                                param.kappa, param.omega, param.gamma, initial_set_size, batch_size, iterations, lr; relaxation_tol=param.relaxation_tol)

    return param, X
end