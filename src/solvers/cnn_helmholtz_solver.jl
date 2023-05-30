export CnnHelmholtzSolver,getCnnHelmholtzSolver,solveLinearSystem,copySolver,setupSolver,setMediumParameters,setSolverType

include("../unet/model.jl")
include("../data.jl")
include("solver_utils.jl")

function get_solver_type(solver_name)
    if solver_name == "JU"
        return Dict("before_jacobi"=>true, "unet"=>true, "after_jacobi"=>true, "after_vcycle"=>false)
    elseif solver_name == "VU"
        return Dict("before_jacobi"=>false, "unet"=>true, "after_jacobi"=>false, "after_vcycle"=>true)
    elseif solver_name == "U"
        return Dict("before_jacobi"=>false, "unet"=>true, "after_jacobi"=>false, "after_vcycle"=>false)
    end
    return Dict("before_jacobi"=>false, "unet"=>false, "after_jacobi"=>false, "after_vcycle"=>true)
end


println(use_gpu)
println(cgpu)
println(a_type)
println("pwd $(pwd())")
println("dir $(@__DIR__)")

model_name = "without_alpha"

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
end

function getCnnHelmholtzSolver(solver_name; n=128, m=128,h=[], kappa=[], omega=[], gamma=[], model=[], model_parameters=Dict(), kappa_features=[], tuning_size=100, tuning_iterations=100, solver_tol=1e-8, relaxation_tol=1e-4)
    return CnnHelmholtzSolver(get_solver_type(solver_name), n, m, h, kappa, omega, gamma, model, model_parameters, kappa_features, tuning_size, tuning_iterations, 0, solver_tol, relaxation_tol)
end

function getCnnHelmholtzSolver(solver_type::Dict; n=128, m=128,h=[], kappa=[], omega=[], gamma=[], model=[], model_parameters=Dict(), kappa_features=[], tuning_size=100, tuning_iterations=100, solver_tol=1e-8, relaxation_tol=1e-4)
    return CnnHelmholtzSolver(solver_type, n, m, h, kappa, omega, gamma, model, model_parameters, kappa_features, tuning_size, tuning_iterations, 0, solver_tol, relaxation_tol)
end

# need only B - the rhs of the linear equation. The rest of the computations is done by the cnn model.
import jInv.LinearSolvers.solveLinearSystem;
function solveLinearSystem(A,B,param::CnnHelmholtzSolver,doTranspose::Int=0)
    return solveLinearSystem!(A,B,[],param,doTranspose)
end

function setupSolver!(param::CnnHelmholtzSolver)
    file = matopen(joinpath(@__DIR__, "../../models/$(model_name)/model_parameters"), "r"); DICT = read(file); close(file);
    e_vcycle_input = DICT["e_vcycle_input"]
    kappa_input = DICT["kappa_input"]
    gamma_input = DICT["gamma_input"]
    kernel = Tuple(DICT["kernel"])
    model_type = FFSDNUnet #@eval $(Symbol(DICT["model_type"]))
    k_type = TFFKappa # @eval $(Symbol(DICT["k_type"])) # TFFKappa
    resnet_type = TSResidualBlockI #@eval $(Symbol(DICT["resnet_type"])) # TSResidualBlockI
    k_chs = DICT["k_chs"]
    indexes = DICT["indexes"]
    σ = elu #@eval $(Symbol(DICT["sigma"]))
    arch = DICT["arch"]

    # param.model = load_model!(joinpath(@__DIR__, "../../models/$(model_name)/model.bson"), e_vcycle_input, kappa_input, gamma_input; kernel=kernel, model_type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)
    model = create_model!(e_vcycle_input, kappa_input, gamma_input; kernel=kernel, type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)
    model = model|>cpu
    println("after create")
    @load joinpath(@__DIR__, "../../models/$(model_name)/model.bson") model #"../../models/$(test_name).bson" model
    @info "$(Dates.format(now(), "HH:MM:SS.sss")) - Load Model"
    param.model = model|>cgpu
    param.kappa_features = Base.invokelatest(get_kappa_features,param.model, param.n, param.m, param.kappa, param.gamma; arch=arch, indexes=indexes)
    param.model_parameters = DICT
    
    return param
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
    # heatmap(param.kappa|>cpu, color=:blues)
    # savefig("m_from_fwi")
    # heatmap(param.gamma|>cpu, color=:blues)
    # savefig("gamma_from_fwi")
    
    if param.model == []
        param = setupSolver!(param)
    end

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
        # negate the imaginary part of B (rhs)
        print("GG")
        B = real(B) - im*imag(B)
    end

    res = Base.invokelatest(solve, param.solver_type, param.model, param.n, param.m, param.h, B|>cgpu, param.kappa, param.kappa_features, param.omega, param.gamma, 10, 30; arch=(param.model_parameters)["arch"], solver_tol=param.solver_tol, relaxation_tol=param.relaxation_tol)
    
    if doTranspose == 1
        # negate the imaginary part of res
        print("GG")
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
    return getCnnHelmholtzSolver(param.solver_type)
end