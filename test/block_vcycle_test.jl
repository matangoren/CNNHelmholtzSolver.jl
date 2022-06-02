using Statistics
using LinearAlgebra: mul!
using Flux
using Flux: @functor
using LaTeXStrings
using KrylovMethods
using Distributions: Normal
using BSON: @load
using Plots
using Dates
using CSV, DataFrames
using Random

pu = cpu # gpu
r_type = Float64
c_type = ComplexF64
u_type = Float32
gmres_type = ComplexF64
a_type = Array{gmres_type}

println("before includes")
include("../src/multigrid/helmholtz_methods.jl")
include("../src/data.jl")
include("../src/kappa_models.jl")
include("../src/unet/utils.jl")
println("after includes")

fgmres_func = KrylovMethods.fgmres
level = 3
v2_iter = 10
after_vcycle = false
gamma_val = 0.00001
pad_cells = [10;10]
blocks = 10

kappa_type = 4 # 0 - uniform, 1 - CIFAR10, 2 - STL10
kappa_threshold = 25 # kappa ∈ [0.01*threshold, 1]
smooth = true
k_kernel = 5
f = 10.0
n = m = 128
h = 2.0./(m+n)

println("before kappa")
kappa = r_type.(generate_kappa!(n, m; type=kappa_type, smooth=smooth, threshold=kappa_threshold, kernel=k_kernel)|>pu) #ones(r_type,n-1,n-1)
println("after kappa")
omega = r_type(2*pi*f);
gamma = gamma_val*2*pi * ones(r_type,size(kappa))
gamma = r_type.(absorbing_layer!(gamma, pad_cells, omega))|>pu

# my version for v_cycle_helmholtz - in block form
function block_M_Jacobi(n, m, h, x, matrix, iterations, v;use_gmres_alpha=0)
    b = reshape(v, n-1, m-1, blocks)
    x = block_jacobi_helmholtz_method!(n, m, h, x, b, matrix; max_iter=iterations, use_gmres_alpha=use_gmres_alpha)
    return vec(x)
end

function block_jacobi_helmholtz_method!(n, m, h, x, b, matrix; max_iter=1, w=0.8, use_gmres_alpha=0)
    for i in 1:max_iter
        y = helmholtz_chain!(real(reshape(x, n-1, m-1, 1, blocks)), matrix; h=h) + im*helmholtz_chain!(imag(reshape(x, n-1, m-1, 1, blocks)), matrix; h=h)
        residual = b - y
        d = r_type(4.0 / h^2) .- matrix
        alpha = r_type(w) ./ d
        x = x + alpha .* residual
    end
    return x
end

function block_v_cycle_helmholtz!(n, m, h, x, b, kappa, omega, gamma; u = 1, v1_iter = 1, v2_iter = 10, use_gmres_alpha = 0, alpha= 0.5, log = 0, level = nothing)

    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(alpha))

    # Relax on Ax = b v1_iter times with initial guess x
    x = block_jacobi_helmholtz_method!(n, m, h, x, b, shifted_laplacian_matrix; max_iter=v1_iter, use_gmres_alpha=use_gmres_alpha)

    if( n % 2 == 0 && n > 4 && m % 2 == 0 && m > 4 && (level == nothing || level > 1))

        # Compute residual on fine grid
        x_matrix = reshape(x, n-1, m-1, 1, blocks)
        residual_fine = b - (helmholtz_chain!(real(x_matrix), helmholtz_matrix; h=h) + im*helmholtz_chain!(imag(x_matrix), helmholtz_matrix; h=h))[:,:,1,1] #) helmholtz_chain!(x_matrix, helmholtz_matrix; h=h)[:,:,1,1]

        # Compute residual, kappa and gamma on coarse grid
        residual_coarse = down(reshape(real(residual_fine), n-1, m-1, 1, blocks)|>pu)[:,:,1,1] + im * down(reshape(imag(residual_fine), n-1, m-1, 1, 1)|>pu)
        kappa_coarse = down(reshape(kappa, n-1, m-1, 1, blocks)|>pu)
        gamma_coarse = down(reshape(gamma, n-1, m-1, 1, blocks)|>pu)

        # Recursive operation of the method on the coarse grid
        n_coarse = size(residual_coarse,1)+1
        m_coarse = size(residual_coarse,2)+1
        x_coarse = zeros(c_type,n_coarse-1, m_coarse-1, blocks)|>pu

        for i = 1:u
            x_coarse, helmholtz_matrix_coarse = block_v_cycle_helmholtz!(n_coarse, m_coarse, h*2, x_coarse, residual_coarse, kappa_coarse, omega, gamma_coarse; use_gmres_alpha = use_gmres_alpha,
                                                                    u=u, v1_iter=v1_iter, v2_iter=v2_iter, log=log, level = (level == nothing ? nothing : (level-1)))
        end
        x_coarse_matrix = reshape(x_coarse, n_coarse-1, m_coarse-1, 1, blocks)

        # Correct
        fine_error = up(real(x_coarse_matrix)|>pu)+ im * up(imag(x_coarse_matrix)|>pu)
        x = x + fine_error

        if log == 1
            r1 = residual_fine
            r2 = b - reshape(helmholtz_chain!(reshape(x, n-1, m-1, 1, blocks), helmholtz_matrix; h=h), n-1, m-1)
            println("n = $(n), norm of x = $(norm(x)), norm of fine_error = $(norm(fine_error)), residual before vcycle =$(norm(r1)/norm(b)), residual after vcycle =$(norm(r2)/norm(b)), level =$(level)")
        end
    else
        # Coarsest grid
        A_Coarsest(v::a_type) = vec(helmholtz_chain!(reshape(real(v), n-1, m-1, 1, blocks), shifted_laplacian_matrix; h=h) + im*helmholtz_chain!(reshape(imag(v), n-1, m-1, 1, blocks), shifted_laplacian_matrix; h=h)) # vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), shifted_laplacian_matrix; h=h))
        M_Coarsest(v::a_type) = block_M_Jacobi(n, m, h, x, shifted_laplacian_matrix, 1, v; use_gmres_alpha=use_gmres_alpha)
        x,flag,err,iter,resvec = fgmres_func(A_Coarsest, vec(b), v2_iter, tol=1e-15, maxIter=1,
                                                    M=M_Coarsest, x=vec(x), out=-1, flexible=true)
        x = reshape(x, n-1, m-1, blocks)
    end

    # Relax on Ax = b v1_iter times with initial guess x
    x = jacobi_helmholtz_method!(n, m, h, x, b, shifted_laplacian_matrix; max_iter=v1_iter, use_gmres_alpha=use_gmres_alpha)

    return x, helmholtz_matrix
end

use_gpu = false
if use_gpu == true
    using CUDA
    # CUDA.allowscalar(false)
    cgpu = gpu
else
    cgpu = cpu
    pyplot()
end


shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))

function M(r)
    e_vcycle = zeros(c_type,n-1,m-1)
    e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,1], n-1, m-1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
    if after_vcycle == true
        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,1], n-1, m-1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
    end
    res = vec(e_vcycle)
    for i = 2:blocks
        e_vcycle = zeros(c_type,n-1,m-1)
        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,i], n-1, m-1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
        if after_vcycle == true
            e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,i], n-1, m-1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
        end
        res = cat(res, vec(e_vcycle), dims=2)
    end

    return res
end

function Jacobi_vector(r)
    e_vcycle = zeros(c_type,n-1,m-1)
    x = jacobi_helmholtz_method!(n, m, h, e_vcycle, reshape(r[:,1], n-1, m-1), shifted_laplacian_matrix; max_iter=1, use_gmres_alpha=false)
    res = vec(x)
    for i = 2:blocks
        e_vcycle = zeros(c_type,n-1,m-1)
        x = jacobi_helmholtz_method!(n, m, h, e_vcycle, reshape(r[:,i], n-1, m-1), shifted_laplacian_matrix; max_iter=1, use_gmres_alpha=false)
        res = cat(res, vec(x), dims=2)
    end

    return res
end
function Jacobi_block(r)
    e_vcycle = zeros(c_type,n-1,m-1, 1, blocks)
    # e_vcycle, = block_v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r, n-1, m-1, blocks), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
    x = block_jacobi_helmholtz_method!(n, m, h, e_vcycle, reshape(r, n-1, m-1, 1, blocks), shifted_laplacian_matrix; max_iter=1, use_gmres_alpha=false)
    return x
end


x_true = randn(c_type,n-1,m-1, 1, 1)
r_vcycle, _ = generate_r_vcycle!(n, m, kappa, omega, gamma, reshape(x_true,n-1,m-1,1,1))
r_vcycle = vec(r_vcycle)
for i = 2:blocks
    global r_vcycle, x_true
    x_true = randn(c_type,n-1,m-1, 1, 1)
    r_vcycle1, _ = generate_r_vcycle!(n, m, kappa, omega, gamma, reshape(x_true,n-1,m-1,1,1))
    r_vcycle = cat(r_vcycle, vec(r_vcycle1), dims=2)
end

println("r_vcycle size: ",size(r_vcycle))

result_vector = Jacobi_vector(r_vcycle)

println("Jacobi_vector(r_vcycle) size: ", size(result_vector))
println("Jacobi_vector(r_vcycle) norm: ", norm(result_vector))

result_block = Jacobi_block(r_vcycle)
println("Jacobi_block(r_vcycle) size: ", size(result_block))
println("Jacobi_block(r_vcycle) norm: ", norm(result_block))
