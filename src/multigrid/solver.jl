using KrylovMethods
include("../data.jl")
include("../flux_components.jl")
include("helmholtz_methods.jl")

function vcycle_cpu_gmres!(model, n, m, kappa, kappa_features, omega, gamma, x_true, after_vcycle, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, relax_jacobi; v2_iter=10, level=3, axb=false, norm_input=false, log_error=true, test_name="", before_jacobi=false, unet_in_vcycle=false, arch=1)

    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    h = [1.0/n ; 1.0/m]

    if axb == true
        r_vcycle = zeros(c_type,n+1,m+1,1,1)
        r_vcycle[floor(Int32,n / 2.0),floor(Int32,m / 2.0),1,1] = r_type(1.0 ./minimum(h)^2)
        r_vcycle = vec(r_vcycle)
        for i = 2:blocks
            r_vcycle1 = zeros(c_type,n+1,m+1,1,1)
            r_vcycle1[floor(Int32,(n / blocks)*(i-1)),floor(Int32,(m / blocks)*(i-1)),1,1] = r_type(1.0 ./minimum(h)^2)
            r_vcycle = cat(r_vcycle, vec(r_vcycle1), dims=2)
        end
    else
        x_true = randn(c_type,n+1,m+1, 1, 1)
        r_vcycle, _ = generate_r_vcycle!(n, m, h, kappa, omega, gamma, reshape(x_true,n+1,m+1,1,1))
        r_vcycle = vec(r_vcycle)
        for i = 2:blocks
            x_true = randn(c_type,n+1,m+1, 1, 1)
            r_vcycle1, _ = generate_r_vcycle!(n, m, h, kappa, omega, gamma, reshape(x_true,n+1,m+1,1,1))
            r_vcycle = cat(r_vcycle, vec(r_vcycle1), dims=2)
        end
    end

    A(v) = vec(helmholtz_chain!(reshape(v, n+1, m+1, 1, 1), helmholtz_matrix; h=h))
    function As(v)
        res = vec(A(v[:,1]))
        for i = 2:blocks
            res = cat(res, vec(A(v[:,i])), dims=2)
        end

        return res
    end

    function M(r)
        e_vcycle = zeros(c_type,n+1,m+1)
        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,1], n+1, m+1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
        if after_vcycle == true
            e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,1], n+1, m+1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
        end
        res = vec(e_vcycle)
        for i = 2:blocks
            e_vcycle = zeros(c_type,n+1,m+1)
            e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,i], n+1, m+1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
            if after_vcycle == true
                e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,i], n+1, m+1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
            end
            res = cat(res, vec(e_vcycle), dims=2)
        end

        return res
    end

    x_init = zeros(c_type,(n+1)*(m+1),blocks)
    x,flag,err,iter,resvec = KrylovMethods.blockFGMRES(As, r_vcycle, restrt, tol=1e-30, maxIter=max_iter,
                                                        M=M, X=x_init, out=-1,flexible=true)

    return x
end