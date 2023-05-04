using KrylovMethods
include("../data.jl")
include("../flux_components.jl")

# check the purpose of this file

function unet_cpu_gmres!(model, n, m, kappa, kappa_features, omega, gamma, x_true, after_vcycle, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, relax_jacobi; v2_iter=10, level=3, axb=false, norm_input=false, log_error=true, test_name="", before_jacobi=false, unet_in_vcycle=false, arch=1)

    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    h = r_type.([1.0/n ; 1.0/m])

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

    coefficient = r_type(h^2)

    A(vs) = reshape(helmholtz_chain!(reshape(vs, n+1, m+1, 1, blocks), helmholtz_matrix; h=h),(n+1)*(m+1),blocks)

    function M_Unet(r)
        r = reshape(r, n+1, m+1)
        rj = reshape(r, n+1, m+1)
        e = zeros(c_type, n+1, m+1)
        ej = zeros(c_type, n+1, m+1)

        if after_vcycle != true
            ej = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
            rj = r - reshape(A(ej), n+1, m+1)
        end

        input = complex_grid_to_channels!(reshape(rj , n+1, m+1, 1, 1))
        if arch == 1
            input = cat(input, reshape(kappa, n+1, m+1, 1, 1), reshape(gamma, n+1, m+1, 1, 1), kappa_features, dims=3)
            e_unet = (model.solve_subnet(u_type.(input)|>cgpu)|>cpu)
        elseif arch == 2
            input = cat(input, reshape(kappa, n+1, m+1, 1, 1), reshape(gamma, n+1, m+1, 1, 1), dims=3)
            e_unet = (model.solve_subnet(u_type.(input)|>cgpu, kappa_features|>cgpu)|>cpu)
        else
            input = cat(input, reshape(kappa, n+1, m+1, 1, 1), reshape(gamma, n+1, m+1, 1, 1), dims=3)
            e_unet = (model(u_type.(input)|>cgpu)|>cpu)
        end

        e_unet = reshape((e_unet[:,:,1,1] + im*e_unet[:,:,2,1]),n+1,m+1,1,1) .* coefficient
        e = reshape(e_unet, n+1,m+1)

        e = ej + e

        if relax_jacobi == true
            e = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
        end

        if after_vcycle == true
            e, = v_cycle_helmholtz!(n, m, h, e, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        end
        return vec(e)
    end

    function M_Unets(rs)
        res = M_Unet(rs[:,1])
        for i = 2:blocks
            res = cat(res, M_Unet(rs[:,i]), dims=2)
        end

        return res
    end

    function SM(r)
        e_vcycle = zeros(c_type,n+1,m+1)
        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,1], n+1, m+1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
        res = vec(e_vcycle)
        for i = 2:blocks
            e_vcycle = zeros(c_type,n+1,m+1)
            e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,i], n+1, m+1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
            res = cat(res, vec(e_vcycle), dims=2)
        end

        return res
    end

    x_init = zeros(c_type,(n+1)*(m+1),blocks)
    x3,flag3,err3,iter3,resvec3= KrylovMethods.blockFGMRES(A, r_vcycle, 3, tol=1e-30, maxIter=1,
                                                    M=SM, X=x_init, out=-1,flexible=true)
    i = 1
    x1 = x3
    x1,flag1,err1,iter1,resvec1 = KrylovMethods.blockFGMRES(A, r_vcycle, restrt, tol=1e-30, maxIter=max_iter,
                                                            M=M_Unets, X =x1, out=-1,flexible=true)

    return x
end