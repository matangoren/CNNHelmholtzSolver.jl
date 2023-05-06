include("../flux_components.jl");

# Multigrid Helmholtz Shifted Laplacian Methods

function get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    println("NEW NEW NEW - GOREN - get_helmholtz_matrices!")
    shifted_laplacian_matrix = kappa .* kappa .* omega .* (omega .- (im .* gamma) .- (im .* omega .* alpha))
    helmholtz_matrix = kappa .* kappa .* omega .* (omega .- (im .* gamma))
    println("typeof matrices = $(typeof(shifted_laplacian_matrix)) $(typeof(helmholtz_matrix))")
    return shifted_laplacian_matrix, helmholtz_matrix
end

function jacobi_helmholtz_method!(n, m, h, x, b, matrix; max_iter=1, w=0.8, use_gmres_alpha=0)
    println("NEW NEW NEW - GOREN - jacobi_helmholtz_method!")
    println("jacobi HERE HERE HERE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
	println("dir $(@__DIR__)")
    h1 = 1.0 / (h[1]^2)
    h2 = 1.0 / (h[2]^2)
    for i in 1:max_iter
        residual = b - helmholtz_chain!(x, matrix; h=h)  
        d = r_type(2.0 * (h1 + h2)) .- matrix    
        alpha = r_type(w) ./ d
        x += (alpha .* residual)
    end
    return x
end

function jacobi_helmholtz_method_channels!(n, m, h, x, b, matrix, matrixch; max_iter=1, w=0.8, use_gmres_alpha=0)
    h1 = 1.0 / (h[1]^2)
    h2 = 1.0 / (h[2]^2)
    for i in 1:max_iter
        y = helmholtz_chain_channels!(x, matrix; h=h)
        residual = b - y
        d = r_type(2.0 * (h1 + h2)) .- matrixch
        alpha = r_type(w) ./ d
        x .+= (alpha .* residual)
    end
    return x
end

function v_cycle_helmholtz!(n, m, h, x, b, kappa, omega, gamma; u = 1, v1_iter = 1, v2_iter = 10, use_gmres_alpha = 0, alpha= 0.5, log = 0, level = nothing, blocks=1)
    println("NEW NEW NEW - GOREN - v_cycle_helmholtz!")
    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(alpha))
    # Relax on Ax = b v1_iter times with initial guess x
    x = jacobi_helmholtz_method!(n, m, h, x, b, shifted_laplacian_matrix; max_iter=v1_iter, use_gmres_alpha=use_gmres_alpha)

    if( n % 2 == 0 && n > 4 && m % 2 == 0 && m > 4 && (level == nothing || level > 1))
        # Compute residual on fine grid

        residual_fine = b - helmholtz_chain!(x, helmholtz_matrix; h=h)#[:,:,1,1]

        # Compute residual, kappa and gamma on coarse grid
        residual_coarse = (down(real(residual_fine))+ im*down(imag(residual_fine)))#[:,:,1,1]

        kappa_coarse = down(reshape(kappa, n+1, m+1, 1, 1))[:,:,1,1]

        gamma_coarse = down(reshape(gamma, n+1, m+1, 1, 1))[:,:,1,1]

        # Recursive operation of the method on the coarse grid
        n_coarse = size(residual_coarse,1)-1
        m_coarse = size(residual_coarse,2)-1
        x_coarse = (zeros(c_type,n_coarse+1, m_coarse+1,1,blocks))|>cgpu
        for i = 1:u
            x_coarse, helmholtz_matrix_coarse = v_cycle_helmholtz!(n_coarse, m_coarse, h.*2, x_coarse, residual_coarse, kappa_coarse, omega, gamma_coarse; use_gmres_alpha = use_gmres_alpha,
                                                                    u=u, v1_iter=v1_iter, v2_iter=v2_iter, log=log, level = (level == nothing ? nothing : (level-1)), blocks=blocks)
        end

        # Correct
        fine_error = (up(real(x_coarse)) + im * up(imag(x_coarse)))#[:,:,1,1]
        x += fine_error

        if log == 1
            r1 = residual_fine
            r2 = b - helmholtz_chain!(x, helmholtz_matrix; h=h)
            println("n = $(n), norm of x = $(norm(x)), norm of fine_error = $(norm(fine_error)), residual before vcycle =$(norm(r1)/norm(b)), residual after vcycle =$(norm(r2)/norm(b)), level =$(level)")
        end
    else
        # Coarsest grid
        A_Coarsest(v::a_type) =  vec(helmholtz_chain!(reshape(v,n+1,m+1,1,blocks), shifted_laplacian_matrix; h=h))
        M_Coarsest(v::a_type) = vec(jacobi_helmholtz_method!(n, m, h, x, reshape(v,n+1,m+1,1,blocks), shifted_laplacian_matrix; max_iter=1, use_gmres_alpha=use_gmres_alpha))
        x,flag,err,iter,resvec = fgmres_func(A_Coarsest, vec(b), v2_iter, tol=1e-15, maxIter=1,
                                                    M=M_Coarsest, x=vec(x), out=-1, flexible=true)
        x = reshape(x,n+1,m+1,1,blocks)
    end

    # Relax on Ax = b v1_iter times with initial guess x
    x = jacobi_helmholtz_method!(n, m, h, x, b, shifted_laplacian_matrix; max_iter=v1_iter, use_gmres_alpha=use_gmres_alpha)

    return x, helmholtz_matrix
end

# function v_cycle_helmholtz!(n, m, h, x, b, h_matrix_level1, sl_matrix_level1, h_matrix_level2, sl_matrix_level2, h_matrix_level3, sl_matrix_level3; u = 1, v1_iter = 1, v2_iter = 10, use_gmres_alpha = 0, alpha= 0.5, log = 0, level = nothing)
#     if level == 3
#         h_matrix = h_matrix_level3
#         sl_matrix = sl_matrix_level3
#     elseif level == 2
#         h_matrix = h_matrix_level2
#         sl_matrix = sl_matrix_level2
#     else
#         h_matrix = h_matrix_level1
#         sl_matrix = sl_matrix_level1
#     end

#     # Relax on Ax = b v1_iter times with initial guess x
#     x = jacobi_helmholtz_method!(n, m, h, x, b, sl_matrix; max_iter=v1_iter, use_gmres_alpha=use_gmres_alpha)

#     if( n % 2 == 0 && n > 4 && m % 2 == 0 && m > 4 && (level == nothing || level > 1))
#         # Compute residual on fine grid
#         x_matrix = reshape(x, n+1, m+1, 1, 1)
#         residual_fine = b - helmholtz_chain!(x_matrix, h_matrix; h=h)[:,:,1,1]

#         # Compute residual, kappa and gamma on coarse grid
#         residual_coarse = (down(reshape(residual_fine, n+1, m+1, 1, 1)))[:,:,1,1]
#         # residual_coarse = (down(reshape(real(residual_fine), n+1, m+1, 1, 1)) + im*down(reshape(imag(residual_fine), n+1, m+1, 1, 1)))[:,:,1,1]

#         # Recursive operation of the method on the coarse grid
#         n_coarse = size(residual_coarse,1)-1
#         m_coarse = size(residual_coarse,2)-1
#         x_coarse = a_type(zeros(n_coarse+1, m_coarse+1))

#         for i = 1:u
#             x_coarse, _ = v_cycle_helmholtz!(n_coarse, m_coarse, h*2, x_coarse, residual_coarse, h_matrix_level1, sl_matrix_level1, h_matrix_level2, sl_matrix_level2, h_matrix_level3, sl_matrix_level3; use_gmres_alpha = use_gmres_alpha,
#                                                                     u=u, v1_iter=v1_iter, v2_iter=v2_iter, log=log, level = (level == nothing ? nothing : (level-1)))
#         end
#         x_coarse_matrix = reshape(x_coarse, n_coarse+1, m_coarse+1, 1, 1)

#         # Correct
#         # fine_error = (up(x_coarse_matrix))[:,:,1,1]
#         fine_error = up(real(x_coarse_matrix))[:,:,1,1] + im * up(imag(x_coarse_matrix))[:,:,1,1]
#         x = x + fine_error

#         if log == 1
#             r1 = residual_fine
#             r2 = b - reshape(helmholtz_chain!(reshape(x, n+1, m+1, 1, 1), h_matrix; h=h), n+1, m+1)
#             println("n = $(n), norm of x = $(norm(x)), norm of fine_error = $(norm(fine_error)), residual before vcycle =$(norm(r1)/norm(b)), residual after vcycle =$(norm(r2)/norm(b)), level =$(level)")
#         end
#     else
#         # Coarsest grid
#         A_Coarsest(v::a_type) = vec(helmholtz_chain!(reshape(v, n+1, m+1, 1, 1), sl_matrix; h=h))
#         M_Coarsest(v::a_type) = vec(jacobi_helmholtz_method!(n, m, h, x, reshape(v, n+1, m+1), sl_matrix)) #         M_Jacobi(n, h, x, sl_matrix, 1, v; use_gmres_alpha=use_gmres_alpha)
#         x,flag,err,iter,resvec = fgmres_func(A_Coarsest, vec(b), v2_iter, tol=1e-15, maxIter=1,
#                                                     M=M_Coarsest, x=vec(x), out=-1, flexible=true)
#         x = reshape(x, n+1, m+1)
#     end

#     # Relax on Ax = b v1_iter times with initial guess x
#     x = jacobi_helmholtz_method!(n, m, h, x, b, sl_matrix; max_iter=v1_iter, use_gmres_alpha=use_gmres_alpha)

#     return x, h_matrix
# end


# Eran Code
function absorbing_layer!(gamma::Array,pad,ABLamp;NeumannAtFirstDim=false)

    n=size(gamma)

    #FROM ERAN ABL:

    b_bwd1 = ((pad[1]:-1:1).^2)./pad[1]^2;
	b_bwd2 = ((pad[2]:-1:1).^2)./pad[2]^2;

	b_fwd1 = ((1:pad[1]).^2)./pad[1]^2;
	b_fwd2 = ((1:pad[2]).^2)./pad[2]^2;
	I1 = (n[1] - pad[1] + 1):n[1];
	I2 = (n[2] - pad[2] + 1):n[2];

	if NeumannAtFirstDim==false
		gamma[:,1:pad[2]] += ones(n[1],1)*b_bwd2'.*ABLamp;
		gamma[1:pad[1],1:pad[2]] -= b_bwd1*b_bwd2'.*ABLamp;
		gamma[I1,1:pad[2]] -= b_fwd1*b_bwd2'.*ABLamp;
	end

	gamma[:,I2] +=  (ones(n[1],1)*b_fwd2').*ABLamp;
	gamma[1:pad[1],:] += (b_bwd1*ones(1,n[2])).*ABLamp;
	gamma[I1,:] += (b_fwd1*ones(1,n[2])).*ABLamp;
	gamma[1:pad[1],I2] -= (b_bwd1*b_fwd2').*ABLamp;
	gamma[I1,I2] -= (b_fwd1*b_fwd2').*ABLamp;

    return gamma
end

function fgmres_v_cycle_helmholtz!(n, m, h, b, kappa, omega, gamma; restrt=30, maxIter=10)
    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    
    A(v) = vec(helmholtz_chain!(reshape(v, n+1, m+1, 1, 1), helmholtz_matrix; h=h))
    function M(v)
        v = reshape(v, n+1, m+1)
        x = zeros(gmres_type,n+1,m+1)|>pu
        x, = v_cycle_helmholtz!(n, m, h, x, v, kappa, omega, gamma; u=1,
                    v1_iter = 1, v2_iter = 20, alpha=0.5, log = 0, level = 3)
        return vec(x)
    end

    x = zeros(gmres_type,n+1,m+1)|>pu
    x,flag,err,iter,resvec = fgmres_func(A, vec(b), restrt, tol=1e-30, maxIter=maxIter,
                                                    M=M, x=vec(x), out=-1, flexible=true)

    return reshape(x, n+1, m+1)
end