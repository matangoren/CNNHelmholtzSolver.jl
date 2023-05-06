include("../multigrid/helmholtz_methods.jl")
include("../src/gpu_krylov.jl")

function get_kappa_features(model, n, m, kappa, gamma; arch=2, indexes=3)
    kappa_features = NaN
    if arch != 0
        kappa_input = reshape(kappa.^2, n+1, m+1, 1, 1)
        if indexes != 3
            kappa_input = cat(kappa_input, reshape(gamma, n+1, m+1, 1, 1), dims=3)
        end
        kappa_features = model.kappa_subnet(kappa_input|>cgpu)
    end
    return kappa_features
end

function solve(solver_type, model, n, m, h, r_vcycle, kappa, kappa_features, omega, gamma, restrt, max_iter; v2_iter=10, level=3, axb=false, arch=2)

    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    blocks = size(r_vcycle,2)
    coefficient = r_type(minimum(h)^2)

    A(v) = vec(helmholtz_chain!(reshape(v, n+1, m+1, 1, blocks), helmholtz_matrix; h=h))
    
    function M_Unet(r)
        r = reshape(r, n+1, m+1, 1, blocks)
        rj = reshape(r, n+1, m+1, 1, blocks)
        e = zeros(c_type, n+1, m+1, 1, blocks)|>cgpu
        ej = zeros(c_type, n+1, m+1, 1, blocks)|>cgpu

        if solver_type["before_jacobi"] == true
            ej = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
            rj = r - reshape(A(ej), n+1, m+1, 1, blocks)
        end
        
        if solver_type["unet"] == true
            for i=1:blocks  
                input = complex_grid_to_channels!(reshape(rj[:,:,1,i],n+1,m+1,1,1); blocks=1)
                if arch == 1
                    input = cat(input, reshape(kappa.^2, n+1, m+1, 1, 1), reshape(gamma, n+1, m+1, 1, 1), kappa_features, dims=3)
                    e_unet = model.solve_subnet(input)
                elseif arch == 2
                    input = cat(input, reshape(kappa.^2, n+1, m+1, 1, 1), reshape(gamma, n+1, m+1, 1, 1), dims=3)
                    e_unet = model.solve_subnet(input, kappa_features)
                else
                    input = cat(input, reshape(kappa.^2, n+1, m+1, 1, 1), reshape(gamma, n+1, m+1, 1, 1), dims=3)
                    e_unet = model(input)
                end

                e[:,:,1,i] = (e_unet[:,:,1,1] + im*e_unet[:,:,2,1]) .* coefficient
            end
        end
        e += ej
        
        if solver_type["after_jacobi"] == true
            e = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
        elseif solver_type["after_vcycle"] == true
            e, = v_cycle_helmholtz!(n, m, h, e, r, kappa, omega, gamma; v2_iter = v2_iter, level = level, blocks=blocks)
        end
        return vec(e)
    end

    function SM(r)
        e_vcycle = zeros(c_type,n+1,m+1,1,blocks)|>cgpu
        println("In SM --- $(n) $(m) $(h)")
        println("In SM --- $(maximum(kappa))")
        println("In SM --- $(omega)")
        println("In SM --- $(v2_iter)")
        println("In SM --- $(blocks)")
        println("In SM --- $(typeof(r)) $(size(r))")

        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r,n+1,m+1,1,blocks), kappa, omega, gamma; v2_iter = v2_iter, level=3, blocks=blocks)
        return vec(e_vcycle)
    end

    x_init = zeros(c_type,(n+1)*(m+1),blocks)|>cgpu
    ##### just for run-time compilation #####
    # @info "$(Dates.format(now(), "HH:MM:SS")) - before warm-up" 
    # x3,flag3,err3,iter3,resvec3 = fgmres_func(A, vec(r_vcycle), 3, tol=1e-5, maxIter=1,
    #                                                 M=SM, x=vec(x_init), out=-1,flexible=true)
    # x1,flag1,err1,iter1,resvec1 = fgmres_func(A, vec(r_vcycle), 1, tol=1e-5, maxIter=1,
    #                                                 M=M_Unet, x=vec(x3), out=1,flexible=true)
    # @info "$(Dates.format(now(), "HH:MM:SS")) - after warm-up" 
    #########################################
    
    # println("before gmres $(typeof(r_vcycle)) $(norm(r_vcycle)) $(norm(x_init))")
    # x3,flag3,err3,iter3,resvec3 =@time fgmres_func(A, vec(r_vcycle), 3, tol=1e-15, maxIter=1,
    #                                                 M=SM, x=vec(x_init), out=-1,flexible=true)
    # println("In CNN solve - number of iterations=$(iter3) err1=$(err3)")

    # x1,flag1,err1,iter1,resvec1 =@time fgmres_func(A, vec(r_vcycle), restrt, tol=1e-5, maxIter=max_iter,
    #                                                         M=M_Unet, x=vec(x3), out=1,flexible=true)
    
    # println("In CNN solve - number of iterations=$(iter1) err1=$(err1)")
    # return reshape(x1,(n+1)*(m+1),blocks)|>pu
    println("Call SM")
    return SM(r_vcycle)
end