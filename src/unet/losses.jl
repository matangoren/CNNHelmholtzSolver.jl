function jacobi_channels!(n, x, b, matrix; w=0.8)
    h = 2.0 ./ (n+m)

    y = helmholtz_chain_channels!(x, matrix; h=h)|> cgpu
        residual = b - y
    d = u_type(4.0 / h^2) .- sum(matrix, dims=4)
    alpha = u_type(w) ./ d
    step = alpha .* residual

    return x .+ step
end

function norm_diff!(x,y)
    return sqrt(sum((x - y).^2) / sum(y.^2))
end

function error_loss!(model, input, output; in_tuning=false)
    model_result = model(input |> cgpu; in_tuning=in_tuning)
    
    return norm_diff!(model_result, output|>cgpu)
end

function get_matrices!(alpha, kappa, omega, gamma)
    inv = reshape([u_type(-1.0)],1,1,1,1)|> cgpu

    s_h_real = (kappa .* kappa .* omega .* omega)
    h_imag = kappa .* kappa .* omega .* gamma
    s_imag = (kappa .* kappa .* omega .* omega .* alpha) .+ (kappa .* kappa .* omega .* gamma)

    h_matrix = cat(cat(s_h_real, h_imag, dims=3),cat(inv .* h_imag, s_h_real, dims=3),dims=4) |> cgpu
    s_matrix = cat(cat(s_h_real, s_imag, dims=3),cat(inv .* s_imag, s_h_real, dims=3),dims=4) |> cgpu

    return h_matrix, s_matrix
end

function residual_loss!(model, n, m, f, input, output)
    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    h = u_type(2.0 ./ (n+m))
    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu

    e0 = model(input)

    r_unet = (h^2) .* helmholtz_chain_channels!(e0, h_matrix; h=h)|> cgpu
    return norm_diff!(r_unet, r)
end

function error_residual_loss!(model, n, m, f, input, output)
    e0 = model(input)

    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    h = u_type(2.0 ./ (n+m))
    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu
    r_unet = (h^2) .* helmholtz_chain_channels!(e0, h_matrix; h=h)|> cgpu

    e_loss = norm_diff!(e0, output|>cgpu)
    r_loss = norm_diff!(r_unet, r)

    return e_loss + 0.1 * r_loss
end

function error_residual_loss_details!(model, n, m, f, input, output)
    e0 = model(input)

    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    h = u_type(1.0 ./ n)
    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu
    r_unet = (h^2) .* helmholtz_chain_channels!(e0, h_matrix; h=h)|> cgpu

    e_loss = norm_diff!(e0, output|>cgpu)
    r_loss = norm_diff!(r_unet, r)

    return [e_loss + 0.1 * r_loss, r_loss, e_loss]
end

function dataset_loss!(dataloader, loss_func)
    loss = 0
    for (x, y) in dataloader
        loss += loss_func(x,y)
    end

    return loss
end

# function batch_loss!(set, loss; errors_count=1, gamma_input=false, append_gamma=identity)
#     set_size = size(set,1)
#     batch_size = min(1000,set_size)
#     batchs = floor(Int64,set_size/batch_size)
#     loss_sum = zeros(errors_count)
#     for batch_idx in 1:batchs
#         batch_set = set[(batch_idx-1)*batch_size+1:batch_idx*batch_size]
#         if gamma_input == true
#             batch_set = append_gamma.(batch_set)
#         end
#         current_loss = loss.(batch_set|>cgpu)
#         loss_sum = loss_sum .+ sum(hcat(current_loss...),dims=2)
#     end
#     return (loss_sum ./ set_size)
# end

function full_solution_loss!(model, input, output, n, m, f)
    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    e = output
    h = u_type(2.0 ./ (n+m))

    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu

    e0 = model(input)

    ae1 = (h^2) .* helmholtz_chain_channels!(e0, h_matrix; h=h)|> cgpu

    return norm_diff!(e0, e|> cgpu) + norm_diff!(ae1, r)
end

function full_solution_loss_details!(model, input, output, n, m, f)
    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    e = output
    h = u_type(2.0 ./ (n+m))

    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu

    e0 = model(input)

    ae1 = (h^2) .* helmholtz_chain_channels!(e0, h_matrix; h=h)|> cgpu

    return [norm_diff!(e0, e|> cgpu) + norm_diff!(ae1, r), norm_diff!(e0, e|> cgpu), norm_diff!(ae1, r)]
end

function full_solution_loss1!(model, input, output, n, m, f)
    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    e = output
    h = u_type(2.0 ./ (n+m))

    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu

    e0 = model(input)

    e1 = jacobi_channels!(n, e0, r ./ (h^2), h_matrix)|> cgpu
    ae1 = (h^2) .* helmholtz_chain_channels!(e1, h_matrix; h=h)|> cgpu
    r1 = r - ae1

    e2 = model(cat(r1,kappa,gamma,dims=3)|> cgpu)
    e3 = e1 + e2

    e4 = jacobi_channels!(n, e3, r ./ (h^2), h_matrix)
    # ae4 = (h^2) .* helmholtz_chain_channels!(e4, h_matrix; h=h)|> cgpu
    # r2 = r - ae4

    # e5 = model(cat(r2,kappa,gamma,dims=3)|> cgpu)
    # e6 = e4 + e5

    # e7 = jacobi_channels!(n, e6, r ./ (h^2), h_matrix)

    return norm_diff!(e0, e|> cgpu) + norm_diff!(e3, e|> cgpu) + norm_diff!(e4, e|> cgpu)
    #return norm_diff!(e0, e|> cgpu) + norm_diff!(e1, e|> cgpu) + norm_diff!(e3, e|> cgpu) + norm_diff!(e4, e|> cgpu) + norm_diff!(e6, e|> cgpu) + norm_diff!(e7, e|> cgpu)
end

function full_solution_loss_details1!(model, input, output, n, m, f)
    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    e = output
    h = u_type(2.0 ./ (n+m))

    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu

    e0 = model(input)
    # @info "e0 $(size(e0)) $(typeof(e0)) h_matrix $(size(h_matrix)) $(typeof(h_matrix))"
    e1 = jacobi_channels!(n, e0, r ./ (h^2), h_matrix)|> cgpu
    ae1 = (h^2) .* helmholtz_chain_channels!(e1, h_matrix; h=h)|> cgpu
    r1 = r - ae1

    e2 = model(cat(r1,kappa,gamma,dims=3)|> cgpu)
    e3 = e1 + e2

    e4 = jacobi_channels!(n, e3, r ./ (h^2), h_matrix)
    # ae4 = (h^2) .* helmholtz_chain_channels!(e4, h_matrix; h=h)|> cgpu
    # r2 = r - ae4

    # e5 = model(cat(r2,kappa,gamma,dims=3)|> cgpu)
    # e6 = e4 + e5

    # e7 = jacobi_channels!(n, e6, r ./ (h^2), h_matrix)

    # if norm(r) > 70 && norm(r) < 70.005 # mod(print_index,1000) == 0
    #     @info "e $(norm(e)) r $(norm(r)) e0 $(norm(e0)) e1 $(norm(e1)) ae1 $(norm(ae1)) r1 $(norm(r1)) e2 $(norm(e2)) e3 $(norm(e3)) e4 $(norm(e4)) ae4 $(norm(ae4)) r2 $(norm(r2)) e5 $(norm(e5)) e6 $(norm(e6)) e7 $(norm(e7))"
    #     @info "e $(norm(e)) r $(norm(r)) e0 $(norm(e0)) e1 $(norm(e1)) ae1 $(norm(ae1)) r1 $(norm(r1)) e2 $(norm(e2)) e3 $(norm(e3)) e4 $(norm(e4)) ae4 $(norm(ae4)) r2 $(norm(r2)) e5 $(norm(e5)) e6 $(norm(e6)) e7 $(norm(e7))"
    #
    # end
    # print_index = print_index+1
    return [norm_diff!(e0, e|> cgpu) + norm_diff!(e3, e|> cgpu) + norm_diff!(e4, e|> cgpu),
            norm_diff!(e0, e|> cgpu), norm_diff!(e3, e|> cgpu), norm_diff!(e4, e|> cgpu)] #  + norm_diff!(e4, e|> cgpu) + norm_diff!(e6, e|> cgpu) + norm_diff!(e7, e|> cgpu)
end