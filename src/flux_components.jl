# smooth_up_filter = (r_type.( reshape((1/4) * [1 2 1;2 4.0 2;1 2 1],3,3,1,1)))|>cgpu
# smooth_down_filter = (r_type.( reshape((1/16) * [1 2 1;2 4 2;1 2 1],3,3,1,1)))|>cgpu
# laplacian_filter = r_type.(reshape([0 -1 0;-1 4.0 -1;0 -1 0],3,3,1,1))


function get_laplacian_filter(h)
    h1 = -1.0 / (h[1]^2)
    h2 = -1.0 / (h[2]^2)
    return (r_type.(reshape([0 h1 0;h2 -2*(h1+h2) h2;0 h1 0],3,3,1,1)))|>cgpu
end

function block_filter!(filter_size, kernel, channels)
    w = zeros(r_type, filter_size, filter_size, channels, channels)
    for i in 1:channels
        w[:,:,i,i] = kernel
    end
    return w
end

# function big_block_filter!(filter_size, kernel, channels)
#     w = u_type.(zeros(r_type, filter_size, filter_size, channels, channels))|>cgpu
#     for i in 1:channels
#         w[:,:,:,i] = u_type.(kernel)|>cgpu
#     end
#     return w
# end

# block_laplacian_filter = block_filter!(3, get_laplacian_filter(h), 2)

# up = ConvTranspose(smooth_up_filter, (zeros(r_type,1))|>cgpu, stride=2,pad=1)|>cgpu;
# down = Conv(smooth_down_filter, (zeros(r_type,1))|>cgpu, stride=2,pad=1)|>cgpu;

block_up = ConvTranspose(block_filter!(3, smooth_up_filter, 2), true, stride=2,pad=1)
block_down = Conv(block_filter!(3, smooth_down_filter, 2), true, stride=2,pad=1)

# i_conv = Conv(block_filter!(1, reshape([1.0],1,1,1,1), 2),true,pad=1)|> pu

# check how to perform mirroring padding
function laplacian_conv!(grid; h=[0.0225 ; 0.014])
    filter = get_laplacian_filter(h)
    conv = Conv(filter, r_type.([0.0]), pad=(1,1))
    return conv(grid)
end

# function helmholtz_chain!(grid::Union{Array{ComplexF64}, Array{ComplexF32}, CuArray{ComplexF32}, CuArray{ComplexF64}}, matrix::Union{Array{ComplexF64}, Array{ComplexF32}, CuArray{ComplexF32}, CuArray{ComplexF64}}; h=[0.0225 ; 0.014])
function helmholtz_chain!(grid::a_type, matrix::a_type; h=[0.0225 ; 0.014])
    # zero-padding
    # filter = get_laplacian_filter(h)
    # conv = Conv(get_laplacian_filter(h), r_type.([0.0]), pad=(1,1))|>cgpu
    # y = conv(grid|>cgpu) - matrix .* grid
    # return y

    # mirror-padding
    println("NEW NEW NEW - GOREN - helmholtz_chain!")
	println("dir $(@__DIR__)")
    term = matrix .* (grid)
    helmholtz_filter = get_laplacian_filter(h)
    conv = Conv(helmholtz_filter, (zeros(r_type,1))|>cgpu; pad=0)|>cgpu
    grid = pad_repeat(grid, (1,1,1,1))
    return conv(real(grid))+im*conv(imag(grid)) - term
end

function helmholtz_chain_channels!(grid, matrix; h=[0.0225 ; 0.014])
    block_laplacian_filter = block_filter!(3, get_laplacian_filter(h), 2)
    filter = r_type.((1.0 / (h^2)) * block_laplacian_filter)
    # conv = Conv(filter, [0.0], pad=(1,1))|> pu
    conv = Conv(filter, true, pad=(1,1))|> pu
    y = conv(grid)-sum(matrix .* grid, dims=4)
    
    return y
end