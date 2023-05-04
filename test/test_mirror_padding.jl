using Flux
using Flux: @functor

using CUDA

cgpu = gpu


pu = gpu

# struct rConv
#     a
#     layer
# end

# @functor rConv

# function getrConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
#         init = Flux.glorot_uniform, stride = 1, pad = 0, dilation = 1, groups = 1,
#         bias = true) where N
    
#     a = Flux.calc_padding(Flux.Conv,pad,k,dilation,stride)
#     weight = Flux.convfilter(k, ch; init, groups)

#     layer = Flux.Conv(weight, bias, σ; stride=stride, pad=0, dilation=dilation, groups=groups)

#     rConv(a,layer)
# end

# function (m::rConv)(x)
#     m.layer(NNlib.pad_reflect(x,m.a))
# end


# filter = reshape([0 0 0 ; 0 1 0 ; 0 0 0], 3,3,1,1)|>gpu
# display(filter)
# r = reshape(collect(1:9), 3,3,1,1)|>gpu
# r = pad_repeat(r, (1,1,1,1))
# display(r)

# # println(size(r[:,:,1]))

# layer = Conv(filter, Float32.([0.0]))|> gpu

# y = layer(r|>gpu)
# println(y)

#

# xs = rand(ComplexF32, 3, 3, 1, 1)|>gpu
# xs = pad_reflect(xs, (1,1,1,1))

# real_xs = real(xs)
# im_xs = imag(xs)
# println("real part")
# display(real_xs)
# println()
# println("im part")
# display(im_xs)
# println()
# weight =reshape(Float32.([0 0 0 ; 0 1 0 ; 0 0 0]), 3,3,1,1)|>gpu
# display(weight)
# println()

# bias = zeros(Float32, 1)|>gpu

# layer = Conv(weight, bias; pad=0)|>gpu
# println("real part")
# display(layer(real_xs))
# println()
# println("im part")
# display(layer(im_xs))
# println()

# xs = ComplexF32.(real_xs+im*im_xs)
# println("result")
# display(xs)
# println()

smooth_up_filter = Float32.( reshape((1/4) * [1 2 1;2 4.0 2;1 2 1],3,3,1,1))
up = ConvTranspose(smooth_up_filter, true, stride=2,pad=1)|> pu
xs = rand(ComplexF32, 3, 3, 1, 1)|>gpu
println(up(xs))