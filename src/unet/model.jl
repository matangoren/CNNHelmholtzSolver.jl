include("../../src/unet/utils.jl")


ConvDown(in_chs,out_chs;kernel = (5,5), stride=(2,2), pad=2, σ=elu) = Chain(Conv(kernel, in_chs=>out_chs, stride=stride, pad = pad; init=_random_normal),
                                                        BatchNorm(out_chs), 
                                                        x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))

ConvUp(in_chs,out_chs;kernel = (5,5), stride=(2,2), pad=2, σ=elu) = Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                                    ConvTranspose(kernel, in_chs=>out_chs, stride=stride, pad = pad; init=_random_normal), 
                                                    BatchNorm(out_chs))

  
EncoderDownSampleBlock(in_channels, out_channels; kernel=(5,5), pad=2, σ=elu) = Chain(
                                    Conv(kernel, in_channels=>in_channels, stride=(2,2), pad=pad; init=_random_normal),
                                    Chain(
                                        Conv(kernel, in_channels=>out_channels, stride=(1,1), pad = pad; init=_random_normal),
                                        BatchNorm(out_channels), 
                                        x->(σ == elu ? σ.(x,0.2f0) : σ.(x))
                                    ),
                                    Chain(
                                        Conv(kernel, out_channels=>out_channels, stride=(1,1), pad = pad; init=_random_normal),
                                        BatchNorm(out_channels), 
                                        x->(σ == elu ? σ.(x,0.2f0) : σ.(x))
                                    )
                                )                                                    


struct UNetUpBlock
    upsample
  end
  
  (u::UNetUpBlock)(input, bridge) = cat(u.upsample(input), bridge, dims = 3)
  
  @functor UNetUpBlock
  
  UNetUpBlock(in_chs::Int, out_chs::Int; kernel = (5, 5), p = 0.5f0, σ=elu) =
      UNetUpBlock(Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                      ConvTranspose(kernel, in_chs=>out_chs, stride=(2, 2), pad = 2; init=_random_normal),
                      BatchNorm(out_chs)))|> cgpu
  

# relevant models
struct FFSDNUnet
    conv_down_blocks
    conv_blocks
    up_blocks
    # alpha
end
  
@functor FFSDNUnet

function FFSDNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
            ConvDown(32,32;σ=σ),
            ConvDown(64,64;σ=σ),
            ConvDown(128,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(16,16; kernel = kernel, σ=σ),
        resnet_type(32,32; kernel = kernel, σ=σ),
        resnet_type(64,64; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(256, 64; σ=σ),
                        UNetUpBlock(256, 32; σ=σ),
                        UNetUpBlock(128, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 64=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    
    # alpha = [0.001] |> cgpu
    FFSDNUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::FFSDNUnet)(x::AbstractArray, features)

    # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](x))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](cat(op, features[1], dims=3)))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](cat(x1, features[2], dims=3)))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[4](u.conv_down_blocks[4](cat(x2, features[3], dims=3)))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[5](x3)
    up_x3 = u.conv_blocks[6](up_x3)
    up_x3 = u.conv_blocks[7](up_x3)

    # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
    up_x1 = u.up_blocks[1](cat(up_x3, features[4], dims=3), x2)
    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
    up_x2 = u.up_blocks[2](cat(up_x1, features[5], dims=3), x1)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.up_blocks[3](cat(up_x2, features[6], dims=3), op)
    # (n/2) X (n/2) X 32 X bs -> (n/2) X (n/2) X 16 X bs
    up_x5 = u.up_blocks[4](cat(up_x4, features[7], dims=3))

    # (n/2) X (n/2) X 16 X bs -> n X n X 2 X bs
    return u.up_blocks[end](up_x5)
end


# TFFKappa: an encoder that prepares hierarchical inputand uses double ResNet steps

struct TFFKappa
    conv_down_blocks
    conv_blocks
    up_blocks
end

@functor TFFKappa

function TFFKappa(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
            ConvDown(16,32;σ=σ),
            ConvDown(32,64;σ=σ),
            ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(16,16; kernel = kernel, σ=σ),
        resnet_type(16,16; kernel = kernel, σ=σ),
        resnet_type(32,32; kernel = kernel, σ=σ),
        resnet_type(32,32; kernel = kernel, σ=σ),
        resnet_type(64,64; kernel = kernel, σ=σ),
        resnet_type(64,64; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ),
        resnet_type(16,16; kernel = kernel, σ=σ),
        resnet_type(32,32; kernel = kernel, σ=σ),
        resnet_type(64,64; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    TFFKappa(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::TFFKappa)(x::AbstractArray)
    # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[2](u.conv_blocks[1](u.conv_down_blocks[1](x)))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[4](u.conv_blocks[3](u.conv_down_blocks[2](op)))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[6](u.conv_blocks[5](u.conv_down_blocks[3](x1)))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[8](u.conv_blocks[7](u.conv_down_blocks[4](x2)))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[9](x3)
    up_x3 = u.conv_blocks[10](up_x3)
    up_x3 = u.conv_blocks[11](up_x3)

    # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
    up_x1 = u.conv_blocks[15](u.up_blocks[1](up_x3, x2))
    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
    up_x2 = u.conv_blocks[14](u.up_blocks[2](up_x1, x1))

    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.conv_blocks[13](u.up_blocks[3](up_x2, op))

    return [op, x1, x2, up_x3, up_x1, up_x2, up_x4]
end


# Input as is without activation and batch normalization
# and doubling the channels and reducing back between the 2 convolution actions

struct TSResidualBlockI
    layers
    activation
end

(r::TSResidualBlockI)(input) = r.layers(r.activation(input)) + input

@functor TSResidualBlockI

function TSResidualBlockI(in_chs::Int, out_chs::Int; kernel = (3, 3), pad =1, σ=elu)
    layers = Chain(Conv(kernel, in_chs=>(2*in_chs), pad = pad; init=_random_normal),
                    BatchNorm(2*in_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, (2*in_chs)=>out_chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(BatchNorm(out_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    TSResidualBlockI(layers, activation)|> cgpu
end

# Preservation of input without convolution

struct SResidualBlock
    layers
    activation
end

(r::SResidualBlock)(input) = r.activation(r.layers(input) + input)

@functor SResidualBlock

function SResidualBlock(in_chs::Int, out_chs::Int; kernel = (3, 3), pad =1, σ=elu)
    layers = Chain(Conv(kernel, in_chs=>in_chs, pad = pad; init=_random_normal),
                    BatchNorm(in_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, in_chs=>out_chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(BatchNorm(out_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    SResidualBlock(layers, activation)|> cgpu
end


# FeaturesUNet: the encoder prepares hierarchical input for the solver network

struct FeaturesUNet
    kappa_subnet
    solve_subnet
    indexes::Int64
end

@functor FeaturesUNet

function FeaturesUNet(in_chs::Int64, k_chs::Int64, s_model::DataType, k_model::DataType; kernel = (3, 3), indexes=3, σ=elu, resnet_type=SResidualBlock)
    kappa_subnet = k_model(indexes-2,k_chs;kernel=kernel,σ=σ,resnet_type=resnet_type)
    solve_subnet = s_model(in_chs,2;kernel=kernel,σ=σ,resnet_type=resnet_type)
    FeaturesUNet(kappa_subnet, solve_subnet, indexes)
end

function (u::FeaturesUNet)(x; in_tuning=false)
    # if in_tuning == true
    #     kappa =  reshape(x[:,:,3:u.indexes,1], size(x,1), size(x,2), u.indexes-2, 1)
    #     features = repeat.(u.kappa_subnet(kappa), 1, 1, 1, size(x,4)) # error in GPU
    # else
    #     kappa = reshape(x[:,:,3:u.indexes,:], size(x,1), size(x,2), u.indexes-2, size(x,4))
    #     features = u.kappa_subnet(kappa)
    # end
    kappa = reshape(x[:,:,3:u.indexes,:], size(x,1), size(x,2), u.indexes-2, size(x,4))
    features = u.kappa_subnet(kappa)
    u.solve_subnet(x, features)
end
# end of relevant models


function create_model!(e_vcycle_input,kappa_input,gamma_input;kernel=(3,3),type=SUnet,k_type=NaN,resnet_type=SResidualBlock,k_chs=-1, indexes=3, σ=elu, arch=0)
    input = 2
    if e_vcycle_input == true
        input = input+2
    end
    if kappa_input == true
        input = input+1
    end
    if gamma_input == true
        input = input+1
    end

    if arch == 0 # A stand-alone U-Net
        return type(input,2;kernel=kernel,σ=σ,resnet_type=resnet_type)
    else
        if arch == 2 # Encoder with a hierarchical context
            @info "$(Dates.format(now(), "HH:MM:SS")) - FeaturesUNet"
            return FeaturesUNet(input,k_chs,type,k_type;kernel=kernel,indexes=indexes,σ=σ,resnet_type=resnet_type)
        else # Encoder with a simple context
            @info "$(Dates.format(now(), "HH:MM:SS")) - SplitUNet $(input) $(k_chs)"
            return SplitUNet(input,k_chs,type,k_type;kernel=kernel,indexes=indexes,σ=σ,resnet_type=resnet_type)
        end
    end
end


# Bar and Ido's model
# Encoder: half unet netwrok (only encoding down-stream)
# struct Encoder
#     conv_in_8
#     downsample_8_16
#     downsample_16_32
#     downsample_32_64
#     downsample_64_128
# end

# @functor Encoder
# function Encoder(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
#     conv_in_8 = Chain(
#         Conv(kernel, channels=>8, stride=(1,1), pad=1; init=_random_normal),
#         BatchNorm(8)
#     )|>cgpu
    
#     downsample_8_16 = EncoderDownSampleBlock(8, 16)|>cgpu
#     downsample_16_32 = EncoderDownSampleBlock(16, 32)|>cgpu
#     downsample_32_64 = EncoderDownSampleBlock(32, 64)|>cgpu
#     downsample_64_128 = EncoderDownSampleBlock(64, 128)|>cgpu

#     Encoder(conv_in_8, downsample_8_16, downsample_16_32, downsample_32_64, downsample_64_128)
# end

# function (u::Encoder)(x::AbstractArray)
#     # nxmxinxbs => nxmx8xbs
#     x_8 = u.conv_in_8(x)
#     # nxmx8xbs => n/2xm/2x16xbs
#     x_16 = u.downsample_8_16(x_8)
#     # n/2xm/2x16xbs => n/4xm/4x32xbs
#     x_32 = u.downsample_16_32(x_16)
#     # n/4xm/4x32xbs => n/8xm/8x64xbs
#     x_64 = u.downsample_32_64(x_32)
#     # n/8xm/8x64xbs => n/16xm/16x128xbs
#     x_128 = u.downsample_64_128(x_64)

#     return [x_16, x_32, x_64, x_128]
# end

# struct bottleNeckBlock
#     expand_layer
#     depth_layer
#     shrink_layer
# end

# @functor bottleNeckBlock

# function bottleNeckBlock(in_channels, expanded_channels; kernel=(3,3), pad=1, σ=elu)
#     expand_layer = Chain(
#         Conv((1,1), in_channels=>expanded_channels, stride=(1,1), pad=0; init=_random_normal),
#         BatchNorm(expanded_channels)
#     )|>cgpu
#     depth_layer = Chain(
#         Conv(kernel, expanded_channels=>expanded_channels, pad=pad, stride=(1,1), groups=expanded_channels; init=_random_normal),
#         BatchNorm(expanded_channels)
#     )|> cgpu
#     shrink_layer = Chain(
#         Conv((1,1), expanded_channels=>in_channels, stride=(1,1), pad = 0; init=_random_normal),
#         BatchNorm(in_channels), 
#         x->(σ == elu ? σ.(x,0.2f0) : σ.(x))
#     )|>cgpu

#     bottleNeckBlock(expand_layer, depth_layer, shrink_layer)
# end

# function (u::bottleNeckBlock)(x::AbstractArray, encoded_vector::AbstractArray)
#     x = u.expand_layer(x)
#     x = u.depth_layer(x) + encoded_vector
#     return u.shrink_layer(x)
# end

# # solver
# struct Solver
#     downsample_in_8 
#     downsample_8_16
#     downsample_16_32
#     downsample_32_64
#     bottleneck_8
#     bottleneck_16
#     bottleneck_32
#     bottleneck_64
#     conv_64
#     conv_32
#     conv_16
#     conv_8
#     upsample_64_32
#     upsample_32_16
#     upsample_16_8
#     upsample_8_out
# end

# @functor Solver

# function Solver(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
#     # nxmxinxbs => n/2xm/2x8xbs
#     downsample_in_8 = Chain(
#         Conv((5,5), channels=>8, stride=(2,2), pad = 2; init=_random_normal),
#         BatchNorm(8), 
#         x->(σ == elu ? σ.(x,0.2f0) : σ.(x))
#     ) |> cgpu

#     # n/2xm/2x8xbs => n/4xm/4x16xbs
#     downsample_8_16 = Chain(
#         Conv((5,5), 8=>16, stride=(2,2), pad = 2; init=_random_normal),
#         BatchNorm(16),
#         x->(σ == elu ? σ.(x,0.2f0) : σ.(x))
#     ) |> cgpu

#     # n/4xm/4x16xbs => n/8xm/8x32xbs
#     downsample_16_32 = Chain(
#         Conv((5,5), 16=>32, stride=(2,2), pad = 2; init=_random_normal),
#         BatchNorm(32),
#         x->(σ == elu ? σ.(x,0.2f0) : σ.(x))
#     ) |> cgpu

#     # n/8xm/8x32xbs => n/16xm/16x64xbs
#     downsample_32_64 = Chain(
#         Conv((5,5), 32=>64, stride=(2,2), pad = 2; init=_random_normal),
#         BatchNorm(64),
#         x->(σ == elu ? σ.(x,0.2f0) : σ.(x))
#     ) |> cgpu

#     bottleneck_8 = bottleNeckBlock(8,16) |> cgpu
#     bottleneck_16 = bottleNeckBlock(16,32) |> cgpu
#     bottleneck_32 = bottleNeckBlock(32,64) |> cgpu
#     bottleneck_64 = bottleNeckBlock(64,128) |> cgpu

#     conv_64 = Conv((1,1), 64=>64, stride=(1,1), pad = 0, groups=64; init=_random_normal) |> cgpu

#     conv_32 = Chain(
#         Conv((5,5), 64=>32, stride=(1,1), pad = 2; init=_random_normal),
#         BatchNorm(32),
#         x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
#         Conv((5,5), 32=>32, stride=(1,1), pad = 2; init=_random_normal),
#         BatchNorm(32),
#         x->(σ == elu ? σ.(x,0.2f0) : σ.(x))
#     ) |> cgpu

#     conv_16 = Chain(
#         Conv((5,5), 32=>16, stride=(1,1), pad = 2; init=_random_normal),
#         BatchNorm(16),
#         x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
#         Conv((5,5), 16=>16, stride=(1,1), pad = 2; init=_random_normal),
#         BatchNorm(16),
#         x->(σ == elu ? σ.(x,0.2f0) : σ.(x))
#     ) |> cgpu

#     conv_8 = Chain(
#         Conv((5,5), 16=>8, stride=(1,1), pad = 2; init=_random_normal),
#         BatchNorm(8),
#         x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
#         Conv((5,5), 8=>8, stride=(1,1), pad = 2; init=_random_normal),
#         BatchNorm(8),
#         x->(σ == elu ? σ.(x,0.2f0) : σ.(x))
#     ) |> cgpu

#     upsample_64_32 = ConvTranspose((5,5), 64=>32, stride=(2,2), pad = 2; init=_random_normal) |> cgpu
#     upsample_32_16 = ConvTranspose((5,5), 32=>16, stride=(2,2), pad = 2; init=_random_normal) |> cgpu
#     upsample_16_8 = ConvTranspose((5,5), 16=>8, stride=(2,2), pad = 2; init=_random_normal) |> cgpu
#     upsample_8_out = ConvUp(8, labels; σ=σ) |> cgpu
    

#     Solver(
#         downsample_in_8, downsample_8_16, downsample_16_32, downsample_32_64,
#         bottleneck_8, bottleneck_16, bottleneck_32, bottleneck_64,
#         conv_64, conv_32, conv_16, conv_8,
#         upsample_64_32, upsample_32_16, upsample_16_8, upsample_8_out
#     )
# end

# function (u::Solver)(x::AbstractArray, encoded_vectors)
#     x_8 = u.bottleneck_8(u.downsample_in_8(x), encoded_vectors[1])
#     x_16 = u.bottleneck_16(u.downsample_8_16(x_8), encoded_vectors[2])
#     x_32 = u.bottleneck_32(u.downsample_16_32(x_16), encoded_vectors[3])
#     # => n/16xm/16x64xbs
#     x_64 = u.bottleneck_64(u.downsample_32_64(x_32), encoded_vectors[4])

#     x_64_up = u.conv_64(x_64)

#     x_32_up = u.upsample_64_32(x_64_up)
#     x_32_up = u.conv_32(cat(x_32_up, x_32, dims=3))

#     x_16_up = u.upsample_32_16(x_32_up)
#     x_16_up = u.conv_16(cat(x_16_up, x_16, dims=3))

#     x_8_up = u.upsample_16_8(x_16_up)
#     x_8_up = u.conv_8(cat(x_8_up, x_8, dims=3))
    
#     return u.upsample_8_out(x_8_up)
# end