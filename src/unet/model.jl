include("../../src/unet/utils.jl")

BatchNormWrap(out_ch) = Chain(x->expand_dims(x,2)|>cgpu, 
                                BatchNorm(out_ch)|> cgpu,
                                x->squeeze(x))|> cgpu


# check with pad=2 and in up to.
ConvDown(in_chs,out_chs;kernel = (5,5), σ=elu) = Chain(Conv(kernel, in_chs=>out_chs, stride=(2,2), pad = 2; init=_random_normal),
                                                        BatchNorm(out_chs), 
                                                        x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu

ConvUp(in_chs,out_chs;kernel = (5,5), σ=elu) = Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                                    ConvTranspose(kernel, in_chs=>out_chs, stride=(2, 2), pad = 2; init=_random_normal), 
                                                    BatchNorm(out_chs))|> cgpu

                                                    
struct UNetUpBlock
  upsample
end

(u::UNetUpBlock)(input, bridge) = cat(u.upsample(input), bridge, dims = 3)

@functor UNetUpBlock

UNetUpBlock(in_chs::Int, out_chs::Int; kernel = (5, 5), p = 0.5f0, σ=elu) =
    UNetUpBlock(Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    ConvTranspose(kernel, in_chs=>out_chs, stride=(2, 2), pad = 2; init=_random_normal),
                    BatchNorm(out_chs)))|> cgpu

UNetConvBlock(in_chs, out_chs; kernel = (3, 3), pad=1, σ=elu) =
    Chain(Conv(kernel, in_chs=>out_chs, pad=pad; init=_random_normal),
                BatchNorm(out_chs),
                x-> (σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu


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

    # n X n X 4 X bs + n X n X 16 X bs-> (n/2) X (n/2) X 16 X bs
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

    # if boundary_gamma === nothing
    #     return u.up_blocks[end](up_x5)
    # end
    # e = (1 .- (u.alpha .* boundary_gamma)) |> cgpu
    return u.up_blocks[end](up_x5)
end

# FFKappa + FFSDNUnet: an encoder that prepares hierarchical input

struct FFKappa
    conv_down_blocks
    conv_blocks
    up_blocks
end
  
@functor FFKappa

function FFKappa(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
            ConvDown(16,32;σ=σ),
            ConvDown(32,64;σ=σ),
            ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(16,16; kernel = kernel, σ=σ),
        resnet_type(32,32; kernel = kernel, σ=σ),
        resnet_type(64,64; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ),
        resnet_type(128,128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    FFKappa(conv_down_blocks, conv_blocks, up_blocks)
end


function (u::FFKappa)(x::AbstractArray)

    # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](x))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](op))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](x1))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[4](u.conv_down_blocks[4](x2))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[5](x3)
    up_x3 = u.conv_blocks[6](up_x3)
    up_x3 = u.conv_blocks[7](up_x3)

    # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
    up_x1 = u.up_blocks[1](up_x3, x2)
    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
    up_x2 = u.up_blocks[2](up_x1, x1)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.up_blocks[3](up_x2, op)

    return [op, x1, x2, up_x3, up_x1, up_x2, up_x4]
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