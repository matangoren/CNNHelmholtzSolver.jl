use_gpu = false
if use_gpu == true
    using CUDA
    # CUDA.allowscalar(false)
    cgpu = gpu
else
    cgpu = cpu
    pyplot()
end

pu = cpu # gpu
r_type = Float64
c_type = ComplexF64
u_type = Float32
gmres_type = ComplexF64
# a_type = CuArray{gmres_type}
a_type = Array{gmres_type}
run_title = "64bit_cpu"