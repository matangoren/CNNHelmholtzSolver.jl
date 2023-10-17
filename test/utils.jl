using PyPlot
using CSV, DataFrames, DelimitedFiles


function expandModelNearest(m,n,ntarget)
    if length(size(m))==2
        mnew = zeros(Float64,ntarget[1],ntarget[2]);
        for j=1:ntarget[2]
            for i=1:ntarget[1]
                jorig = convert(Int64,ceil((j/ntarget[2])*n[2]));
                iorig = convert(Int64,ceil((i/ntarget[1])*n[1]));
                mnew[i,j] = m[iorig,jorig];
            end
        end
    elseif length(size(m))==3
        mnew = zeros(Float64,ntarget[1],ntarget[2],ntarget[3]);
        for k=1:ntarget[3]
            for j=1:ntarget[2]
                for i=1:ntarget[1]
                    korig = convert(Int64,floor((k/ntarget[3])*n[3]));
                    jorig = convert(Int64,floor((j/ntarget[2])*n[2]));
                    iorig = convert(Int64,floor((i/ntarget[1])*n[1]));
                    mnew[i,j,k] = m[iorig,jorig,korig];
                end
            end
        end
    end
    return mnew
end

function get_seg_model(kappa_file, n,m; doTranspose=false)
    newSize = [n+1, m+1]
    medium = readdlm(kappa_file);
    # medium = medium*1e-3;
    if doTranspose
        medium = medium';
    end
    medium = expandModelNearest(medium,   collect(size(medium)),newSize);

    return r_type.(medium) # m is velocity model
end

function get_rhs(n, m, h; blocks=2)
    rhs = zeros(ComplexF64,n+1,m+1,1,1)
    rhs[floor(Int32,n / 2.0),floor(Int32,m / 2.0),1,1] = r_type(1.0 ./minimum(h)^2)
    rhs = vec(rhs)
    if blocks == 1
        return reshape(rhs, (length(rhs),1))
    end
    for i = 2:blocks
        rhs1 = zeros(ComplexF64,n+1,m+1,1,1)
        rhs1[floor(Int32,(n / blocks)*(i-1)),floor(Int32,(m / blocks)*(i-1)),1,1] = r_type(1.0 ./minimum(h)^2)
        rhs = cat(rhs, vec(rhs1), dims=2)
    end
    return rhs
end

function plot_results(filename, res, n, m)
    heatmap(real(reshape(res[:,1],n+1, m+1)), color=:blues)
    savefig(filename)
end

function get_setup(n,m,domain, f_fwi; blocks=4, kappa_file="")
    h = r_type.([(domain[2]-domain[1])./ n, (domain[4]-domain[3])./ m])   

    if kappa_file != ""
        seg_model = get_seg_model(kappa_file, n,m; doTranspose=false) # velocity model
        kappa_i = velocityToSlowness(seg_model)
        medium = kappa_i.^2

        # figure()
        # imshow(seg_model', clim = [1.5,4.5],cmap = "jet"); colorbar();
        # savefig("$(kappa_file).png")
        # close()
    else # default linear kappa
        kappa_i, _ = get2DSlownessLinearModel(n,m;normalized=false)
        medium = kappa_i.^2
    end
    
    c = maximum(kappa_i)
    omega_exact = r_type((0.1*2*pi) / (c*maximum(h)))
    omega_fwi = r_type(2*pi*f_fwi)
    omega = omega_exact * c
    
    ABLpad = 16
    ABLamp = omega_exact
    gamma = r_type.(getABL([n+1,m+1],true,ones(Int64,2)*ABLpad,Float64(ABLamp)))
    attenuation = r_type(0.01*4*pi);
    gamma .+= attenuation

    M = getRegularMesh(domain,[n;m])

    rhs = get_rhs(n,m,h; blocks=blocks) # move outside
    return HelmholtzParam(M,Float64.(gamma),Float64.(medium),Float64(omega_fwi),true,useSommerfeldBC), rhs
end


function convergence_factor_106!(vector)
    val0 = vector[1]
    index = length(vector)
    for i=2:length(vector)
        if vector[i] < (val0 ./ (10^6))
            index = i
            break
        end
    end
    # length = argmin(vector)[1]
    if index > 200
        return round(((vector[index] / vector[index-30])^(1.0 / 30)), digits=3), index
    else
        return round(((vector[index] / vector[1])^(1.0 / index)), digits=3), index
    end
end

function VU_V_graph(title, histories, labels, colors; scale="log")

    figure()

    for i in 1:length(histories)
        h = histories[i]
        l = labels[i]
        c = colors[i]
        factor,index = convergence_factor_106!(h)
        plot(range(1,index),h[1:index],label=latexstring("\$\\rho_{$(l)}=$(factor) ($(index))\$"), color=c)
    end


    legend(loc="upper right")
    xlabel("iterations")
    ylabel(L"\Vert b - Hx \Vert_2")
    yscale(scale)
    savefig(joinpath(@__DIR__, "plots/$(title)"))
    close()
end


function getIterationHistory(df, fromFunction)
    df = df[df.FromFunction .== fromFunction, :]
    omegas = unique!(df[:, "Omega"])
    iterations_vectors = []
    for omega in omegas
        df_w = df[df.Omega .== omega, :]
        append!(iterations_vectors, [df_w[:, "Iterations"]])
    end

    return iterations_vectors
end

filename = "dataset_608X304_gamma_16_ABLamp_1_120_solver_info_with_retraining_1e-6.csv"
df = DataFrame(CSV.File(filename))

v_getData = getIterationHistory(df, "getData")
v_SensMatVec = getIterationHistory(df, "SensMatVec")
v_SensTMatVec = getIterationHistory(df, "SensTMatVec")

VU_V_graph("iterations",[v_getData[3]', v_getData[4]', v_getData[5]'], ["VU", "V", "VU-retraining"], ["blue", "green", "orange"]; scale="linear")


