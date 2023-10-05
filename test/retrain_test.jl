using DelimitedFiles
using PyPlot

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

filename = "SEGmodel2Dsalt.dat"
doTranspose = true

n = 608
m = 304
newSize = [n+1, m+1]

m = readdlm(filename);
m = m*1e-3;

if doTranspose
    m = m';
end

m    = expandModelNearest(m,   collect(size(m)),newSize);

println(size(m))
figure()
imshow(m', clim = [1.5,4.5],cmap = "jet"); colorbar();
savefig("m.png")
close()