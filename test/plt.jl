using PyPlot

figure()
x = range(0,2*pi,1000); y = sin.(3*x + 4*cos.(2*x))
x2 = range(0,2*pi,1000); y2 = sin.(3*x + 4*cos.(2*x)).+1
l="V"
println()
plot(x, y, color="red", linewidth=2.0, label=latexstring("\$\\rho_{$(l)}=$(0.5)\$"), linestyle="--")
plot(x2, y2, color="blue", linewidth=2.0, label="2", linestyle="--")

legend(loc="upper right")
xlabel("iterations")
ylabel(L"\Vert b - Hx \Vert_2")
yscale("log")
savefig(joinpath(@__DIR__, "plots/$(title)"))
close()