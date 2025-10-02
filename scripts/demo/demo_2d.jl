## Simple demo on 2D problem
# 
# Lorenzo Shaikewitz, 9/29/2025

using PoseUncertaintySets
using LinearAlgebra
import Plots

# constraints
# center = [-0.2; -0.2]
# A1 = [0 0.5 0; 0.5 1 0; 0 0 1] # x² + y² + x ≤ 0
# A2 = [0 0 0.5; 0 1 0; 0.5 0 0] # x² + y ≤ 0
# A3 = [-0.5 0 -0.5; 0 1 0; -0.5 0 0] # x² - y - 0.5 ≤ 0
# # A4 = [0.05 0 0; 0 -1 0; 0 0 1] # y² - x² + 0.05 ≤ 0
# As = [A1, A2, A3]

# this is a cool example: you can see it get tighter up to order 3
# and maybe a good logo?
center = [-0.6; -0.2]
A1 = [0 0.5 0; 0.5 1 0; 0 0 1] # x² + y² + x ≤ 0
A2 = [0.2 0 1; 0 -0.5 0; 1 0 0] # -0.5x² + 2y + 0.2 ≤ 0
A3 = [0 0 -0.5; 0 -1 0; -0.5 0 0] # -x² - y ≤ 0
As = [A1, A2, A3]

# box
# center = [0; 0]
# A1 = [-4. 0 0; 0 1 0; 0 0 1] # x² + y² - 4 ≤ 0
# # A2 = [-1 0 0.5; 0 0 0; 0.5 0 0] 
# A2 = -[1.1 0.1 -0.5; 0.1 0.1 0; -0.5 0 0]
# # A3 = [-1 0 -0.5; 0 0 0; -0.5 0 0]
# A3 = [-1 0 -0.5; 0 -0.1 0; -0.5 0 0]
# # A4 = [-1 0.5 0; 0.5 0 0; 0 0 0]
# A4 = [-2.5 0.5 0; 0.5 0 0; 0 0 1.1]
# # A5 = [-1 -0.5 0; -0.5 0 0; 0 0 0]
# A5 = [-0.7 -0.5 0; -0.5 -0.072 0.12; 0 0.12 -0.2]
# As = [A1, A2, A3, A4, A5]

# center = [-0; -0]
# # A1 = [0 0.5 0; 0.5 1 0; 0 0 3] # x² + y² + x ≤ 0
# A1 = [-1 0 0; 0 1 0; 0 0 1]
# B1 = [0 0.5 0; 0.5 5 0; 0 0 5] # x² + y² + x ≤ 0
# As = [A1]
Bs = []

plts = []
Ms = []
Xs = []
Hs = []
vols = []
for order = 1:4
    # solve
    H, status, M, X = bound_2d(center, As, Bs; order=order, silent=true)
    println("Status: $status")
    printstyled("Eigvals\n",underline=true)
    display(["Dual"  "Dual Dual"; eigvals(M)[1:3]  eigvals(X)[end-2:end]])

    # plot solution
    Plots.plot(ellipse_to_surf_2d(H, center), label=false)
    Plots.scatter!([center[1]], [center[2]],c=1; label=false)
    # plot constraints
    xs = range(-3,3,length=300)
    ys = range(-3,3,length=300)
    for A in As
        Plots.contour!(xs, ys, (x,y)->[1;x;y]'*A*[1;x;y], levels=[0], colorbar=false)
    end
    for B in Bs
        Plots.contour!(xs, ys, (x,y)->[1;x;y]'*B*[1;x;y], levels=[0], colorbar=false)
    end
    p=Plots.plot!(xlims=[-1.5,0.5], ylims=(-1,1), aspect_ratio=:equal)
    # p = Plots.plot!(lims=[-3,3], aspect_ratio=:equal)
    push!(plts, p)

    push!(Hs, H)
    push!(vols, 1/logdet(H))
    push!(Ms, M)
    push!(Xs, X)
end

# eigvals
ev_ratio = [eigvals(X)[end-1] / eigvals(X)[end] for X in Xs]
p_evs = Plots.plot(ev_ratio, xlabel="order", ylabel="λ₂/λ₁")


Plots.plot(plts...)
# TODO: plot all on one plot...