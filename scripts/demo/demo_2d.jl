## Simple demo on 2D problem
# 
# Lorenzo Shaikewitz, 9/29/2025

using PoseUncertaintySets
using LinearAlgebra
import Plots

# constraints
center = [-0.2; -0.2]
A1 = [0 0.5 0; 0.5 1 0; 0 0 1] # x² + y² + x ≤ 0
A2 = [0 0 0.5; 0 1 0; 0.5 0 0] # x² + y ≤ 0
A3 = [-0.5 0 -0.5; 0 1 0; -0.5 0 0] # x² - y - 0.5 ≤ 0
# A4 = [0.05 0 0; 0 -1 0; 0 0 1] # y² - x² + 0.05 ≤ 0
As = [A1, A2, A3]

# this is a cool example: you can see it get tighter up to order 3
# and maybe a good logo?
# center = [-0.6; -0.2]
# A1 = [0 0.5 0; 0.5 1 0; 0 0 1] # x² + y² + x ≤ 0
# A2 = [0.2 0 1; 0 -0.5 0; 1 0 0] # -0.5x² + 2y + 0.2 ≤ 0
# A3 = [0 0 -0.5; 0 -1.5 0; -0.5 0 0] # -x² - y ≤ 0
# As = [A1, A2, A3]

# do AeroAstro logo?
# center = [-0.65; -0.2]
# A1 = [0 0.5 0; 0.5 1 0; 0 0 1] # x² + y² + x ≤ 0
# A2 = [0.2 0 1; 0 -0.5 0; 1 0 0] # -0.5x² + 2y + 0.2 ≤ 0
# A3 = [0.3 0.5 -0.9; 0.5 0 0; -0.9 0 -5] # -x² - y ≤ 0
# As = [A1, A2, A3]

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
# A1 = [-0.2 0.2 0; 0.2 1 0; 0 0 1]
# As = [A1]
Bs = []

function certify(H, center, As, Bs)
    # check if any point on the ellipse intersects the set boundary.
    # ellipse: (x - center)'*H*(x - center) = 1
    num_pts = 500
    x, y = ellipse_to_surf_2d(H, center, num_pts)
    z = [ones(num_pts)  x  y]

    tol = 1e-3
    vals = zeros(num_pts, length(As))
    sat = zeros(Bool, num_pts)
    for (i,A) in enumerate(As)
        vals[:,i] = [zi'*A*zi for zi in eachrow(z)]
        sat .&= vals[:,i] .<= tol
    end
    optimal = false
    location = nothing
    sat = sum(eachcol(vals .<= tol)) .== length(As)
    if maximum(sat) 
        optimal = true
        location = z[argmax(sat),2:3]
    end
    return optimal, location
end

plts = []
Ms = []
Xs = []
Hs = []
vols = []
cert = []
for order = 1:4
    # solve
    H, status, M, X = bound_2d(center, As, Bs; order=order, silent=true)
    println("Status: $status")
    printstyled("Eigvals\n",underline=true)
    display(["Dual"  "Dual Dual"; eigvals(M)[1:3]  eigvals(X)[end-2:end]])
    optimal, location = certify(H, center, As, Bs)
    push!(cert, location)

    # plot solution
    Plots.plot(ellipse_to_surf_2d(H, center, 100), label=false)
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

    if !isnothing(cert[end])
        Plots.scatter!([location[1]], [location[2]], label=false, ma=0.5)
    end
    
    p=Plots.plot!(xlims=[-1.5,0.5], ylims=(-1,1), aspect_ratio=:equal)
    # p = Plots.plot!(lims=[-3,3], aspect_ratio=:equal)
    push!(plts, p)

    push!(Hs, H)
    push!(vols, 1/logdet(H))
    push!(Ms, M)
    push!(Xs, X)
end
printstyled("Certificates\n",underline=true)
display(["Vols"  "cert"; vols cert])

# eigvals
ev_ratio = [eigvals(X)[end-1] / eigvals(X)[end] for X in Xs]
p_evs = Plots.plot(ev_ratio, xlabel="order", ylabel="λ₂/λ₁")

# all on one plot
p_all = Plots.scatter([center[1]], [center[2]],c=1; label=false)
# plot constraints
xs = range(-3,3,length=300)
ys = range(-3,3,length=300)
for A in As
    Plots.contour!(xs, ys, (x,y)->[1;x;y]'*A*[1;x;y], levels=[0], colorbar=false)
end
for B in Bs
    Plots.contour!(xs, ys, (x,y)->[1;x;y]'*B*[1;x;y], levels=[0], colorbar=false)
end
for (i,H) in enumerate(Hs)
    Plots.plot!(ellipse_to_surf_2d(H, center, 100), label="order $i")
end
Plots.plot!(xlims=[-1.5,0.5], ylims=(-1,1), aspect_ratio=:equal)


Plots.plot(plts...)