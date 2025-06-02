## Check the ellipse and its marginalizations against samples
# run sample first
# Lorenzo Shaikewitz, 4/30/2025

# ellipse_dict = deserialize("ellipse10_fromlinf")
# H = ellipse_dict[9]["all_H"][frame]
# H_t = ellipse_dict[9]["all_H_ts"][frame]
# center = [vec(R_center); t_center]

# check against the samples
cover_both = zeros(Bool, size(poses,1))
cover_t = zeros(Bool, size(poses,1))
cover_t_const_r = zeros(Bool, size(poses,1))
# cover_r = zeros(Bool, size(poses,1))
ang_deviation = zeros(size(poses,1))
t_deviation = zeros(3,size(poses,1))
for (i,p) in enumerate(poses)
    (R, t) = p
    x = [vec(R); t]

    # check both
    if (x - center)'*H*(x - center) <= 1
        cover_both[i] = true
    end
    # check marginalized t
    if (t - center[10:12])'*H_t*(t - center[10:12]) <= 1
        cover_t[i] = true
    end
    # check const r
    if (t - center[10:12])'*H[10:12,10:12]*(t - center[10:12]) <= 1
        cover_t_const_r[i] = true
    end

    # update deviations
    Rc = reshape(center[1:9],3,3)
    ang_deviation[i] = roterror(R,Rc)
    tc = center[10:12]
    t_deviation[:,i] = tc - t
end

printstyled("Coverage of samples\n", underline=true)
@printf "full ellipse: %d/%d (%.2f%%)\n" sum(cover_both) size(poses,1) (sum(cover_both)/size(poses,1)*100)
@printf "t marginal: %d/%d (%.2f%%)\n" sum(cover_both) size(poses,1) (sum(cover_both)/size(poses,1)*100)
@printf "constant R: %d/%d (%.2f%%)\n" sum(cover_both) size(poses,1) (sum(cover_both)/size(poses,1)*100)

@printf "Angular deviation: %.2f deg\n" maximum(ang_deviation)
@printf "Translation deviation: %.2f, %.2f, %.2f mm\n" maximum(abs.(t_deviation[1,:]))*1000 maximum(abs.(t_deviation[2,:]))*1000 maximum(abs.(t_deviation[3,:]))*1000

## Plot translation
function generate_ellipse(A, c, num_points=100)
    # Eigenvalue decomposition of A
    λ, V = eigen(Symmetric(A))
        
    # Generate points on unit sphere
    u = LinRange(0, 2π, num_points)
    v = LinRange(0, π, num_points)
    x = zeros(num_points, num_points)
    y = zeros(num_points, num_points)
    z = zeros(num_points, num_points)
    for (i,ui) in enumerate(u)
        # Parametric equations of the unit sphere
        x[:,i] = cos(ui)*sin.(v)
        y[:,i] = sin(ui)*sin.(v)
        z[:,i] = cos.(v)
    end
    sphere = [vec(x) vec(y) vec(z)]' # 3 x num_points^2

    # Transform sphere to ellipsoid
    axes_lengths = 1. ./ sqrt.(λ)
    surf = V * diagm(axes_lengths) * sphere
    surf .+= c
    return surf
end

# remove duplicates
# i = unique(i -> surf[1:2,i], eachindex(eachcol(surf)))



Plots.plot([center[10]],[center[11]],[center[12]], seriestype=:scatter, label="center")
surf = generate_ellipse(H[10:12,10:12], center[10:12])
Plots.scatter!([0],[0],[0],label="camera")
Plots.plot!(eachrow(good_centers)...,seriestype=:scatter,zcolor=good_margins, color=:blues, label="samples")
p1 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="Ellipse", xlabel="X", ylabel="Y", zlabel="Z", title="Constant Rotation")

Plots.plot([center[10]],[center[11]],[center[12]], seriestype=:scatter, label="center")
surf = generate_ellipse(H_t, center[10:12], 100)
Plots.scatter!([0],[0],[0],label="camera")
Plots.plot!(eachrow(good_centers)...,seriestype=:scatter,zcolor=good_margins, color=:blues, label="samples")
p2 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], msw=0., alpha = 0.4)
Plots.plot!(label="Ellipse", xlabel="X", ylabel="Y", zlabel="Z", title="Projected")


## ELLIPSES in rotation space
P = [diagm(ones(3)) zeros(3,9)]
H_r1 = inv(P*inv(H)*P')
surf = generate_ellipse(H_r1, center[1:3])
Plots.scatter3d(surf[1,:], surf[2,:], surf[3,:], label="R1", xlabel="X", ylabel="Y", zlabel="Z", title="R1 Ellipse")
surf = generate_ellipse(diagm(ones(3)), zeros(3))
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="2-Ball")
pr1=Plots.plot!(xlims=[-2,2], ylims=[-2,2], zlims=[-2,2])

P = [zeros(3,3) diagm(ones(3)) zeros(3,6)]
H_r2 = inv(P*inv(H)*P')
surf = generate_ellipse(H_r2, center[1:3])
Plots.scatter3d(surf[1,:], surf[2,:], surf[3,:], label="R2", xlabel="X", ylabel="Y", zlabel="Z", title="R2 Ellipse")
surf = generate_ellipse(diagm(ones(3)), zeros(3))
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="2-Ball")
pr2=Plots.plot!(xlims=[-2,2], ylims=[-2,2], zlims=[-2,2])

P = [zeros(3,6) diagm(ones(3)) zeros(3,3)]
H_r3 = inv(P*inv(H)*P')
surf = generate_ellipse(H_r3, center[1:3])
Plots.scatter3d(surf[1,:], surf[2,:], surf[3,:], label="R3", xlabel="X", ylabel="Y", zlabel="Z", title="R3 Ellipse")
surf = generate_ellipse(diagm(ones(3)), zeros(3))
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="2-Ball")
pr3=Plots.plot!(xlims=[-2,2], ylims=[-2,2], zlims=[-2,2])