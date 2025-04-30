## Check the ellipse and its marginalizations against samples
# run sample first
# Lorenzo Shaikewitz, 4/30/2025

# check against the samples
cover_both = zeros(Bool, size(poses,1))
cover_t = zeros(Bool, size(poses,1))
cover_t_const_r = zeros(Bool, size(poses,1))
# cover_r = zeros(Bool, size(poses,1))
ang_deviation = zeros(size(poses,1))
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

    # update angular deviation
    Rc = reshape(center[1:9],3,3)
    ang_deviation[i] = roterror(R,Rc)
end

printstyled("Coverage of samples\n", underline=true)
@printf "full ellipse: %d/%d (%.2f%%)\n" sum(cover_both) size(poses,1) (sum(cover_both)/size(poses,1)*100)
@printf "t marginal: %d/%d (%.2f%%)\n" sum(cover_both) size(poses,1) (sum(cover_both)/size(poses,1)*100)
@printf "constant R: %d/%d (%.2f%%)\n" sum(cover_both) size(poses,1) (sum(cover_both)/size(poses,1)*100)

@printf "Angular deviation: %.2f deg\n" maximum(ang_deviation)

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

Plots.plot([center[10]],[center[11]],[center[12]], seriestype=:scatter, label="center")
surf = generate_ellipse(H[10:12,10:12], center[10:12])
Plots.scatter!([0],[0],[0],label="camera")
Plots.plot!(eachrow(good_centers)...,seriestype=:scatter,zcolor=good_margins, color=:blues, label="samples")
p1 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="Ellipse", xlabel="X", ylabel="Y", zlabel="Z", title="Constant Rotation")

Plots.plot([center[10]],[center[11]],[center[12]], seriestype=:scatter, label="center")
surf = generate_ellipse(H_t, center[10:12])
Plots.scatter!([0],[0],[0],label="camera")
Plots.plot!(eachrow(good_centers)...,seriestype=:scatter,zcolor=good_margins, color=:blues, label="samples")
p2 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="Ellipse", xlabel="X", ylabel="Y", zlabel="Z", title="Projected")