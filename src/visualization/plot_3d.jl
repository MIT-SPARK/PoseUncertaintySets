## Functions to plot ellipses
# Lorenzo Shaikewitz, 6/11/2025

"""
    ellipse_to_surf(A, c, grid_pts=100)

Generate a surface (list of points) from an ellipse matrix `A` centered at `c`.
"""
function ellipse_to_surf(A, c, grid_pts=100)
    # Eigenvalue decomposition of A
    λ, V = eigen(Symmetric(A))
        
    # Generate points on unit sphere
    u = LinRange(0, 2π, grid_pts)
    v = LinRange(0, π, grid_pts)
    x = zeros(grid_pts, grid_pts)
    y = zeros(grid_pts, grid_pts)
    z = zeros(grid_pts, grid_pts)
    for (i,ui) in enumerate(u)
        # Parametric equations of the unit sphere
        x[:,i] = cos(ui)*sin.(v)
        y[:,i] = sin(ui)*sin.(v)
        z[:,i] = cos.(v)
    end
    sphere = [vec(x) vec(y) vec(z)]' # 3 x grid_pts^2

    # Transform sphere to ellipsoid
    axes_lengths = 1. ./ sqrt.(λ)
    surf = V * diagm(axes_lengths) * sphere
    surf .+= c
    return surf
end


"""
Plot bounding box aligned with `H_t`
"""
function plot_bbox!(plt,center, H_t, bounds; kwargs...)
    V = eigvecs(H_t)
    xyz = V'*center[end-2:end] .+ bounds'
    bbox = [[xyz[1,1]*ones(4); xyz[1,2]*ones(4)] repeat([xyz[2,1]; xyz[2,1]; xyz[2,2]; xyz[2,2]],2) repeat([xyz[3,1]; xyz[3,2]; xyz[3,1]; xyz[3,2]],2)]
    bbox = V*bbox'
    return Plots.scatter!(plt,bbox[1,:], bbox[2,:], bbox[3,:]; kwargs...)
end


"""
Plot a CDF
"""
function plot_cdf!(plt, data; kwargs...)
    Plots.plot!(plt, sort(data), (1:length(data))/length(data); kwargs...)
end