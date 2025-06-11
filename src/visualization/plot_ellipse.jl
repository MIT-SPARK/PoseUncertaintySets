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