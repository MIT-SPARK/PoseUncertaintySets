## Generate Pose Uncertainty Set
# Lorenzo Shaikewitz, 4/29/2025

using LinearAlgebra

# TODO: move to module
include("utils.jl")

"""
    [q_front, q_backproj] = uncertaintyset_l2(y, r, b, K)

Generate pose uncertainty set (l2 norm) from problem data.

# Arguments
- `y`: pixel keypoints [3 x N] (homogenized)
- `r`: l2 radii for each keypoint [N]
- `b`: 3D canonical keypoint frame, meters [3 x N]
- `K`: camera calibration matrix [3 x 3]

# Returns
- `q_front`: list of chirality constraints ≤ 0 [N]
- `q_backproj`: list of backprojection constraints ≤ 0 [N]
"""
function uncertaintyset_l2(y, r, b, K)
    N = size(y,2)
    q_front = []
    q_backproj = []

    for i = 1:N
        iy3 = (I - y[:,i]*[0 0 1])
        kr_bK = kron(b[:,i]', K)

        # front of camera
        H = zeros(12,12)
        c = -[[0 0 1.]*kr_bK [0 0 1.]*K]'
        s = 0.
        push!(q_front, Quadratic(H, c[:,1], s))
        
        # backprojection (l2 norm)
        H = zeros(12,12)
        H[1:9,1:9] = (iy3*kr_bK)'*(iy3*kr_bK) - r[i]^2*([0 0 1.]*kr_bK)'*([0 0 1.]*kr_bK) # verified
        H[10:12,10:12] = (iy3*K)'*(iy3*K) - r[i]^2*([0 0 1.]*K)'*([0 0 1.]*K) # verified
        H[10:12,1:9] = (iy3*K)'*(iy3*kr_bK) - r[i]^2*([0 0 1.]*K)'*([0 0 1.]*kr_bK) # verified
        H[1:9,10:12] = H[10:12,1:9]'
        H ./= r[i]^2
        c = zeros(12)
        s = 0.
        push!(q_backproj, Quadratic(H, c, s))
    end

    return q_front, q_backproj
end


"""
    [q_front, q_backproj] = uncertaintyset_linf(y, r, b, K)

Generate pose uncertainty set (linf norm) from problem data.

# Arguments
- `y`: pixel keypoints [3 x N] (homogenized)
- `r`: linf radii for each keypoint [N]
- `b`: 3D canonical keypoint frame, meters [3 x N]
- `K`: camera calibration matrix [3 x 3]

# Returns
- `q_front`: list of chirality constraints ≤ 0 [N]
- `q_backproj`: list of backprojection constraints ≤ 0 [4*N]
"""
function uncertaintyset_linf(y, r, b, K)
    N = size(y,2)
    q_front = []
    q_backproj = []

    for i = 1:N
        iy3 = (I - y[:,i]*[0 0 1])
        kr_bK = kron(b[:,i]', K)

        # front of camera
        H = zeros(12,12)
        c = -[[0 0 1.]*kr_bK [0 0 1.]*K]'
        s = 0.
        push!(q_front, Quadratic(H, c[:,1], s))
        
        # backprojection (linf norm)
        # unlike l2 there are 4*N constraints!
        for j = 1:2 # x and y
            # both sides of absolute value
            c = zeros(12)
            c[1:9]   = (iy3*kr_bK)[j,:] - r[i]*([0 0 1.]*kr_bK)[1,:]
            c[10:12] = (iy3*K)[j,:] - r[i]*([0 0 1.]*K)[1,:]
            c ./= r[i]
            push!(q_backproj, Quadratic(zeros(12,12), c, 0.))

            c = zeros(12)
            c[1:9]   = -(iy3*kr_bK)[j,:] - r[i]*([0 0 1.]*kr_bK)[1,:]
            c[10:12] = -(iy3*K)[j,:]     - r[i]*([0 0 1.]*K)[1,:]
            c ./= r[i]
            push!(q_backproj, Quadratic(zeros(12,12), c, 0.))
        end
    end

    return q_front, q_backproj
end