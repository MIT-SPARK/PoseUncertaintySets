## Generate Pose Uncertainty Set
# Lorenzo Shaikewitz, 4/29/2025

using LinearAlgebra
using DynamicPolynomials

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
function uncertaintyset_l2(y, r, b, K; func=false)
    N = size(y,2)
    q_front = Vector{Quadratic}(undef, 0)
    q_backproj = Vector{Quadratic}(undef, 0)

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

# TODO: redundant constraints for S-Lemma


"""
    check_feas(q_front, q_backproj, q_eqs, vars_proj; tol=1e-3, silent=true)

Check feasibility of pose given margins with tolerance `tol`.

`vars_proj` should be `[margins; vec(R); t]`
"""
function check_feas(q_front, q_backproj, q_eqs, vars_proj; tol=1e-3, silent=true)
    margin = vars_proj[1:end-3-9]
    
    # check for linf
    N = length(q_front)
    if length(q_backproj) > N
        margin = repeat(margin, inner=4)
    end

    feasible = true
    x_test = [vars_proj[end-3-9+1:end]; 1]
    X_test = x_test*x_test'
    for (i,q) in enumerate(q_front)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        if !(tr(Q*X_test) <= tol)
            if !silent
                printstyled("FoC Ineq. $i fails: ",color=:red)
                print(tr(Q*X_test))
                println(" > 0")
            end
            feasible = false
        end
    end
    for (i,q) in enumerate(q_backproj)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        if !(tr(Q*X_test) <= -margin[i] + tol)
            if !silent
                printstyled("BP Ineq. $i fails: ",color=:red)
                print(tr(Q*X_test) - -margin[i])
                println(" > 0")
            end
            feasible = false
        end
    end
    for (i,q) in enumerate(q_eqs)
        Q = [q.H  q.c;  q.c'  q.d]
        if !(abs(tr(Q*X_test)) <= tol)
            if !silent
                printstyled("Eq. $i fails: ",color=:red)
                print(tr(Q*X_test))
                println(" != 0")
            end
            feasible = false
        end
    end
    return feasible
end