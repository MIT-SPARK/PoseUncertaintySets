## Generate Pose Uncertainty Set
# Lorenzo Shaikewitz, 4/29/2025

# Quadratic data struct
# x'*H*x + 2c'*x + d
struct Quadratic
    H ::Matrix{Float64}
    c ::Vector{Float64}
    d ::Float64
end

"""
computes x'*Q*x
"""
function quadform(q::Quadratic, x)
    [x;1]'*[q.H  q.c;  q.c'  q.d]*[x;1]
end

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
    [q_front, q_backproj] = uncertaintyset_linf_R(y, r, b, K)

Generate pose uncertainty set (linf norm) from problem data.
These are exclusively linear constraints when expressed in the variable `R`.

# Arguments
- `y`: pixel keypoints [3 x N] (homogenized)
- `r`: l2 radii for each keypoint [N]
- `b`: 3D canonical keypoint frame, meters [3 x N]
- `K`: camera calibration matrix [3 x 3]

# Returns
- `q_front`: list of chirality constraints ≤ 0 [N]
- `q_backproj`: list of backprojection constraints ≤ 0 [4*N]
"""
function uncertaintyset_linf_R(y, r, b, K)
    N = size(y,2)
    q_front = Vector{Quadratic}(undef, 0)
    q_backproj = Vector{Quadratic}(undef, 0)

    e3 = [0;0;1]
    for i = 1:N
        iy3 = (I - y[:,i]*e3')
        kr_bK = kron(b[:,i]', K)

        # front of camera
        H = zeros(12,12)
        c = -[e3'*kr_bK e3'*K]'
        s = 0.
        push!(q_front, Quadratic(H, c[:,1], s))
        
        # backprojection (linf norm)
        for j = 1:2
            ej = zeros(3); ej[j] = 1

            H = zeros(12,12)
            c = zeros(12)
            c[1:9] = ej'*iy3*kr_bK - r[i]*e3'*kr_bK
            c[10:12] = ej'*iy3*K   - r[i]*e3'*K
            s = 0.
            push!(q_backproj, Quadratic(H, c, s))

            # negative term
            H = zeros(12,12)
            c = zeros(12)
            c[1:9] = -ej'*iy3*kr_bK - r[i]*e3'*kr_bK
            c[10:12] = -ej'*iy3*K   - r[i]*e3'*K
            s = 0.
            push!(q_backproj, Quadratic(H, c, s))
        end
    end

    return q_front, q_backproj
end


## SO(3) constraints in quadratic form
function SO3_constraints()
    q_eqs = Vector{Quadratic}(undef, 0)
    # r'*r = 1
    for i = 0:2
        H = zeros(12,12)
        H[3*i+1,3*i+1] = 1.; H[3*i+2,3*i+2] = 1.; H[3*i+3,3*i+3] = 1.
        q = Quadratic(H, zeros(12), -1.)
        push!(q_eqs, q)
    end
    # r1'*r2 = r1'*r3 = r2'*r3 = 0
    begin
        # r1'*r2
        H = zeros(12,12)
        H[1,4] = 0.5; H[2,5] = 0.5; H[3,6] = 0.5
        H += H'
        q = Quadratic(H, zeros(12), 0.)
        push!(q_eqs, q)
        # r1'*r3
        H = zeros(12,12)
        H[1,7] = 0.5; H[2,8] = 0.5; H[3,9] = 0.5
        H += H'
        q = Quadratic(H, zeros(12), 0.)
        push!(q_eqs, q)
        # r2'*r3
        H = zeros(12,12)
        H[4,7] = 0.5; H[5,8] = 0.5; H[6,9] = 0.5
        H += H'
        q = Quadratic(H, zeros(12), 0.)
        push!(q_eqs, q)
    end
    # cross products
    begin
        # r1 x r2 - r3
        H = zeros(12,12)
        H[2,6] = 1; H[3,5] = -1; H += H'
        c = zeros(12); c[7] = -1.
        q = Quadratic(H, c, 0.)
        push!(q_eqs, q)
        H = zeros(12,12)
        H[3,4] = 1; H[1,6] = -1; H += H'
        c = zeros(12); c[8] = -1.
        q = Quadratic(H, c, 0.)
        push!(q_eqs, q)
        H = zeros(12,12)
        H[1,5] = 1; H[2,4] = -1; H += H'
        c = zeros(12); c[9] = -1.
        q = Quadratic(H, c, 0.)
        push!(q_eqs, q)

        # r2 x r3 - r1
        H = zeros(12,12)
        H[5,9] = 1; H[6,8] = -1; H += H'
        c = zeros(12); c[1] = -1.
        q = Quadratic(H, c, 0.)
        push!(q_eqs, q)
        H = zeros(12,12)
        H[6,7] = 1; H[4,9] = -1; H += H'
        c = zeros(12); c[2] = -1.
        q = Quadratic(H, c, 0.)
        push!(q_eqs, q)
        H = zeros(12,12)
        H[4,8] = 1; H[5,7] = -1; H += H'
        c = zeros(12); c[3] = -1.
        q = Quadratic(H, c, 0.)
        push!(q_eqs, q)

        # r3 x r1 - r2
        H = zeros(12,12)
        H[8,3] = 1; H[9,2] = -1; H += H'
        c = zeros(12); c[4] = -1.
        q = Quadratic(H, c, 0.)
        push!(q_eqs, q)
        H = zeros(12,12)
        H[9,1] = 1; H[7,3] = -1; H += H'
        c = zeros(12); c[5] = -1.
        q = Quadratic(H, c, 0.)
        push!(q_eqs, q)
        H = zeros(12,12)
        H[7,2] = 1; H[8,1] = -1; H += H'
        c = zeros(12); c[6] = -1.
        q = Quadratic(H, c, 0.)
        push!(q_eqs, q)
    end
    return q_eqs
end


## QUATERNIONS

"""
    [q_front, q_backproj] = uncertaintyset_linf_q(y, r, b, K)

Generate pose uncertainty set (linf norm) from problem data.
Expressed in the quaternion form `q`.

# Arguments
- `y`: pixel keypoints [3 x N] (homogenized)
- `r`: l2 radii for each keypoint [N]
- `b`: 3D canonical keypoint frame, meters [3 x N]
- `K`: camera calibration matrix [3 x 3]

# Returns
- `q_front`: list of chirality constraints ≤ 0 [N]
- `q_backproj`: list of backprojection constraints ≤ 0 [4*N]
"""
function uncertaintyset_linf_q(y, r, b, K)
    N = size(y,2)
    q_front = Vector{Quadratic}(undef, 0)
    q_backproj = Vector{Quadratic}(undef, 0)

    e3 = [0;0;1]
    for i = 1:N
        # front of camera
        H = zeros(7,7)
        H[1:4, 1:4] = -(-Ω1(e3)*Ω2(b[:,i]))
        H += H'
        c = -[zeros(4); e3]
        s = 0.
        push!(q_front, Quadratic(H, c, s))
        
        # backprojection (linf norm)
        iy3 = (I - y[:,i]*e3')
        for j = 1:2
            ej = zeros(3); ej[j] = 1

            H = zeros(7,7)
            H[1:4, 1:4] = (-Ω1((iy3*K)'*ej)*Ω2(b[:,i])) - r[i]*(-Ω1(e3)*Ω2(b[:,i]))
            H += H'
            c = [zeros(4); (ej'*iy3*K)' - r[i]*e3]
            s = 0.
            push!(q_backproj, Quadratic(H, c, s))

            # negative term
            H = zeros(7,7)
            H[1:4, 1:4] = -(-Ω1((iy3*K)'*ej)*Ω2(b[:,i])) - r[i]*(-Ω1(e3)*Ω2(b[:,i]))
            H += H'
            c = [zeros(4); -(ej'*iy3*K)' - r[i]*e3]
            s = 0.
            push!(q_backproj, Quadratic(H, c, s))
        end
    end

    ## remove sign ambiguity
    # H = zeros(7,7)
    # c = [1; zeros(6)]
    # s = 0.
    # # q₁ ≤ 0
    # push!(q_front, Quadratic(H, c, s))

    return q_front, q_backproj
end

## SO(3) constraints in quadratic form
function q_constraints()
    q_eqs = Vector{Quadratic}(undef, 0)
    # q'*q = 1
    H = zeros(7,7)
    H[1:4,1:4] = diagm(ones(4)) 
    # q₁² + q₂² + q₃² + q₄² - 1 = 0
    q = Quadratic(H, zeros(7), -1.)
    push!(q_eqs, q)
    return q_eqs
end


"""
Sample from the pose uncertainty set.

Offer 2 methods:
1. RANSAG (fails when set is very large)
2. Griding + max margin (more robust but slower)
"""
function sample_set(prob; method="ransag", Ht=nothing, T=1000)
    y = prob.y
    r = prob.r
    b = prob.b
    camK = prob.camK
    N = size(r,1)

    # set of feasible poses found
    S_R = []
    S_t = []
    if method == "ransag"
        # search for T iterations
        for _ = 1:T
            idxs = sample(1:N, 3, replace=false)

            # perturb from center by (uniformly) random magnitude in random direction
            r_selected = r[idxs] .* rand(3) # random magnitude
            dir = normalize.(eachcol(randn(2,3))) # random direction
            kpts = y[1:2,idxs] + reduce(hcat, r_selected .* dir)
            kpts = [kpts; ones(1,3)]

            # run p3p
            R_p3p, t_p3p = p3p(kpts, b[:,idxs], camK)

            # save all which are in PURSE
            for i = axes(t_p3p,2)
                R = R_p3p[:,:,i]
                t = t_p3p[:,i]
                y_p3p = camK*(R*b .+ t)
                y_p3p = reduce(hcat, eachcol(y_p3p) ./ y_p3p[3,:])

                # just check backproj
                if sum(norm.(eachcol(y_p3p - y), prob.p) .<= r) == N
                    push!(S_R, R)
                    push!(S_t, t)
                end
            end
        end
    elseif method == "grid"
        # convert ellipse into bbox
        # convert bbox into grid
        # also check outside ellipse (and warn if failure!)
        error("Not implemented yet")
    end
    return S_R, S_t
end