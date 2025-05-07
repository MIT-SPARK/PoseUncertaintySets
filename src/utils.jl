## Utils for purse
# Lorenzo Shaikewitz, 4/17/2025

## Quadratic data struct
# x'*H*x + 2c'*x + d
struct Quadratic
    H ::Matrix{Float64}
    c ::Vector{Float64}
    d ::Float64
end

struct QuadraticFunction
    H
    c
    d
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

## Quaternion tools
"""
    Ω1(q)

Defined by `a ⊗ b = Ω1(a)*b` where `a,b` are quaternions.

Automatically adds leading 0 if dimension is 3.

Also satisfies `Ω1(a^{-1}) = Ω1(a)^T`.
"""
function Ω1(q)
    if size(q)[1] == 3
        q = [0; q]
    end
    [q[1] -q[2] -q[3] -q[4];
     q[2]  q[1] -q[4]  q[3];
     q[3]  q[4]  q[1] -q[2];
     q[4] -q[3]  q[2]  q[1]]
end

"""
    Ω2(q)

Defined by `a ⊗ b = Ω2(b)*a` where `a,b` are quaternions.

Automatically adds leading 0 if dimension is 3.

Also satisfies `Ω2(a^{-1}) = Ω2(a)^T`.
"""
function Ω2(q)
    if size(q)[1] == 3
        q = [0; q]
    end
    [q[1] -q[2] -q[3] -q[4];
     q[2]  q[1]  q[4] -q[3];
     q[3] -q[4]  q[1]  q[2];
     q[4]  q[3] -q[2]  q[1]]
end

## Copied from LorenzoRotations
# TODO: just import LorenzoRotations
@noinline my_slow_acos(x) = x ≈ 1 ? zero(x) : x ≈ -1 ? one(x)*π : acos(x)
my_acos(x) = abs(x) <= one(x) ? acos(x) : my_slow_acos(x)

"""
    roterror(R₁, R₂)

Calculate angular difference between `R₁` and `R₂`
"""
function roterror(R₁, R₂)
    R = R₁'*R₂
    θ = my_acos((tr(R) - 1) / 2)
    # ω = [R[3,2] - R[2,3]; R[1,3] - R[3,1]; R[2,1] - R[1,2]] ./ (2*sin(θ))
    return θ*180/π
end

"""
    project2SO3(M)
    
Project the 3x3 matrix `M` to SO(3) via SVD.
"""
function project2SO3(M)
    d = 3
    F = svd(M);
    R = F.U*F.V';
    if (det(R)<0)
        R = F.U * Diagonal([ones(d-1); -1]) * F.V';  
    end
    return R
end

"""
    axang2rotm(ω, θ)
    
Convert an axis `ω` and angle `θ` to a 3x3 rotation matrix.

Normalizes `ω` and returns identity when `θ = 0`.
"""
function axang2rotm(ω, θ)
    normalize!(ω)
    if θ == 0
        return diagm(ones(3))
    end
    K = [0. -ω[3] ω[2]; ω[3] 0. -ω[1]; -ω[2] ω[1] 0.]
    R = I + sin(θ)*K + (1 - cos(θ))*(K*K)
end
