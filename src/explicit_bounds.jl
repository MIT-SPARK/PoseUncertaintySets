## Functions for propagating uncertainty via conformal prediction
# Lorenzo Shaikewitz, 6/11/2025


"""
    angular_bounds_perturbation(center, H; silent=true)

Compute angular bounds given center, ellipse.

We marginalize out positions via projection, 
and assume `cos(θ) > 0`.

This version is wrong! It does not give true bounds,
only the amount that can be perturbed in a given direction.
"""
function angular_bounds_perturbation(center, H; silent=false, order=2)
    Rc = reshape(center[1:9],3,3)
    # can marginalize out positions via projection
    P = [diagm(ones(9)) zeros(9,3)]
    H_r = inv(P*inv(H)*P')

    @polyvar c
    @polyvar s
    vars = [c; s]

    Rx = [1 0 0; 0 c -s; 0 s c]
    Ry = [c 0 s; 0 1 0; -s 0 c]
    Rz = [c s 0; -s c 0; 0 0 1]
    Rws = [Rx, Ry, Rz]

    # solve for each axis
    Δθs = Vector{Any}(undef, 3)
    status_sdp = Vector{MOI.TerminationStatusCode}(undef, 3)
    gaps = -ones(3)
    for (i,Rw) in enumerate(Rws)
        R = Rw*Rc

        # objective: minimize cos(θ)
        obj = c
        
        # constraints
        # expr ≥ 0
        ineq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0)
        push!(ineq, 1 - (vec(R) - vec(Rc))'*H_r*(vec(R) - vec(Rc)))
        
        # c > 0 forces to be within π/2 of center--this is an assumption
        # but if it does not hold these bounds are the wrong approach anyways
        push!(ineq, c)

        eq = [s^2 + c^2 - 1]

        # Solve with TSSOS
        pop = [obj; ineq; eq]
        opt, sol, data, gap = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=silent, solution=true, refine=true)

        if !silent
            println("SDP status: $(data.SDP_status)")
            # println("Loc status: $(refine_status)")
        end

        R_est = project2SO3(reshape([r(vars=>sol) for r in vec(R)],3,3))
        Rc = project2SO3(Rc)

        # save
        Δθs[i] = roterror(R_est, Rc)
        status_sdp[i] = data.SDP_status
        gaps[i] = gap
    end

    return Δθs, status_sdp, gaps
end


"""
    angular_bounds_quat(center, H; silent=true)

Compute a single angular bound with quaternions.

We marginalize out positions via projection, 
and assume `cos(θ) > 0`.
"""
function angular_bounds_quat(center, H; silent=false, order=1)
    qc = center[1:4]
    # marginalize out positions via projection
    P = [diagm(ones(4)) zeros(4,3)]
    H_r = inv(P*inv(H)*P')

    @polyvar q[1:4]

    obj = -(q - qc)'*(q - qc)

    # constraints
    # expr ≥ 0
    ineq = Vector{TSSOS.Poly{Float64}}()
    # expr = 0
    eq = Vector{TSSOS.Poly{Float64}}()

    # bounding ellipse
    push!(ineq, 1 - (q - qc)'*H_r*(q - qc))
    push!(ineq, q'*qc) # enforces within 90 deg

    # SO(3) constraints
    eq = [q'*q - 1]

    # solve
    pop = [obj; ineq; eq]
    order = order
    opt, sol, data, gap = cs_tssos_first(pop, q, order, numeq=length(eq), TS="MD", QUIET=silent, solution=true, refine=true)

    ## Extract solution
    Δθ = roterror(quat2rotm(qc), quat2rotm(normalize(sol)))

    if data.SDP_status != MOI.OPTIMAL
        @warn "[angular_bounds_quat] Returned status $(data.SDP_status). Results may not be lower bound!"
        gap = -1
    end

    return Δθ, data.SDP_status, gap
end

"""
Also include chirality / backproj constraints (returns slow prog in general)
"""
function angular_bounds_quat(center, H, prob; silent=false, order=1)
    qc = center[1:4]
    # marginalize out positions via projection
    P = [diagm(ones(4)) zeros(4,3)]
    H_r = inv(P*inv(H)*P')

    @polyvar q[1:4]
    @polyvar t[1:3]
    vars = [q; t]

    obj = -(q - qc)'*(q - qc)

    # constraints
    # expr ≥ 0
    ineq = Vector{TSSOS.Poly{Float64}}()
    # expr = 0
    eq = Vector{TSSOS.Poly{Float64}}()

    # bounding ellipse
    # push!(ineq, 1 - (q - qc)'*H_r*(q - qc))
    q_front, q_backproj = uncertaintyset_linf_q(prob.y, prob.r, prob.b, prob.camK)
    X = [vars; 1]*[vars; 1]'
    slack = 1e-3
    for (i,q) in enumerate(q_front)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        push!(ineq, -tr(Q*X) - slack)
    end
    for (i,q) in enumerate(q_backproj)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        push!(ineq, -tr(Q*X))
    end

    push!(ineq, q'*qc) # enforces within 90 deg

    # SO(3) constraints
    eq = [q'*q - 1]

    # solve
    pop = [obj; ineq; eq]
    order = order
    opt, sol, data, gap = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=silent, solution=true, refine=true)

    ## Extract solution
    Δθ = roterror(quat2rotm(qc), quat2rotm(normalize(sol)))

    if data.SDP_status != MOI.OPTIMAL
        @warn "[angular_bounds_quat] Returned status $(data.SDP_status). Results may not be lower bound!"
        gap = -1
    end

    return Δθ, data.SDP_status, gap
end


"""
RPY angular bounds.

Current status: solves to optimality with ellipse constraints, need order 4
"""
function angular_bounds_rpy(center, H; silent=false, order=3)
    Rc = reshape(center[1:9],3,3)
    P = [diagm(ones(9)) zeros(9,3)]
    # marginalize out positions via projection
    H_r = inv(P*inv(H)*P')

    @polyvar c[1:3]
    @polyvar s[1:3]
    @polyvar t[1:3]
    vars = [c; s; t]

    Rx = [1 0 0; 0 c[1] -s[1]; 0 s[1] c[1]]
    Ry = [c[2] 0 s[2]; 0 1 0; -s[2] 0 c[2]]
    Rz = [c[3] s[3] 0; -s[3] c[3] 0; 0 0 1]

    # solve for each axis
    Δθs = Vector{Any}(undef, 3)
    status_sdp = Vector{MOI.TerminationStatusCode}(undef, 3)
    gaps = -ones(3)
    for i = 1:3
        R = Rx*Ry*Rz*Rc

        # objective: minimize cos(θ)
        obj = c[i]
        
        # constraints
        # expr ≥ 0
        ineq = Vector{TSSOS.Poly{Float64}}()
        push!(ineq, 1 - (vec(R) - vec(Rc))'*H_r*(vec(R) - vec(Rc)))
        
        # c > 0 forces to be within π/2 of center--this is an assumption
        # but if it does not hold these bounds are the wrong approach anyways
        append!(ineq, c)

        eq = Vector{TSSOS.Poly{Float64}}()
        for i = 1:3
            push!(eq, s[i]^2 + c[i]^2 - 1)
        end

        # Solve with TSSOS
        pop = [obj; ineq; eq]
        opt, sol, data, gap = cs_tssos_first(pop, vars, order, numeq=length(eq), CS="MD", TS="block", QUIET=silent, solution=true, refine=false)

        if !silent
            println("SDP status: $(data.SDP_status)")
            # println("Loc status: $(refine_status)")
        end

        c_val = (vars=>sol) .|> c
        s_val = (vars=>sol) .|> s
        eq_val = (vars=>sol) .|> eq

        (sum(abs.(eq_val) .> 1e-3) == 0) || @warn "At least one equality constraint violated"

        # save
        # Δθs[i] = roterror(R_est, Rc)
        Δθs[i] = atan(s_val[i], c_val[i])*180/π
        status_sdp[i] = data.SDP_status
        gaps[i] = gap

        # Main.@infiltrate
    end

    return Δθs, gaps, status_sdp
end


"""
Version that uses ellipse and backproj/chirality constraints
"""
function angular_bounds_rpy(center, H, prob; silent=false, order=3)
    Rc = reshape(center[1:9],3,3)

    # TEMP: get constraints
    q_front, q_backproj = uncertaintyset_l2(prob.y, prob.r, prob.b, prob.camK)

    @polyvar c[1:3]
    @polyvar s[1:3]
    @polyvar t[1:3]
    vars = [c; s; t]

    Rx = [1 0 0; 0 c[1] -s[1]; 0 s[1] c[1]]
    Ry = [c[2] 0 s[2]; 0 1 0; -s[2] 0 c[2]]
    Rz = [c[3] s[3] 0; -s[3] c[3] 0; 0 0 1]

    # solve for each axis
    Δθs = Vector{Any}(undef, 3)
    status_sdp = Vector{MOI.TerminationStatusCode}(undef, 3)
    gaps = -ones(3)
    for i = 1:3
        R = Rx*Ry*Rz*Rc

        # objective: minimize cos(θ)
        obj = c[i]
        
        # constraints
        # expr ≥ 0
        ineq = Vector{TSSOS.Poly{Float64}}()
        push!(ineq, 1 - ([vec(R); t] - center)'*H*([vec(R); t] - center))
        
        # c > 0 forces to be within π/2 of center--this is an assumption
        # but if it does not hold these bounds are the wrong approach anyways
        append!(ineq, c)

        # extra constraints
        for (i,q) in enumerate(q_backproj)
            push!(ineq, -[vec(R);t;1]'*[q.H  q.c;  q.c'  q.d]*[vec(R);t;1])
        end
        for (i,q) in enumerate(q_front)
            push!(ineq, -[vec(R);t;1]'*[q.H  q.c;  q.c'  q.d]*[vec(R);t;1])
        end

        eq = Vector{TSSOS.Poly{Float64}}()
        for i = 1:3
            push!(eq, s[i]^2 + c[i]^2 - 1)
        end

        # Solve with TSSOS
        pop = [obj; ineq; eq]
        opt, sol, data, gap = cs_tssos_first(pop, vars, order, numeq=length(eq), CS="MD", TS="block", QUIET=silent, solution=true, refine=false)

        if !silent
            println("SDP status: $(data.SDP_status)")
            # println("Loc status: $(refine_status)")
        end

        c_val = (vars=>sol) .|> c
        s_val = (vars=>sol) .|> s
        eq_val = (vars=>sol) .|> eq

        (sum(abs.(eq_val) .> 1e-3) == 0) || @warn "At least one equality constraint violated"

        # save
        # Δθs[i] = roterror(R_est, Rc)
        Δθs[i] = atan(s_val[i], c_val[i])*180/π
        status_sdp[i] = data.SDP_status
        gaps[i] = gap

        # Main.@infiltrate
    end

    return Δθs, gaps, status_sdp
end


"""
Luca's idea: use auxillary variable for rotations
"""
function angular_bounds_rpy2(center, H; silent=false, order=2)
    Rc = reshape(center[1:9],3,3)
    P = [diagm(ones(9)) zeros(9,3)]
    # marginalize out positions via projection
    H_r = inv(P*inv(H)*P')

    @polyvar c[1:3]
    @polyvar s[1:3]
    @polyvar R[1:3,1:3]
    vars = [c; s; vec(R)]

    Rx = [1 0 0; 0 c[1] -s[1]; 0 s[1] c[1]]
    Ry = [c[2] 0 s[2]; 0 1 0; -s[2] 0 c[2]]
    Rz = [c[3] s[3] 0; -s[3] c[3] 0; 0 0 1]

    # solve for each axis
    Δθs = Vector{Any}(undef, 3)
    status_sdp = Vector{MOI.TerminationStatusCode}(undef, 3)
    gaps = -ones(3)
    for i = 1:3
        # objective: minimize cos(θ)
        obj = c[i]
        
        # constraints
        # expr ≥ 0
        ineq = Vector{TSSOS.Poly{Float64}}()
        push!(ineq, 1 - (vec(R) - vec(Rc))'*H_r*(vec(R) - vec(Rc)))
        
        # c > 0 forces to be within π/2 of center--this is an assumption
        # but if it does not hold these bounds are the wrong approach anyways
        append!(ineq, c)

        eq = Vector{TSSOS.Poly{Float64}}()
        for i = 1:3
            push!(eq, s[i]^2 + c[i]^2 - 1)
        end
        append!(eq, vec(Rx'*R) - vec(Ry*Rz*Rc))
        # SO(3) equality constraints
        # orthogonality
        append!(eq, vec(R'*R - I))
        # right hand rule
        append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
        append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
        append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

        # Solve with TSSOS
        pop = [obj; ineq; eq]
        opt, sol, data, gap = cs_tssos_first(pop, vars, order, numeq=length(eq), CS="MD", TS="block", QUIET=silent, solution=true, refine=false)

        if !silent
            println("SDP status: $(data.SDP_status)")
            # println("Loc status: $(refine_status)")
        end

        c_val = (vars=>sol) .|> c
        s_val = (vars=>sol) .|> s
        eq_val = (vars=>sol) .|> eq

        (sum(abs.(eq_val) .> 1e-3) == 0) || @warn "At least one equality constraint violated"

        # save
        # Δθs[i] = roterror(R_est, Rc)
        Δθs[i] = atan(s_val[i], c_val[i])*180/π
        status_sdp[i] = data.SDP_status
        gaps[i] = gap

        # Main.@infiltrate
    end

    return Δθs, gaps, status_sdp
end


"""
RPY angular bounds for quaternion.

Current status: solves with SLOW_PROGRESS for full PURSE constraints.
Solves to optimality with only ellipse constraints, but bounds are all 90 deg.

This is stupidly slow.
"""
function angular_bounds_rpy_quat(center, H, prob; silent=false, order=6)
    qc = center[1:4]
    P = [diagm(ones(4)) zeros(4,3)]
    # marginalize out positions via projection
    H_r = inv(P*inv(H)*P')

    @polyvar c[1:3]
    @polyvar s[1:3]
    vars = [c; s]

    qx = [c[1]; s[1]; 0; 0]
    qy = [c[2]; 0; s[2]; 0]
    qz = [c[3]; 0; 0; s[3]]

    # solve for each axis
    Δθs = Vector{Any}(undef, 3)
    status_sdp = Vector{MOI.TerminationStatusCode}(undef, 3)
    gaps = -ones(3)
    for i = 1:3
        function qmult(r, s)
            q = zeros(TSSOS.Poly{Float64}, 4)
            q[1] = r[1]*s[1] - r[2:4]'*s[2:4]
            q[2] = r[1]*s[2] + r[2]*s[1] - r[3]*s[4] + r[4]*s[3]
            q[3] = r[1]*s[3] + r[2]*s[4] + r[3]*s[1] - r[4]*s[2]
            q[4] = r[1]*s[4] - r[2]*s[3] + r[3]*s[2] + r[4]*s[1]
            return q
        end
        function qrot(r, s)
            # r ⊗ s ⊗ r⁻¹
            q = qmult(s, [r[1]; -r[2:4]])
            q = qmult(r, q)
        end
        q = qrot(qx, qrot(qy, qrot(qz, qc)))

        # objective: minimize cos(θ)
        obj = c[i]
        
        # constraints
        # expr ≥ 0
        ineq = Vector{TSSOS.Poly{Float64}}()
        push!(ineq, 1 - (q - qc)'*H_r*(q - qc))
        
        # c > 0 forces to be within π/2 of center--this is an assumption
        # but if it does not hold these bounds are the wrong approach anyways
        append!(ineq, c)

        # for i = 1:3
        #     push!(ineq, s[i]^2 + c[i]^2 - 1)
        # end

        # extra constraints
        # for (i,q) in enumerate(q_backproj)
        #     push!(ineq, -[vec(R);t;1]'*[q.H  q.c;  q.c'  q.d]*[vec(R);t;1])
        # end
        # for (i,q) in enumerate(q_front)
        #     push!(ineq, -[vec(R);t;1]'*[q.H  q.c;  q.c'  q.d]*[vec(R);t;1])
        # end

        eq = Vector{TSSOS.Poly{Float64}}()
        for i = 1:3
            push!(eq, s[i]^2 + c[i]^2 - 1)
        end

        # Solve with TSSOS
        pop = [obj; ineq; eq]
        opt, sol, data, gap = cs_tssos_first(pop, vars, order, numeq=length(eq), CS="MD", TS="block", QUIET=silent, solution=true, refine=false)

        if !silent
            println("SDP status: $(data.SDP_status)")
            # println("Loc status: $(refine_status)")
        end

        # R_est = project2SO3((vars => sol) .|> R)
        # Rc = project2SO3(Rc)

        c_val = (vars=>sol) .|> c
        s_val = (vars=>sol) .|> s
        eq_val = (vars=>sol) .|> eq

        (sum(abs.(eq_val) .> 1e-3) == 0) || @warn "At least one equality constraint violated"

        # save
        # Δθs[i] = roterror(R_est, Rc)
        Δθs[i] = atan(s_val[i], c_val[i])*180/π
        status_sdp[i] = data.SDP_status
        gaps[i] = gap

        # Main.@infiltrate
    end

    return Δθs, gaps, status_sdp
end


"""
Axis-Angle space ellipse?
"""
function angular_ellipse_axang(center, H; silent=false, order=3)
    r̄ = center[1:9]
    R̄ = reshape(r̄, 3,3)
    P = [diagm(ones(9)) zeros(9,3)]
    H_r = inv(P*inv(H)*P')

    @polyvar s
    @polyvar c
    @polyvar ω[1:3]
    # vars = [vec(R); s; c; ω]
    vars = [s;c;ω]

    # objective
    obj = c

    # constraints
    # expr ≥ 0
    ineq = Vector{TSSOS.Poly{Float64}}()
    # expr = 0
    eq = Vector{TSSOS.Poly{Float64}}()

    # 90 degree rotation constraint
    push!(ineq, c)

    # ellipse constraint
    K = [0  -ω[3]  ω[2];
         ω[3]  0  -ω[1];
         -ω[2]  ω[1]  0]
    push!(ineq, 1 - (vec((I + s*K + (1 - c)*K*K)*R̄) - r̄)'*H_r*(vec((I + s*K + (1 - c)*K*K)*R̄) - r̄))

    # R ∈ SO(3)
    push!(eq, s^2 + c^2 - 1)
    push!(eq, ω[1]^2 + ω[2]^2 + ω[3]^2 - 1)

    # solve
    pop = [obj; ineq; eq]
    order = order
    opt, sol, data, gap = cs_tssos_first(pop, vars, order, numeq=length(eq), TS=false, CS="MD", QUIET=silent, solution=true, refine=false)

    # Main.@infiltrate

    # warn if not actually opt
    # if data.SDP_status != MOI.OPTIMAL

    # extract angle
    angle = acos(opt)
    # angle is radius, axis doesn't matter for this particular problem.
end



"""
Uncertainty bound from "Object Pose Estimation with Statistical Guarantees"

Maximize distance to PURSE while remaining in PURSE.
"""
function purse_bounds(center, y, r, b, camK; p=2, order=2, silent=false)
    (p == 2 || p == Inf) || error("only accepts `p=2` or `p=Inf`")

    if p == 2
        q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
        q_eqs = SO3_constraints()
    else
        # does not work
        # q_front, q_backproj = uncertaintyset_linf_R(y, r, b, camK)
        # q_eqs = SO3_constraints()
        q_front, q_backproj = uncertaintyset_linf_q(y, r, b, camK)
        q_eqs = q_constraints()
    end

    return purse_bounds(center, q_front, q_backproj, q_eqs; order=order, silent=silent)
end

function purse_bounds(center, q_front, q_backproj, q_eqs; order=2, silent=false)
    dim = length(center)
    if dim == 12
        Rc = reshape(center[1:9],3,3)
        @polyvar R[1:3,1:3]
    else
        Rc = center[1:4]
        @polyvar R[1:4] 
    end

    tc = center[end-2:end] 
    @polyvar t[1:3]
    vars = [vec(R); t]

    ang_bound = 0.
    ang_gap = 1e6
    trans_bound = 0.
    trans_gap = 1e6

    status = Array{MOI.TerminationStatusCode}(undef, 2)

    for λ = [0, 1]
        # objective
        if dim == 12
            obj = -( λ*tr((R-Rc)'*(R-Rc)) + (1-λ)*(t-tc)'*(t-tc) )
        else
            obj = -( λ*((R-Rc)'*(R-Rc)) + (1-λ)*(t-tc)'*(t-tc) )
        end

        # constraints
        # expr ≥ 0
        ineq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 
        # expr = 0
        eq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 

        # PURSE constraints
        X = [vars; 1]*[vars; 1]'
        slack = 1e-3
        for (i,q) in enumerate(q_front)
            Q = Symmetric([q.H  q.c;  q.c'  q.d])
            push!(ineq, -tr(Q*X) - slack)
        end
        for (i,q) in enumerate(q_backproj)
            Q = Symmetric([q.H  q.c;  q.c'  q.d])
            push!(ineq, -tr(Q*X))
        end
        for (i,q) in enumerate(q_eqs)
            Q = Symmetric([q.H  q.c;  q.c'  q.d])
            push!(eq, tr(Q*X))
        end

        # 90 degree rotation constraint
        if dim == 7
            push!(ineq, R'*center[1:4])
            @warn "angular bound conversion to degrees not yet implemented"
        end

        # solve
        pop = [obj; ineq; eq]
        order = order # supplementary material: they use second order
        opt, sol, data, gap = cs_tssos_first(pop, vars, order, numeq=length(eq), TS=false, CS=false, QUIET=silent, solution=true, refine=false)

        # if isdefined(Main, :Infiltrator) Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__) end # 🚨 INFILTRATOR 🚨

        if data.SDP_status != MOI.OPTIMAL
            @warn "[purse_bounds] λ=$λ returned status $(data.SDP_status). Results may not be lower bound!"
            gap = -1
        end
        status[λ+1] = data.SDP_status

        if λ == 1
            # rotation case
            # |R₁ - R₂|^2_F = |R₁|^2_F + |R₂|^2_F - 2⟨R₁, R₂⟩
            # ⟨R₁, R₂⟩ = (6 - |R₁ - R₂|^2_F) / 2
            frob_norm = -opt
            if abs(1 - frob_norm/4) > 1
                ang_bound = 180.
            else
                ang_bound = SimpleRotations.robust_acos(1 - frob_norm/4)*180/π
            end
            ang_gap = gap
        else
            # translation case
            # opt = \|t - tc\|^2_2
            trans_bound = sqrt(abs(-opt))
            trans_gap = gap
        end
    end

    return trans_bound, trans_gap, ang_bound, ang_gap, status
end


"""
Refine as bounding box aligned with axes of `H_t`.
Options:
- SLOW: backproj + chilrality (1)
- FASTER: ellipse + backproj + chilrality (2)
    - no loss of tightness compared to (1)
- BEST: ellipse + chirality (3)
- FASTEST: ellipse only (4)
"""
function refine_bbox(center, H, y, r, b, camK; p=2, R=true, mode=3, order=1, H_t=nothing, silent=false)
    (p == 2 || p == Inf) || error("only accepts `p=2` or `p=Inf`")

    if p == 2
        q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
        q_eqs = SO3_constraints()
        R=true
    else
        if R
            q_front, q_backproj = uncertaintyset_linf_R(y, r, b, camK)
            q_eqs = SO3_constraints()
        else
            q_front, q_backproj = uncertaintyset_linf_q(y, r, b, camK)
            q_eqs = q_constraints()
        end
    end

    if isnothing(H_t)
        # marginalize via projection
        P = [zeros(3,R ? 9 : 4) diagm(ones(3))]
        H_t = inv(P*pinv(H)*P')
    end
    return refine_bbox(center, H, q_front, q_backproj, q_eqs; p=p, mode=mode, order=order, H_t=H_t, silent=silent)
end

function refine_bbox(center, H, prob; R=true, mode=3, order=1, H_t=nothing, silent=false)
    refine_bbox(center, H, prob.y, prob.r, prob.b, prob.camK; p=prob.p, R=R, mode=mode, order=order, H_t=H_t, silent=silent)
end

function refine_bbox(center, H, q_front, q_backproj, q_eqs; p=2, mode=3, order=1, H_t=nothing, silent=false)
    dim = length(center)
    if dim == 12
        @polyvar R[1:3,1:3]
    else
        @polyvar R[1:4] # quaternion
        # @polyvar R[1:3,1:3]
    end
    @polyvar t[1:3]
    vars = [vec(R); t]

    # save bounds for each axis
    bounds = zeros(2,3)
    gaps = zeros(2,3)
    statuses = Array{MOI.TerminationStatusCode}(undef, 2,3)

    # eigendecomposition
    V = eigvecs(H_t)
    for idx = [1, 2, 3]
        # objective: axis-aligned bbox
        obj = (V'*(t - center[end-2:end]))[idx]

        # constraints
        # expr ≥ 0
        ineq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 
        # expr = 0
        eq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 

        if mode == 1
            # pose uncertainty set constraints
            for (i,q) in enumerate(q_backproj)
                push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
            end
            for (i,q) in enumerate(q_front)
                push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
            end
        elseif mode == 2
            # sphere constraints + pose uncertainty set constraints
            push!(ineq, -( (vars-center)'*H*(vars-center) - 1 ))
            for (i,q) in enumerate(q_backproj)
                push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
            end
            for (i,q) in enumerate(q_front)
                push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
            end
        elseif mode == 3
            # pose uncertainty set constraints
            push!(ineq, -( (vars-center)'*H*(vars-center) - 1 ))
            # chirality
            for (i,q) in enumerate(q_front)
                push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
            end
        elseif mode == 4
            # only pose uncertainty set constraints
            push!(ineq, -( (vars-center)'*H*(vars-center) - 1 ))
        end

        # Always enforce SO(3) constraints
        for (i,q) in enumerate(q_eqs)
            push!(eq, [vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
        end

        # solve
        for (row, mult) in enumerate([-1, 1])
            obj_cur = obj*mult
            pop = [obj_cur; ineq; eq]
            order = order
            opt, sol, data, gap = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", CS="MF", QUIET=silent, solution=true, refine=false)

            if data.SDP_status != MOI.OPTIMAL
                @warn "[refine_bbox] Status $(data.SDP_status) along axis $idx ($mult)"
            end

            bounds[row,idx] = mult*opt
            gaps[row,idx] = gap
            statuses[row,idx] = data.SDP_status
        end

        # if isdefined(Main, :Infiltrator) Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__) end # 🚨 INFILTRATOR 🚨
    end

    return bounds, gaps, statuses
end