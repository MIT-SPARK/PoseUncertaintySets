## Functions for propagating uncertainty via conformal prediction
# Lorenzo Shaikewitz, 6/11/2025


"""
    bounding_ellipse(center, y, r, b, K[; p=2, solver=Clarabel.Optimizer, silent=false])

S-Lemma to outer bound pose uncertainty set. `(x-center)'*H*(x-center) ≤ 1`

Returns ellipse matrix `H`, optimization status.

# Arguments
- `center`: center of ellipse [vec(R), t]
- `y`: pixel keypoints [3 x N] (homogenized)
- `r`: l2 radii for each keypoint [N]
- `b`: 3D canonical keypoint frame, meters [3 x N]
- `camK`: camera calibration matrix [3 x 3]
## Optional Arguments
- `p=2`: what calibration norm to use (`p=Inf` is an option but does not work)
- `solver=Clarabel.Optimizer`: what solver to use
- `silent=false`: should we print things?

# Returns
- `H`: PSD ellipse matrix
- `status`: termination status of optimization
"""
function bounding_ellipse(center, y, r, b, camK; p=2, solver=Mosek.Optimizer, silent=false)
    (p == 2 || p == Inf) || error("only accepts `p=2` or `p=Inf`")

    if p == 2
        q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
        q_eqs = SO3_constraints()
    else
        ## Rotation Version
        q_front, q_backproj = uncertaintyset_linf_R(y, r, b, camK)
        q_new = []
        for q1 in q_backproj, q2 in q_backproj
            H = -q1.c*q2.c'
            H += H'
            push!(q_new, Quadratic(H, zeros(12), 0.)) # ≤ 0
        end
        q_front = [q_front; q_new]
        q_eqs = SO3_constraints()

        ## Quaternion Version
        # q_front, q_backproj = uncertaintyset_linf_q(y, r, b, camK)
        # q_eqs = q_constraints()
    end

    return bounding_ellipse(center, q_front, q_backproj, q_eqs; solver=solver, silent=silent)
end


function bounding_ellipse(center, prob; solver=Mosek.Optimizer, silent=false)
    return bounding_ellipse(center, prob.y, prob.r, prob.b, prob.camK; p=prob.p, solver=solver, silent=silent)
end


function bounding_ellipse(center, q_front, q_backproj, q_eqs; solver=Mosek.Optimizer, silent=false)

    # JuMP model
    model = Model(solver)
    if silent
        set_silent(model)
    end
    @variable(model, log_det_H0)
    @variable(model, λ[1:length(q_front) + length(q_backproj)] .>= 0)
    @variable(model, η[1:length(q_eqs)])

    # full 12 x 12
    dim = size(q_front[1].H,1)
    @variable(model, H0[1:dim,1:dim] ∈ PSDCone())
    # @variable(model, h0 ≥ 0)
    # H0 = diagm(ones(dim))*h0
    # H0 = [zeros(9,12); zeros(3,9) diagm(ones(3))*h0]

    # rotation and position independent
    # @variable(model, H0_r[1:9,1:9] ∈ PSDCone())
    # @variable(model, H0_p[1:3,1:3] ∈ PSDCone())
    # H0 = [H0_r zeros(9,3); zeros(3,9) H0_p]
    # H0 = [zeros(9,9) zeros(9,3); zeros(3,9) H0_p]
    # H0 = [zeros(4,4) zeros(4,3); zeros(3,4) H0_p]
    # H0 = [H0_r zeros(9,3); zeros(3,9) zeros(3,3)]

    # diagonal
    # @variable(model, r_H0[1:12] .>= 0)
    # H0 = diagm(r_H0)

    # objective: max logdet(H0)
    # use auxillary variable
    @objective(model, Max, log_det_H0)
    @constraint(model, [log_det_H0; 1; triangle_vec(H0)] in MOI.LogDetConeTriangle(dim))
    # @constraint(model, [log_det_H0; triangle_vec(H0)] in MOI.RootDetConeTriangle(dim)) # alt logdet cone
    # @constraint(model, H0[end-2:end,end-2:end] - log_det_H0*diagm(ones(3)) >= 0, PSDCone()) # max minimum eigenvalue

    # alt objective
    # @objective(model, Max, tr(H0))

    # build and constrain M
    # q0 = x'*H0*x + 2(-H0*c)'*x + c'*H0*c <= 1
    # M = -[H0  -H0*center;  (-H0*center)'  center'*H0*center-1]
    # for (i_bp,q) in enumerate(q_backproj)
    #     i = i_bp
    #     M += [λ[i]*q.H  λ[i]*q.c;  λ[i]*q.c'  λ[i]*q.d]
    # end
    # for (i_fc,q) in enumerate(q_front)
    #     i = i_fc + length(q_backproj)
    #     M += [λ[i]*q.H  λ[i]*q.c;  λ[i]*q.c'  λ[i]*q.d]
    # end
    # for (i,q) in enumerate(q_eqs)
    #     M += [η[i]*q.H  η[i]*q.c;  η[i]*q.c'  η[i]*q.d]
    # end
    M = -[center'*H0*center-1  (-H0*center)'; -H0*center  H0]
    for (i_bp,q) in enumerate(q_backproj)
        i = i_bp
        M += [λ[i]*q.d  λ[i]*q.c';  λ[i]*q.c  λ[i]*q.H]
    end
    for (i_fc,q) in enumerate(q_front)
        i = i_fc + length(q_backproj)
        M += [λ[i]*q.d  λ[i]*q.c';  λ[i]*q.c  λ[i]*q.H]
    end
    for (i,q) in enumerate(q_eqs)
        M += [η[i]*q.d  η[i]*q.c';  η[i]*q.c  η[i]*q.H]
    end
    @constraint(model, triangle_vec(M) ∈ MOI.PositiveSemidefiniteConeTriangle(dim+1))
    # @constraint(model, M >= 0, PSDCone())

    # Solve with JuMP
    optimize!(model)

    if !silent && !is_solved_and_feasible(model)
        # not really a warning
        @warn "[bounding_ellipse] Solver terminated with status $(termination_status(model))"
    end
    H0_val = value.(H0)

    return (H0_val, termination_status(model))
end


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
Solving the bounding sphere problem with a direct relaxation.

Returns radius of sphere and SDP status.

Why solve for a joint bounding sphere? The RANSAG approach makes much more sense.
"""
function bounding_sphere(center, y, r, b, camK; p=2, order=1, silent=false)
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

    return bounding_sphere(center, q_front, q_backproj, q_eqs; order=order, silent=silent)
end

function bounding_sphere(center, prob; order=1, silent=false)
    return bounding_sphere(center, prob.y, prob.r, prob.b, prob.camK; p=prob.p, order=order, silent=silent)
end


function bounding_sphere(center, q_front, q_backproj, q_eqs; order=1, silent=false)
    dim = length(center)
    if dim == 12
        @polyvar R[1:3,1:3]
    else
        @polyvar R[1:4] # quaternion
    end
    @polyvar t[1:3]
    vars = [vec(R); t]

    # objective
    W = [I -center; -center' center'*center]
    obj = -[vars;1]'*W*[vars;1] # equivalent to minimizing radius of ellipse centered at `center`

    # minimize individual ones
    # tc = center[end-2:end]
    # obj = -(t - tc)'*(t - tc)
    # qc = center[1:4]
    # obj = -(R - qc)'*(R - qc)
    # this is identical to PURSE bounds!

    # constraints
    # expr ≥ 0
    ineq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 
    # expr = 0
    eq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 

    # PURSE constraints
    for (i,q) in enumerate(q_backproj)
        push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
    end
    for (i,q) in enumerate(q_front)
        push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
    end
    for (i,q) in enumerate(q_eqs)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        push!(eq, [vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
    end

    # 90 degree rotation constraint
    if dim == 7
        push!(ineq, R'*center[1:4])
    # else
        # Rc = reshape(center[1:9],3,3)
        # push!(ineq, (tr(R'*Rc) - 1) / 2)
    end

    # solve
    pop = [obj; ineq; eq]
    order = order
    opt, sol, data, gap = cs_tssos_first(pop, vars, order, numeq=length(eq), TS=false, CS="MD", QUIET=silent, solution=true, refine=false)

    if data.SDP_status != MOI.OPTIMAL
        @warn "[bounding_sphere] Returned status $(data.SDP_status). Results may not be lower bound!"
        gap = -1
    end

    # CONVERT TO RADIUS
    rad = sqrt(-opt)

    return rad, data.SDP_status
end



"""
Refinement feasibility problem
"""
function check_feasibility(center, H_t, y, r, b, camK; order=1, silent=false)
    q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)

    return check_feasibility(center, H_t, q_front, q_backproj; order=order, silent=silent)
end

function check_feasibility(center, H_t, q_front, q_backproj; order=1, silent=false)
    @polyvar R[1:3,1:3]
    @polyvar t[1:3]
    vars = [vec(R); t]

    # objective
    V = eigvecs(H_t)
    l = eigvals(H_t)
    # obj = -t[3]
    obj = -(V'*t)[1]

    # constraints
    # expr ≥ 0
    ineq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 
    # expr = 0
    eq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 

    # pose uncertainty set constraints
    for (i,q) in enumerate(q_backproj)
        push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
    end
    for (i,q) in enumerate(q_front)
        push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
    end
    # ellipsoid constraints
    # push!(ineq, ((vars - center)'*H*(vars-center)-1) - 1e-2)
    push!(ineq, ((t - center[10:12])'*H_t*(t-center[10:12])-1) - 1e-2)

    V = eigvecs(H_t)
    l = eigvals(H_t)
    # # |x| ≤ l
    # append!(ineq, l .- V'*(t - center[10:12]))
    # append!(ineq, l .+ V'*(t - center[10:12]))

    # SO(3) constraints
    append!(eq, vec(R'*R - I)) # O(3)
    append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
    append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
    append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

    # solve
    pop = [obj; ineq; eq]
    order = order
    opt, sol, data, gap = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", CS="MF", QUIET=silent, solution=true, refine=false)

    ineq_val = [i(vars=>sol) for i in ineq]
    eq_val = [e(vars=>sol) for e in eq]

    # if isdefined(Main, :Infiltrator) Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__) end # 🚨 INFILTRATOR 🚨

    return sol, data.SDP_status
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
function refine_bbox(center, H, y, r, b, camK; p=2, mode=3, order=1, H_t=nothing, silent=false)
    (p == 2 || p == Inf) || error("only accepts `p=2` or `p=Inf`")

    if p == 2
        q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
        q_eqs = SO3_constraints()
    else
        ## Rotation Version
        q_front, q_backproj = uncertaintyset_linf_R(y, r, b, camK)
        q_eqs = SO3_constraints()
        ## Quaternion Version
        # q_front, q_backproj = uncertaintyset_linf_q(y, r, b, camK)
        # q_eqs = q_constraints()
    end

    if isnothing(H_t)
        # marginalize via projection
        # P = [zeros(3,p == 2 ? 9 : 4) diagm(ones(3))]
        P = [zeros(3,9) diagm(ones(3))]
        H_t = inv(P*pinv(H)*P')
    end
    return refine_bbox(center, H, q_front, q_backproj, q_eqs; p=p, mode=mode, order=order, H_t=H_t, silent=silent)
end

function refine_bbox(center, H, prob; mode=3, order=1, H_t=nothing, silent=false)
    refine_bbox(center, H, prob.y, prob.r, prob.b, prob.camK; p=prob.p, mode=mode, order=order, H_t=H_t, silent=silent)
end

function refine_bbox(center, H, q_front, q_backproj, q_eqs; p=2, mode=3, order=1, H_t=nothing, silent=false)
    if p == 2
        @polyvar R[1:3,1:3]
    else
        # @polyvar R[1:4] # quaternion
        @polyvar R[1:3,1:3]
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


"""
Idea for pre-marginalized uncertainty bounds
"""
function premarg_bounds(center, y, r, b, camK; silent=true, order=2)
    # variables
    N = size(r,1)
    @polyvar R[1:3,1:3]
    vars = vec(R)

    # eliminate t
    e3 = [0;0;1]
    U = Array{Any}(undef, N)
    for i = 1:N
        U[i] = (I - y[:,i]*e3')*camK
    end
    H = sum([U[i]'*U[i] / (r[i]^2) for i = 1:N])
    t = -inv(H)*sum([U[i]'*U[i]*R*b[:,i] / (r[i]^2) for i = 1:N])

    # objective
    Rc = reshape(center[1:9], 3,3)
    tc = center[10:12]
    # obj = -tr(R'*Rc)
    obj = -( tr((R-Rc)'*(R-Rc)) )
    # obj = -( (t-tc)'*(t-tc) )

    # constraints
    # expr ≥ 0
    ineq = Vector{TSSOS.Poly{Float64}}()
    # expr = 0
    eq = Vector{TSSOS.Poly{Float64}}()

    tol = 1e-3
    for i = 1:N
        # chirality constraints
        append!(ineq, [e3'*camK*(R*b[:,i] + t) + tol])

        # backprojection constraints
        res = (y[:,i]*e3' - I)*camK*(R*b[:,i] + t)
        append!(ineq, [(r[i]*e3'*camK*(R*b[:,i] + t))^2 - res'*res])
    end

    # SO(3) constraints
    append!(eq, vec(R'*R - I)) # O(3)
    append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
    append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
    append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

    # solve
    pop = [obj; ineq; eq]
    order = order
    opt, sol, data, gap, model = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=silent, solve=false, solution=false, MomentOne=true, refine=false)

    ## Extract solution
    R_est = project2SO3(reshape(sol[1:9],3,3))
    t_est = [ti(vars=>vec(R_est)) for ti in t]

    # ineq_val = (vars=>vec(R_est)) .|> ineq

    return R_est, t_est, gap, data.SDP_status
end


"""
RPY angular bounds.

Current status: solves with SLOW_PROGRESS for full PURSE constraints.
Solves to optimality with only ellipse constraints, but bounds are all 90 deg

TODO: try with a quaternion ellipse? Try with bounding sphere?
"""
function angular_bounds_rpy(center, H, prob; silent=false, order=3)
    Rc = reshape(center[1:9],3,3)
    # can marginalize out positions via projection
    P = [diagm(ones(9)) zeros(9,3)]
    H_r = inv(P*inv(H)*P')

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
        push!(ineq, 1 - (vec(R) - vec(Rc))'*H_r*(vec(R) - vec(Rc)))
        
        # c > 0 forces to be within π/2 of center--this is an assumption
        # but if it does not hold these bounds are the wrong approach anyways
        append!(ineq, c)

        # this constraint prevents degenerate solutions where s = 0
        tol = 1e-5
        append!(ineq, -(s .- tol))

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

    return Δθs, status_sdp, gaps
end



"""
Quaternion version of S-Lemma / bounding ellipse.

Implemented in a somewhat hacky way via TSSOS to achieve second order.
1. Build bounding sphere problem automatically with TSSOS.
2. Working in the dual space, replace the objective with a `logdet(H)` remove the `lower` variable.
3. Multiply all (15) constant terms in the dual by terms of `H`.
4. Add `H ⪰ 0` constraint.
5. Solve!
"""
function bounding_ellipse_quat(center, prob; order=2, silent=false)
    q_front, q_backproj = uncertaintyset_linf_q(prob.y, prob.r, prob.b, prob.camK)
    q_eqs = q_constraints()
    return bounding_ellipse_quat(center, q_backproj, q_front, q_eqs; order=order, silent=silent)
end

function bounding_ellipse_quat(center, q_backproj, q_front, q_eqs; order=2, silent=false)
    @polyvar q[1:4]
    @polyvar t[1:3]
    vars = [q; t]

    # sphere objective: minimize with placeholder shape & center
    # Ĥ = ones(7,7) # could replace with separate q, t term (match `H`)
    # Ĥ = diagm([1;1;1;1;1;1;1])
    # W = [ones(7)'*Ĥ*ones(7)  -ones(7)'*Ĥ;  -Ĥ*ones(7)  Ĥ]
    W = [1 -ones(7)'; -ones(7) ones(7,7)]
    # Non-1 terms corrected for later
    obj = -[1;vars]'*W*[1;vars]

    # constraints
    # expr ≥ 0
    ineq = Vector{TSSOS.Poly{Float64}}()
    # expr = 0
    eq = Vector{TSSOS.Poly{Float64}}()

    # backprojection
    for (_,q) in enumerate(q_backproj)
        push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
    end
    # chirality
    for (_,q) in enumerate(q_front)
        push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
    end
    # equality (just q² = 1)
    for (_,q) in enumerate(q_eqs)
        push!(eq, [vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
    end

    # constrain to rotations within 90°
    push!(ineq, q'*center[1:4])

    # use TSSOS to generate redundant constraints
    pop = [obj; ineq; eq]
    order = order
    # CS="MD" doesn't make a difference runtime wise
    opt, sol, data, gap, model = cs_tssos_first(pop, vars, order, numeq=length(eq), TS=false, CS=false, QUIET=false, solve=false, solution=false, MomentOne=true)

    if silent
        set_silent(model)
    end

    ## Modify model
    # add shape variable `H` (density must match `Ĥ`)
    @variable(model, H[1:7,1:7] ∈ PSDCone())
    # @variable(model, Hp[1:3,1:3] ∈ PSDCone())
    # H = [zeros(4,7); zeros(3,4) Hp]
    # @variable(model, h >= 0); H = diagm(h*ones(7))
    shape_mat = -[(center'*H*center - 1)  center'*H; H*center H] # with -1
    shapeΔ = triangle_vec(shape_mat)
    # update objective to logdet
    @variable(model, logdet_H)
    @objective(model, Max, logdet_H)
    @constraint(model, [logdet_H; 1; triangle_vec(H)] ∈ MOI.LogDetConeTriangle(7))
    # @objective(model, Max, tr(H))

    # remove the `lower` variable
    delete(model, model[:lower])
    unregister(model, :lower)
    # this alone completely removes `lower`

    # get constraints
    # they are stored as vector so not easy to modify in place
    co = constraint_object(model[:con])
    # remove `:con` from model
    delete(model, model[:con])
    unregister(model, :con)
    # modify constraints with constant terms
    # get PSD variables
    psdvars = all_variables(model)[1:length(shapeΔ)]
    shapeΔ = Dict(zip(psdvars, shapeΔ))
    tvW = Dict(zip(psdvars, abs.(triangle_vec(W))))
    for constraint in co.func
        if constraint.constant == 0
            @constraint(model, constraint == 0)
            continue
        end
        var = first(keys(constraint.terms))
        mult = -constraint.constant
        # remove constant term
        constraint.constant = 0
        # add constraint and correct for mult issues (division may not be necessary anymore)
        @constraint(model, constraint + mult*shapeΔ[var] / tvW[var] == 0)
    end

    ## optimize!
    set_optimizer(model, Mosek.Optimizer)
    optimize!(model)

    if !is_solved_and_feasible(model)
        @warn "[bounding_ellipse_quat] Returned status $(termination_status(model)). Results may not be lower bound!"
        gap = -1
    end

    return value.(H), termination_status(model)
end

"""
Tried to write all the redundant constraints manually for the bounding sphere problem.

Solves, but is not tight.
"""
function bounding_sphere_jump(center, y, r, b, K; silent=false)
    model = Model(Mosek.Optimizer)
    if silent
        set_silent(model)
    end
    dim = 1+7+28 # [1; q; t; q[1]*q; q[2]*q[2:4]; q[3]*q[3:4]; q[4]^2; t[1]*q; t[2]*q; t[3]*q; t[1]*t; t[2]*t[2:3]; t[3]^2]
    @variable(model, X[1:dim,1:dim] ∈ PSDCone())
    @constraint(model, X[1,1] == 1)

    # objective
    # minimize radius of ellipse centered at `center`
    W = [center'*center -center'; -center I]
    @objective(model, Max, tr(X[1:8,1:8]*W))

    # standard constraints
    N = size(y,2)
    e3 = [0;0;1]
    for i = 1:N
        # front of camera
        H = zeros(7,7)
        H[1:4, 1:4] = -(-Ω1(K'*e3)*Ω2(b[:,i]))
        H += H'
        c = -[zeros(4); K'*e3]
        @constraint(model, tr([0 c'; c H]*X[1:8,1:8]) <= 0)

        # backprojection
        iy3 = (I - y[:,i]*e3')
        for j = 1:2
            ej = zeros(3); ej[j] = 1

            H = zeros(7,7)
            H[1:4, 1:4] = (-Ω1((iy3*K)'*ej)*Ω2(b[:,i])) - r[i]*(-Ω1(K'*e3)*Ω2(b[:,i]))
            H += H'
            c = [zeros(4); (ej'*iy3*K)' - r[i]*(e3'*K)']
            @constraint(model, tr([0 c'; c H]*X[1:8,1:8]) <= 0)

            # negative term
            H = zeros(7,7)
            H[1:4, 1:4] = -(-Ω1((iy3*K)'*ej)*Ω2(b[:,i])) - r[i]*(-Ω1(K'*e3)*Ω2(b[:,i]))
            H += H'
            c = [zeros(4); -(ej'*iy3*K)' - r[i]*(e3'*K)']
            @constraint(model, tr([0 c'; c H]*X[1:8,1:8]) <= 0)
        end
    end
    # quaternion
    @constraint(model, tr(X[2:5,2:5]) == 1)

    # 90 degree rotation constraint (TODO: add redundant versions)
    @constraint(model, X[1,2:5]'*center[1:4] >= 0)
    # redundant versions
    # @constraint(model, tr(center[1:4]*center[1:4]'*X[2:5,2:5]) >= 0) # squared

    # redundant inequalities: backprojection
    ## make variables
    tX = triangle_vec(X[9:18,9:18])
    qqqqΔ = [tX[1:10]; tX[[2,3,5,8,3,3,12,13,14,12,15,16,17,18,19,17]]; tX[20:25];
                tX[[23,26,27,28,4,16,29,9,16,17,18,19,29,16,13,30]];
                tX[[19,17,20,33,27,30,33,6,30,31,32,30,21,34,42,31]]; # 49:64
                tX[[34,36,37,38,39,40,38,41,42,43,32,42,44,45,7,22]];
                tX[[9,10,8,14,19,25,37,38,39,40,46,22,23,24,25,23]];
                tX[[26,27,50,38,41,42,43,47,50,37,38,39,40,38,41,42]];
                tX[[43,39,42,44,45,40,43,45,46,47,48,49,25,28,43,52,48,51,53,54,49,52,54,55]]]
    # convert to symmetric matrix
    qqqq = X[9:24,9:24]*0
    qqqq = let
        counter=1
        for row = 1:16
            for col = 1:row
                qqqq[row,col] = qqqqΔ[counter]
                counter += 1
            end
        end
        qqqq
    end
    # make symmetric
    qqqq = qqqq + qqqq' - diagm(diag(qqqq))
    # other symmetric variable
    qtq = reshape(X[19:30,2:5],4,12) # = q*vec(t*q')'

    # add the backproj redundant constraints
    q_front, q_backproj = uncertaintyset_linf_q(y, r, b, K)
    for q1 in [q_backproj; q_front], q2 in [q_backproj; q_front]
        A1 = q1.H[1:4,1:4]; b1 = q1.c[5:7]
        A2 = q2.H[1:4,1:4]; b2 = q2.c[5:7]
        expr = tr(X[6:8,6:8]*(b1*b2' + b2*b1')/2)
        expr += tr(qqqq*kron(A1,A2))
        expr += tr(qtq'*(kron(A1,b2') + kron(A2,b1')))
        
        @constraint(model, expr >= 0)
    end

    # redundant equalities: q² = 1
    @constraint(model, tr(X[2:5,19:22]) == X[1,6]) # t1
    @constraint(model, tr(X[2:5,23:26]) == X[1,7]) # t2
    @constraint(model, tr(X[2:5,27:30]) == X[1,8]) # t3

    @constraint(model, tr(X[19:22,19:22]) == X[6,6]) # t1^2 
    @constraint(model, tr(X[23:26,23:26]) == X[7,7]) # t2^2
    @constraint(model, tr(X[27:30,27:30]) == X[8,8]) # t3^2

    @constraint(model, tr(X[27:30,19:22]) == X[8,6]) # t1 t3
    @constraint(model, tr(X[23:26,19:22]) == X[7,6]) # t1 t2
    @constraint(model, tr(X[27:30,23:26]) == X[8,7]) # t2 t3

    @constraint(model, tr(X[2:5,9:12]) == X[2,1]) # q1
    @constraint(model, X[2,10] + X[3,13] + X[4,14] + X[5,15] == X[3,1]) # q2
    @constraint(model, X[2,11] + X[3,14] + X[4,16] + X[5,17] == X[4,1]) # q3
    @constraint(model, X[2,12] + X[3,15] + X[4,17] + X[5,18] == X[2,1]) # q4

    # TODO: add q^2 (and cross terms)

    # moment constraints
    # q² = q²
    @constraint(model, [i=1:4], X[ 8+i,1] == X[1+i,2])
    @constraint(model, [i=1:3], X[12+i,1] == X[2+i,3])
    @constraint(model, [i=1:2], X[15+i,1] == X[3+i,4])
    @constraint(model, X[18,1] == X[5,5])
    # t² = t²
    @constraint(model, [i=1:3], X[30+i,1] == X[5+i,6])
    @constraint(model, [i=1:2], X[33+i,1] == X[6+i,7])
    @constraint(model, X[36,1] == X[8,8])
    # qt
    @constraint(model, [i=1:4], X[18+i,1] == X[6,1+i])
    @constraint(model, [i=1:4], X[22+i,1] == X[7,1+i])
    @constraint(model, [i=1:4], X[26+i,1] == X[8,1+i])
    # qt²  (qt*t = q*t²)
    @constraint(model, [i=1:4], X[7,18+i] == X[6,22+i])
    @constraint(model, [i=1:4], X[8,18+i] == X[6,26+i])
    @constraint(model, [i=1:4], X[8,22+i] == X[7,26+i])
    @constraint(model, X[2:5,31:33] .== X[6:8,19:22]')
    @constraint(model, X[2:5,34:35] .== X[7:8,23:26]')
    @constraint(model, X[2:5,36] .== X[8,27:30])
    # @constraint(model, X[6:8,19:30] .== X[2:5,31:36])
    # q²t (qt*q = q²*t)
    @constraint(model, [i=0:4:8], X[3:5,19+i] .== X[2,(20:22).+i]) # 3
    @constraint(model, [i=0:4:8], X[4:5,20+i] .== X[3,(21:22).+i]) # 2
    @constraint(model, [i=0:4:8], X[5,21+i] == X[4,22+i]) # 1
    @constraint(model, X[2,19:30] .== vec(X[6:8,9:12]'))
    @constraint(model, X[3,20:22] .== X[6,13:15])
    @constraint(model, X[3,24:26] .== X[7,13:15])
    @constraint(model, X[3,28:30] .== X[8,13:15])
    @constraint(model, X[4,21:22] .== X[6,16:17])
    @constraint(model, X[4,25:26] .== X[7,16:17])
    @constraint(model, X[4,29:30] .== X[8,16:17])
    @constraint(model, X[5,[22,26,30]] .== X[6:8,18])
    # @constraint(model, X[2:5,19:30] .== X[6:8,9:18])
    # q³ X[2:5,9:19]
    @constraint(model, X[3:5,9] .== X[2,10:12]) # q1^2 qx
    @constraint(model, X[3,10] == X[2,13])
    @constraint(model, X[4,11] == X[2,16])
    @constraint(model, X[5,12] == X[2,18])
    @constraint(model, X[4,13] == X[3,14]) # q2^2
    @constraint(model, X[5,13] == X[3,15])
    @constraint(model, X[4,14] == X[3,16])
    @constraint(model, X[5,15] == X[3,18])
    @constraint(model, X[5,16] == X[4,17]) # q3^2
    @constraint(model, X[5,17] == X[4,18])
    @constraint(model, X[4:5,10] .== X[3,11:12]) # qqq
    @constraint(model, X[4:5,10] .== X[2,14:15])
    @constraint(model, X[5,11] == X[4,12])
    @constraint(model, X[5,11] == X[2,17])
    @constraint(model, X[4,15] == X[5,14])
    @constraint(model, X[4,15] == X[3,17])

    # t³ X[6:8, 31:36]
    @constraint(model, X[7:8,31] .== X[6,32:33])
    @constraint(model, X[7,32] == X[6,34])
    @constraint(model, X[8,33] == X[6,36])
    @constraint(model, X[8,34] == X[7,35])
    @constraint(model, X[8,35] == X[7,36])
    @constraint(model, X[8,32] == X[7,33])
    @constraint(model, X[8,32] == X[6,35])

    # q²t² (q²*t² = qt*qt)
    # 18 repeated
    @constraint(model, X[24:26,19] .== X[23,20:22])
    @constraint(model, X[28:30,19] .== X[27,20:22])
    @constraint(model, X[28:30,23] .== X[27,24:26]) # q1--t2t3
    @constraint(model, X[25:26,20] .== X[24,21:22]) # q2--t1t2
    @constraint(model, X[29:30,20] .== X[28,21:22]) # q2--t1t3
    @constraint(model, X[29:30,24] .== X[28,25:26]) # q2--t2t3
    @constraint(model, X[26,21] == X[25,22]) # q3--t1t2
    @constraint(model, X[30,21] == X[29,22]) # q3--t1t3
    @constraint(model, X[30,25] == X[29,26]) # q3--t2t3

    @constraint(model, X[9:12,31] .== X[19:22,19])
    @constraint(model, X[13:15,31] .== X[20:22,20])
    @constraint(model, X[16:17,31] .== X[21:22,21])
    @constraint(model, X[18,31] == X[22,22]) # col 31
    @constraint(model, X[9:12,32] .== X[23:26,19])
    @constraint(model, X[13:15,32] .== X[24:26,20])
    @constraint(model, X[16:17,32] .== X[25:26,21])
    @constraint(model, X[18,32] == X[26,22]) # col 32
    @constraint(model, X[9:12,33] .== X[27:30,19])
    @constraint(model, X[13:15,33] .== X[28:30,20])
    @constraint(model, X[16:17,33] .== X[29:30,21])
    @constraint(model, X[18,33] == X[30,22]) # col 33
    @constraint(model, X[9:12,34] .== X[23:26,23])
    @constraint(model, X[13:15,34] .== X[24:26,24])
    @constraint(model, X[16:17,34] .== X[25:26,25])
    @constraint(model, X[18,34] == X[26,26]) # col 34
    @constraint(model, X[9:12,35] .== X[27:30,23])
    @constraint(model, X[13:15,35] .== X[28:30,24])
    @constraint(model, X[16:17,35] .== X[29:30,25])
    @constraint(model, X[18,35] == X[30,26]) # col 35
    @constraint(model, X[9:12,36] .== X[27:30,27])
    @constraint(model, X[13:15,36] .== X[28:30,28])
    @constraint(model, X[16:17,36] .== X[29:30,29])
    @constraint(model, X[18,36] == X[30,30]) # col 36

    # @constraint(model, X[9:18,31:36] .== X[19:30,19:30])
    # q⁴ (X[9:18,9:18])
    @constraint(model, X[14,12] == X[15,11])
    @constraint(model, X[14,12] == X[17,10])
    @constraint(model, X[18,9] == X[12,12])
    @constraint(model, X[16,9] == X[11,11])
    @constraint(model, X[13,9] == X[10,10])
    @constraint(model, X[16,13] == X[14,14])
    @constraint(model, X[18,13] == X[15,15])
    @constraint(model, X[18,16] == X[17,17])
    @constraint(model, X[14,9] == X[11,10])
    @constraint(model, X[15,9] == X[12,10])
    @constraint(model, X[17,9] == X[12,11])
    @constraint(model, X[14,10] == X[13,11])
    @constraint(model, X[15,10] == X[13,12])
    @constraint(model, X[17,13] == X[15,14])
    @constraint(model, X[16,10] == X[14,11])
    @constraint(model, X[17,11] == X[16,12])
    @constraint(model, X[17,14] == X[16,15])
    @constraint(model, X[18,10] == X[15,12])
    @constraint(model, X[18,11] == X[17,12])
    @constraint(model, X[18,14] == X[17,15])

    # t⁴
    @constraint(model, X[32:33,32] .== X[34:35,31])
    @constraint(model, X[33,33] == X[36,31])
    @constraint(model, X[34,33] == X[35,32])
    @constraint(model, X[35,35] == X[36,34])
    @constraint(model, X[35,33] == X[36,32])


    # solve
    optimize!(model)

    opt = objective_value(model)
    
    # Main.@infiltrate

    # CONVERT TO RADIUS
    rad = sqrt(opt)


    return rad, termination_status(model)
end