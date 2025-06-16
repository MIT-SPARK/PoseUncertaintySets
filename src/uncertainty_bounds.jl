## Functions for propagating uncertainty via conformal prediction
# Lorenzo Shaikewitz, 6/11/2025


"""
    bounding_ellipse(center, y, r, b, K[; solver=Clarabel.Optimizer, silent=false])

S-Lemma to outer bound pose uncertainty set. `(x-center)'*H*(x-center) ≤ 1`

Returns ellipse matrix `H`, optimization status.

# Arguments
- `center`: center of ellipse [vec(R), t]
- `y`: pixel keypoints [3 x N] (homogenized)
- `r`: l2 radii for each keypoint [N]
- `b`: 3D canonical keypoint frame, meters [3 x N]
- `camK`: camera calibration matrix [3 x 3]
## Optional Arguments
- `solver=Clarabel.Optimizer`: what solver to use
- `silent=false`: should we print things?

# Returns
- `H`: PSD ellipse matrix
- `status`: termination status of optimization
"""
function bounding_ellipse(center, y, r, b, camK; solver=Clarabel.Optimizer, silent=false)
    q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
    q_eqs = SO3_constraints()

    return bounding_ellipse(center, q_front, q_backproj, q_eqs; solver=solver, silent=silent)
end


function bounding_ellipse(center, q_front, q_backproj, q_eqs; solver=Clarabel.Optimizer, silent=false)

    # JuMP model
    model = Model(solver)
    if silent
        set_silent(model)
    end
    @variable(model, log_det_H0)
    @variable(model, λ[1:length(q_front) + length(q_backproj)] .>= 0)
    @variable(model, η[1:length(q_eqs)])

    # full 12 x 12
    @variable(model, H0[1:12,1:12] ∈ PSDCone())

    # rotation and position independent
    # @variable(model, H0_r[1:9,1:9] ∈ PSDCone())
    # @variable(model, H0_p[1:3,1:3] ∈ PSDCone())
    # H0 = [H0_r zeros(9,3); zeros(3,9) H0_p]
    # H0 = [zeros(9,9) zeros(9,3); zeros(3,9) H0_p]
    # H0 = [H0_r zeros(9,3); zeros(3,9) zeros(3,3)]

    # diagonal
    # @variable(model, r_H0[1:12] .>= 0)
    # H0 = diagm(r_H0)

    # objective: max logdet(H0)
    # use auxillary variable
    @objective(model, Max, log_det_H0)
    @constraint(model, [log_det_H0; 1; vec(H0)] in MOI.LogDetConeSquare(12))

    # build and constrain M
    # q0 = x'*H0*x + 2(-H0*c)'*x + c'*H0*c <= 1
    M = -[H0  -H0*center;  (-H0*center)'  center'*H0*center-1]
    for (i_bp,q) in enumerate(q_backproj)
        i = i_bp
        M += [λ[i]*q.H  λ[i]*q.c;  λ[i]*q.c'  λ[i]*q.d]
    end
    for (i_fc,q) in enumerate(q_front)
        i = i_fc + length(q_backproj)
        M += [λ[i]*q.H  λ[i]*q.c;  λ[i]*q.c'  λ[i]*q.d]
    end
    for (i,q) in enumerate(q_eqs)
        M += [η[i]*q.H  η[i]*q.c;  η[i]*q.c'  η[i]*q.d]
    end
    @constraint(model, M >= 0, PSDCone())

    # Solve with JuMP
    optimize!(model)

    if !silent && !is_solved_and_feasible(model)
        @warn "Solver did not find an optimal solution!"
    end
    H0_val = value.(H0)

    return (H0_val, termination_status(model))
end


"""
    angular_bounds(center, H; silent=true)

Compute angular bounds given center, ellipse.

We marginalize out positions via projection, 
and assume `cos(θ) > 0`.
"""
function angular_bounds(center, H; silent=false, order=2)
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
        opt, sol, gap, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=silent, solution=true, refine=true)

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
function purse_bounds(center, y, r, b, camK; order=2, silent=false)
    q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
    q_eqs = SO3_constraints()

    return purse_bounds(center, q_front, q_backproj, q_eqs; order=order, silent=silent)
end

function purse_bounds(center, q_front, q_backproj, q_eqs; order=2, silent=false)
    Rc = reshape(center[1:9],3,3)
    tc = center[10:12]

    @polyvar R[1:3,1:3]
    @polyvar t[1:3]
    vars = [vec(R); t]

    ang_bound = 0.
    ang_gap = 1e6
    trans_bound = 0.
    trans_gap = 1e6

    for λ = [0, 1]
        # objective
        obj = -( λ*tr((R-Rc)'*(R-Rc)) + (1-λ)*(t-tc)'*(t-tc) )

        # constraints
        # expr ≥ 0
        ineq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 
        # expr = 0
        eq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 

        # PURSE constraints
        X = [vars; 1]*[vars; 1]'
        for (i,q) in enumerate(q_front)
            Q = Symmetric([q.H  q.c;  q.c'  q.d])
            push!(ineq, -tr(Q*X))
        end
        for (i,q) in enumerate(q_backproj)
            Q = Symmetric([q.H  q.c;  q.c'  q.d])
            push!(ineq, -tr(Q*X))
        end
        for (i,q) in enumerate(q_eqs)
            Q = Symmetric([q.H  q.c;  q.c'  q.d])
            push!(eq, tr(Q*X))
        end

        # SO(3) constraints
        # append!(eq, vec(R'*R - I)) # O(3)
        # append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
        # append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
        # append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

        # solve
        pop = [obj; ineq; eq]
        order = order # supplementary material: they use second order
        opt, sol, gap, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS=false, CS=false, QUIET=silent, solution=true, refine=false)

        # if isdefined(Main, :Infiltrator) Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__) end # 🚨 INFILTRATOR 🚨

        if data.SDP_status != MOI.OPTIMAL
            @warn "λ=$λ returned status $(data.SDP_status). Results may not be lower bound!"
            gap = -1
        end

        if λ == 1
            # |R₁ - R₂|^2_F = |R₁|^2_F + |R₂|^2_F - 2⟨R₁, R₂⟩
            # ⟨R₁, R₂⟩ = (6 - |R₁ - R₂|^2_F) / 2
            frob_norm = -opt
            inner_prod = (6 - frob_norm)/2
            ang_bound = acos((inner_prod - 1) / 2)*180/π
            ang_gap = gap
        else
            trans_bound = -opt
            trans_gap = gap
        end
    end

    return trans_bound, trans_gap, ang_bound, ang_gap
end

"""
Solving the bounding sphere problem with a direct relaxation.

Returns radius of sphere and SDP status.

Why solve for a joint bounding sphere? The RANSAG approach makes much more sense.
"""
function bounding_sphere(center, y, r, b, camK; order=1, silent=false)
    q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
    q_eqs = SO3_constraints()

    return bounding_sphere(center, q_front, q_backproj, q_eqs; order=order, silent=silent)
end


function bounding_sphere(center, q_front, q_backproj, q_eqs; order=1, silent=false)

    Rc = reshape(center[1:9],3,3)
    tc = center[10:12]

    @polyvar R[1:3,1:3]
    @polyvar t[1:3]
    vars = [vec(R); t]

    # objective
    W = [I -center; -center' center'*center]
    obj = -[vars;1]'*W*[vars;1] # equivalent to minimizing radius of ellipse centered at `center`

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

    # SO(3) constraints
    # append!(eq, vec(R'*R - I)) # O(3)
    # append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
    # append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
    # append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

    # solve
    pop = [obj; ineq; eq]
    order = order
    opt, sol, gap, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS=false, CS="MF", QUIET=silent, solution=true, refine=false)

    # if isdefined(Main, :Infiltrator) Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__) end # 🚨 INFILTRATOR 🚨

    if data.SDP_status != MOI.OPTIMAL
        @warn "Returned status $(data.SDP_status). Results may not be lower bound!"
        gap = -1
    end

    return -opt, data.SDP_status
end