## Compute bounding ellipsoids and spheres
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
function bounding_ellipse(center, y, r, b, camK; p=2, solver=Mosek.Optimizer, order=1, silent=false)
    (p == 2 || p == Inf) || error("only accepts `p=2` or `p=Inf`")

    if p == 2
        q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
        q_eqs = SO3_constraints()
    else
        ## Rotation Version
        q_front, q_backproj = uncertaintyset_linf_R(y, r, b, camK)
        q_new = []
        # TODO: fix this it is slow!
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

    return bounding_ellipse(center, q_front, q_backproj, q_eqs; solver=solver, order=order, silent=silent)
end


function bounding_ellipse(center, prob; solver=Mosek.Optimizer, order=1, silent=false)
    return bounding_ellipse(center, prob.y, prob.r, prob.b, prob.camK; p=prob.p, solver=solver, order=order, silent=silent)
end


function bounding_ellipse(center, q_front, q_backproj, q_eqs; solver=Mosek.Optimizer, order=1, silent=false)
    (order == 1) || return bounding_ellipse_higherorder(center, q_front, q_backproj, q_eqs; solver=solver, order=order, silent=silent)

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
    # @constraint(model, psdcon, triangle_vec(M) ∈ MOI.PositiveSemidefiniteConeTriangle(dim+1))
    @constraint(model, psdcon, M >= 0, PSDCone())

    # Solve with JuMP
    optimize!(model)

    # TODO: not a great way to compute the gap is it?
    X = dual(model[:psdcon])
    gap = sum(eigvals(X) .> 1e-2) - 1

    if !is_solved_and_feasible(model)
        gap = -1
        silent || @warn "[bounding_ellipse] Solver terminated with status $(termination_status(model))"
    end
    H0_val = value.(H0)

    return (H0_val, gap, termination_status(model))
end


"""
Bounding ellipse with higher orders
"""
function bounding_ellipse_higherorder(center, y, r, b, camK; p=2, solver=Mosek.Optimizer, order=1, silent=false)
    (p == 2 || p == Inf) || error("only accepts `p=2` or `p=Inf`")

    if p == 2
        q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
        q_eqs = SO3_constraints()
    else
        ## Rotation Version
        q_front, q_backproj = uncertaintyset_linf_R(y, r, b, camK)
        q_new = []
        # TODO: fix this it is slow!
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

    return bounding_ellipse_higherorder(center, q_front, q_backproj, q_eqs; solver=solver, order=order, silent=silent)
end

function bounding_ellipse_higherorder(center, prob; solver=Mosek.Optimizer, order=1, silent=false)
    return bounding_ellipse_higherorder(center, prob.y, prob.r, prob.b, prob.camK; p=prob.p, solver=solver, order=order, silent=silent)
end

function bounding_ellipse_higherorder(center, q_front, q_backproj, q_eqs; solver=Mosek.Optimizer, order=1, silent=false)
    @polyvar R[1:3,1:3]
    @polyvar t[1:3]
    vars = [vec(R); t]

    # use proxy objective
    W = [1 -ones(12)'; -ones(12) ones(12,12)]
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
    # equality (R∈SO(3))
    for (_,q) in enumerate(q_eqs)
        push!(eq, [vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
    end

    # use TSSOS to generate redundant constraints
    pop = [obj; ineq; eq]
    order = order
    # CS="MD" doesn't make a difference runtime wise
    opt, sol, data, gap, model = cs_tssos_first(pop, vars, order, numeq=length(eq), TS=false, CS=false, QUIET=silent, solve=false, solution=false, MomentOne=true)

    if silent
        set_silent(model)
    end

    ## Modify model
    # add shape variable `H` (density must match `Ĥ`)
    @variable(model, H[1:12,1:12] ∈ PSDCone())
    shape_mat = -[(center'*H*center - 1)  center'*H; H*center H] # with -1
    shapeΔ = triangle_vec(shape_mat)
    # update objective to logdet
    @variable(model, logdet_H)
    @objective(model, Max, logdet_H)
    @constraint(model, [logdet_H; 1; triangle_vec(H)] ∈ MOI.LogDetConeTriangle(12))
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
    # tvW = Dict(zip(psdvars, abs.(triangle_vec(W)))) # all +1
    for constraint in co.func
        if constraint.constant == 0
            @constraint(model, constraint == 0)
            continue
        end
        var = first(keys(constraint.terms))
        mult = -constraint.constant
        # remove constant term
        constraint.constant = 0
        # add constraint and correct for mult issues
        @constraint(model, constraint + mult*shapeΔ[var] == 0)
    end

    ## optimize!
    set_optimizer(model, solver)
    optimize!(model)

    # TODO: less sloppy
    c = [all_constraints(model, t...) for t in list_of_constraint_types(model)]
    X = dual(c[2][1])
    # this appears to be a focal point? It is certainly not an extreme point
    gap = 1
    if rank(X[1:13,1:13],1e-2) == 1
        gap = 0
    end

    # TODO: I actually don't care about SLOW_PROGRESS
    if !is_solved_and_feasible(model)
        gap = -1
        silent || @warn "[bounding_ellipse_higherorder] Returned status $(termination_status(model)). Results may not be lower bound!"
    end

    return value.(H), gap, termination_status(model)
end


"""
Solving the bounding sphere problem with a direct relaxation.

Returns radius of sphere and SDP status.

Why solve for a joint bounding sphere? The RANSAG approach makes much more sense.
"""
function bounding_sphere(center, y, r, b, camK; p=2, R=false, order=1, silent=false)
    (p == 2 || p == Inf) || error("only accepts `p=2` or `p=Inf`")

    if p == 2
        q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
        q_eqs = SO3_constraints()
    else
        if R
            q_front, q_backproj = uncertaintyset_linf_R(y, r, b, camK)
            q_new = []
            for q1 in q_backproj, q2 in q_backproj
                H = -q1.c*q2.c'
                H += H'
                push!(q_new, Quadratic(H, zeros(12), 0.)) # ≤ 0
            end
            q_front = [q_front; q_new]
            q_eqs = SO3_constraints()
        else
            q_front, q_backproj = uncertaintyset_linf_q(y, r, b, camK)
            q_eqs = q_constraints()
        end
    end

    return bounding_sphere(center, q_front, q_backproj, q_eqs; order=order, silent=silent)
end

function bounding_sphere(center, prob; R=false, order=1, silent=false)
    return bounding_sphere(center, prob.y, prob.r, prob.b, prob.camK; p=prob.p, R=R, order=order, silent=silent)
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
    ineq = Vector{TSSOS.Poly{Float64}}()
    # expr = 0
    eq = Vector{TSSOS.Poly{Float64}}()

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
        silent || @warn "[bounding_sphere] Returned status $(data.SDP_status). Results may not be lower bound!"
        gap = -1
    end

    # CONVERT TO RADIUS
    rad = sqrt(-opt)

    return rad, gap, data.SDP_status
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
    return bounding_ellipse_quat(center, prob.y, prob.r, prob.b, prob.camK; order=order, silent=silent)
end

function bounding_ellipse_quat(center, y, r, b, camK; order=2, silent=false)
    q_front, q_backproj = uncertaintyset_linf_q(y, r, b, camK)
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
    opt, sol, data, gap, model = cs_tssos_first(pop, vars, order, numeq=length(eq), TS=false, CS=false, QUIET=silent, solve=false, solution=false, MomentOne=true)

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
    tvW = Dict(zip(psdvars, abs.(triangle_vec(W)))) # all +1
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

    # TODO: less sloppy
    c = [all_constraints(model, t...) for t in list_of_constraint_types(model)]
    X = dual(c[2][1])
    # this appears to be a focal point? It is certainly not an extreme point
    gap = 1
    if rank(X[1:8,1:8],1e-2) == 1
        gap = 0
    end

    # TODO: I actually don't care about SLOW_PROGRESS
    if !is_solved_and_feasible(model)
        gap = -1
        silent || @warn "[bounding_ellipse_quat] Returned status $(termination_status(model)). Results may not be lower bound!"
    end

    return value.(H), gap, termination_status(model)#, X
end