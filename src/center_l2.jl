## Functions to compute the central pose
# Lorenzo Shaikewitz, 4/29/2025

using Printf
using LinearAlgebra
using JuMP, Clarabel, Ipopt

# TODO: move to submodule
include("utils.jl")

"""
    sdp_pose, status, feas, _ = center_l2(q_front, q_backproj, q_eqs;[])

First order relaxation with l2 form of pose uncertainty set.

Returns pose estimate, optimization status, solution data, JuMP model

# Arguments
- `q_front`: list of quadratics ≤ 0 for chirality constraints
- `q_backproj`: list of quadratics ≤ 0 for backproj constraints
- `q_eqs`: list of quadratics = 0 for SO(3) constraints
## Optional Arguments
- `analytic=false`: use the analytic center instead of max. margin
- `lowerb=false`: lower bound the margins by 0 (irrelevant if `analytic=true`)
- `upperb=10`: controls constraint `margin < upperb*r`
- `silent=false`: should we print things?
"""
function center_l2(q_front, q_backproj, q_eqs; analytic=false, lowerb=-1, upperb=10, silent=false, tlower=nothing, tupper=nothing)
    N = length(q_front)

    model = Model(Clarabel.Optimizer)
    if silent
        set_silent(model)
    end
    @variable(model, margin[1:N])
    @variable(model, X[1:13,1:13] ∈ PSDCone()) # rank 1 of [r, t, 1]
    @constraint(model, X[13,13]==1.)

    # objective
    if analytic
        @variable(model, log_margin[1:N])
        obj = sum(log_margin)
        for i = 1:N
            @constraint(model, [log_margin[i], 1, margin[i]] ∈ MOI.ExponentialCone())
        end
    else
        obj = sum(margin)
    end
    @objective(model, Max, obj)

    # constraints
    for (i,q) in enumerate(q_front)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        @constraint(model, tr(Q*X) <= 0.)
    end
    for (i,q) in enumerate(q_backproj)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        @constraint(model, tr(Q*X) <= -margin[i])
    end
    for (i,q) in enumerate(q_eqs)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        @constraint(model,  tr(Q*X) == 0.)
    end
    if !isnothing(lowerb)
        if analytic
            lowerb = max(lowerb, 0.)
        end
        @constraint(model, margin .>= lowerb)
    end
    @constraint(model, margin .<= upperb)
    if !isnothing(tlower)
        @constraint(model, X[10:12,13] >= tlower)
    end
    if !isnothing(tupper)
        @constraint(model, X[10:12,13] <= tupper)
    end

    # solve
    optimize!(model)

    X_val = Symmetric(value.(X))
    margin_val = value.(margin)
    opt = value(obj)

    # round solution
    x = eigvecs(X_val)[:,end]
    x /= x[end]
    R_est = project2SO3(reshape(x[1:9],3,3))
    t_est = x[10:12]

    tight = rank(X_val, 1e-4) == 1
    if !tight
        evs = eigvals(X_val)
        if !silent
            printstyled("X not rank 1: λ₁ = $(evs[end]), λ₂ = $(evs[end-1])\n", color=:red)
        end
    end

    vars_proj = [margin_val; vec(R_est); t_est]
    data = (tight, opt, vars_proj)

    return ((R_est, t_est), termination_status(model), data, model)
end


function local_refine(q_front, q_backproj, q_eqs, data; analytic=analytic, lowerb=lowerb, upperb=upperb, silent=silent, tlower=nothing, tupper=nothing)
    model = Model(Ipopt.Optimizer)

    (_, opt, vars_start) = data
    margin = vars_start[1:end-3-9]
    R_est = reshape(vars_start[end-3-9+1:end-3],3,3)
    t_est = vars_start[end-2:end]

    N = size(margin,1)
    if silent
        set_silent(model)
    end

    @variable(model, margin_local[i=1:N], start=margin[i])
    @variable(model, R[i=1:3,j=1:3], start=R_est[i,j])
    @variable(model, t[i=1:3], start=t_est[i])

    # objective
    if analytic
        @variable(model, log_margin[1:N])
        obj = sum(log_margin)
        for i = 1:N
            @constraint(model, [log_margin[i], 1, margin[i]] ∈ MOI.ExponentialCone())
        end
    else
        obj = sum(margin_local)
    end
    @objective(model, Max, obj)

    # constraints
    x_local = [vec(R); t; 1]
    X_local = x_local*x_local'
    for (i,q) in enumerate(q_front)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        @constraint(model, tr(Q*X_local) <= 0.)
    end
    for (i,q) in enumerate(q_backproj)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        @constraint(model, tr(Q*X_local) <= -margin_local[i])
    end
    for (i,q) in enumerate(q_eqs)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        @constraint(model,  tr(Q*X_local) == 0.)
    end

    if !isnothing(lowerb)
        @constraint(model, margin_local .>= lowerb)
    end
    @constraint(model, margin_local .<= upperb)

    if !isnothing(tlower)
        @constraint(model, X_local[10:12,13] >= tlower)
    end
    if !isnothing(tupper)
        @constraint(model, X_local[10:12,13] <= tupper)
    end

    optimize!(model)
    vars_proj = [value.(margin_local); vec(value.(R)); value.(t)]

    # compute gap
    ub = value(obj)
    gap = abs(opt-ub)/max(1, abs(ub))

    return (value.(R), value.(t)), termination_status(model), vars_proj, gap
end
