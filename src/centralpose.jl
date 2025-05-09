## Functions to compute the central pose solution
# Lorenzo Shaikewitz, 5/7/2025

using Printf
using LinearAlgebra
using TSSOS, DynamicPolynomials
using JuMP, Ipopt

# TODO: move to submodule
include("utils.jl")
include("refine.jl")

"""
    centralpose_l2(y, r, b, camK[;])

second order relaxation with l2 form of pose uncertainty set.

Returns pose estimate, optimization status, solution data, JuMP model

# Arguments
- `y`: pixel keypoints [3 x N] (homogenized)
- `r`: l2 radii for each keypoint [N]
- `b`: 3D canonical keypoint frame, meters [3 x N]
- `camK`: camera calibration matrix [3 x 3]
## Optional Arguments
- `lowerb=-0.8`: lower bound the margins
- `upperb=0.8`: upper bound the margins
- `silent=false`: should we print things?
"""
function centralpose_l2(y, r, b, camK; lowerb=-0.8, upperb=0.8, tol=1e-3, silent=false)
    N = size(r,1)
    @polyvar margin[1:N]
    @polyvar R[1:3,1:3]
    @polyvar t[1:3]
    vars = [margin; vec(R); t]

    # objective: max margin
    obj = -sum(margin)

    # constraints
    ineq = zeros(Polynomial{true, Float64}, 0) # expr ≥ 0
    eq = zeros(Polynomial{true, Float64}, 0)

    # pose uncertainty set
    X = [vec(R); t; 1]*[vec(R); t; 1]'
    for i = 1:N
        proj3dto2d = camK*(R*b[:,i] + t)
        # front of camera
        append!(ineq, [proj3dto2d[3]])
        # PURSE
        residual = (I - y[:,i]*[0;0;1]')*proj3dto2d
        # 2-norm
        append!(ineq, [(r[i] - margin[i])^2*(proj3dto2d[3])^2 - residual'*residual])
    end
    # SO(3) constraints
    append!(eq, vec(R'*R - I)) # O(3)
    append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
    append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
    append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

    # bounds
    if !isnothing(lowerb)
        append!(ineq, margin .- lowerb) # ≥ 0
    end
    append!(ineq, upperb .- margin) # ≥ 0

    # solve
    pop = [obj; ineq; eq]
    order = 2
    opt, sol, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=silent, solution=true, LorenzoOverride=true)
    sdp_sol,gap,data.flag = TSSOS.approx_sol(opt, data.moment, data.n, data.cliques, data.cql, data.cliquesize, data.supp, data.coe, numeq=data.numeq, tol=data.tol)
    
    sdp_sol_rounded = sdp_sol
    sdp_sol_rounded[end-3-9+1:end-3] = vec(project2SO3(reshape(sdp_sol[end-3-9+1:end-3],3,3)))

    sol, refine_status, gap = local_refine_tssos(opt, data; QUIET=silent, startpoint=sdp_sol_rounded)
    # sol, refine_status, gap = refine_l2(y, r, b, camK, sdp_sol_rounded, opt; lowerb=lowerb, upperb=upperb, tol=tol, silent=silent)

    if !silent
        println("SDP status: $(data.SDP_status)")
        println("Loc status: $(refine_status)")
    end

    R_est = project2SO3(reshape(sol[end-3-9+1:end-3],3,3))
    t_est = sol[end-2:end]

    vars_proj = [sol[1:end-3-9]; vec(R_est); t_est]

    # check feasibility
    ineq_subed = [ineq_i(vars=>vars_proj) for ineq_i in ineq]
    feasibility = sum(ineq_subed .< -tol) == 0
    
    return (R_est, t_est), data.SDP_status, refine_status, vars_proj, feasibility, gap
end

"""
    centralpose_percent_l2(y, r, b, camK[;])

second order relaxation with l2 form of pose uncertainty set.

Returns pose estimate, optimization status, solution data, JuMP model

# Arguments
- `y`: pixel keypoints [3 x N] (homogenized)
- `r`: l2 radii for each keypoint [N]
- `b`: 3D canonical keypoint frame, meters [3 x N]
- `camK`: camera calibration matrix [3 x 3]
## Optional Arguments
- `lowerb=0.2`: lower bound percentage
- `lowerb=2`: upper bound percentage
- `silent=false`: should we print things?
"""
function centralpose_percent_l2(y, r, b, camK; lowerb=0.2, upperb=2, tol=1e-3, silent=false)
    N = size(r,1)
    @polyvar margin[1:N]
    @polyvar R[1:3,1:3]
    @polyvar t[1:3]
    vars = [margin; vec(R); t]

    # objective: min margin percentages
    obj = sum(margin)

    # constraints
    ineq = zeros(Polynomial{true, Float64}, 0) # expr ≥ 0
    eq = zeros(Polynomial{true, Float64}, 0)

    # pose uncertainty set
    X = [vec(R); t; 1]*[vec(R); t; 1]'
    for i = 1:N
        proj3dto2d = camK*(R*b[:,i] + t)
        # front of camera
        append!(ineq, [proj3dto2d[3]])
        # PURSE
        residual = (I - y[:,i]*[0;0;1]')*proj3dto2d
        # 2-norm
        append!(ineq, [(r[i]*margin[i])^2*(proj3dto2d[3])^2 - residual'*residual])
    end
    # SO(3) constraints
    append!(eq, vec(R'*R - I)) # O(3)
    append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
    append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
    append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

    # bounds
    append!(ineq, margin  .- lowerb) # ≥ 0
    append!(ineq, upperb .- margin) # ≥ 0

    # solve
    pop = [obj; ineq; eq]
    order = 2
    opt, sol, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=silent, solution=true, LorenzoOverride=true)
    sdp_sol,gap,data.flag = TSSOS.approx_sol(opt, data.moment, data.n, data.cliques, data.cql, data.cliquesize, data.supp, data.coe, numeq=data.numeq, tol=data.tol)

    sdp_sol_rounded = sdp_sol
    sdp_sol_rounded[end-3-9+1:end-3] = vec(project2SO3(reshape(sdp_sol[end-3-9+1:end-3],3,3)))

    sol, refine_status, gap = local_refine_tssos(opt, data; QUIET=silent, startpoint=sdp_sol_rounded)

    if !silent
        println("SDP status: $(data.SDP_status)")
        println("Loc status: $(refine_status)")
    end

    R_est = project2SO3(reshape(sol[end-3-9+1:end-3],3,3))
    t_est = sol[end-2:end]

    vars_proj = [sol[1:end-3-9]; vec(R_est); t_est]

    # check feasibility
    ineq_subed = [ineq_i(vars=>vars_proj) for ineq_i in ineq]
    feasibility = sum(ineq_subed .< -tol) == 0
    
    return (R_est, t_est), data.SDP_status, refine_status, vars_proj, feasibility, gap
end


"""
    centralpose_linf(y, r, b, camK[;])

second order relaxation with linf form of pose uncertainty set.

Returns pose estimate, optimization status, solution data, JuMP model

# Arguments
- `y`: pixel keypoints [3 x N] (homogenized)
- `r`: l2 radii for each keypoint [N]
- `b`: 3D canonical keypoint frame, meters [3 x N]
- `camK`: camera calibration matrix [3 x 3]
## Optional Arguments
- `lowerb=-0.8`: lower bound the margins
- `upperb=0.8`: upper bound the margins
- `silent=false`: should we print things?
"""
function centralpose_linf(y, r, b, camK; lowerb=-0.8, upperb=0.8, tol=1e-3, silent=false)
    N = size(r,1)
    @polyvar margin[1:N]
    @polyvar R[1:3,1:3]
    @polyvar t[1:3]
    vars = [margin; vec(R); t]

    # objective: max margin
    obj = -sum(margin)

    # constraints
    ineq = zeros(Polynomial{true, Float64}, 0) # expr ≥ 0
    eq = zeros(Polynomial{true, Float64}, 0)

    # pose uncertainty set
    for i = 1:N
        proj3dto2d = camK*(R*b[:,i] + t)
        # front of camera
        append!(ineq, [proj3dto2d[3]])
        # PURSE
        residual = (I - y[:,i]*[0;0;1]')*proj3dto2d
        # inf-norm
        append!(ineq, (r[i] - margin[i])*proj3dto2d[3] .- residual)
        append!(ineq, (r[i] - margin[i])*proj3dto2d[3] .+ residual)
    end
    # SO(3) constraints
    append!(eq, vec(R'*R - I)) # O(3)
    append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
    append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
    append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

    # bounds
    if !isnothing(lowerb)
        append!(ineq, margin .- lowerb) # ≥ 0
    end
    append!(ineq, upperb .- margin) # ≥ 0

    # solve
    pop = [obj; ineq; eq]
    order = 1
    opt, sol, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=silent, solution=true, LorenzoOverride=true)
    sdp_sol,gap,data.flag = TSSOS.approx_sol(opt, data.moment, data.n, data.cliques, data.cql, data.cliquesize, data.supp, data.coe, numeq=data.numeq, tol=data.tol)
    
    sdp_sol_rounded = sdp_sol
    sdp_sol_rounded[end-3-9+1:end-3] = vec(project2SO3(reshape(sdp_sol[end-3-9+1:end-3],3,3)))

    sol, refine_status, gap = local_refine_tssos(opt, data; QUIET=silent, startpoint=sdp_sol_rounded)
    # sol, refine_status, gap = refine_linf(y, r, b, camK, sdp_sol_rounded, opt; lowerb=lowerb, upperb=upperb, tol=tol, silent=silent)

    if !silent
        println("SDP status: $(data.SDP_status)")
        println("Loc status: $(refine_status)")
    end

    R_est = project2SO3(reshape(sol[end-3-9+1:end-3],3,3))
    t_est = sol[end-2:end]

    vars_proj = [sol[1:end-3-9]; vec(R_est); t_est]

    # check feasibility
    ineq_subed = [ineq_i(vars=>vars_proj) for ineq_i in ineq]
    feasibility = sum(ineq_subed .< -tol) == 0
    
    return (R_est, t_est), data.SDP_status, refine_status, vars_proj, feasibility, gap
end


"""
    centralpose_percent_l2(y, r, b, camK[;])

second order relaxation with l2 form of pose uncertainty set.

Returns pose estimate, optimization status, solution data, JuMP model

# Arguments
- `y`: pixel keypoints [3 x N] (homogenized)
- `r`: l2 radii for each keypoint [N]
- `b`: 3D canonical keypoint frame, meters [3 x N]
- `camK`: camera calibration matrix [3 x 3]
## Optional Arguments
- `lowerb=0.2`: lower bound percentage
- `lowerb=2`: upper bound percentage
- `silent=false`: should we print things?
"""
function centralpose_percent_linf(y, r, b, camK; lowerb=0.2, upperb=2, tol=1e-3, silent=false, double_local=false)
    N = size(r,1)
    @polyvar margin[1:N]
    @polyvar R[1:3,1:3]
    @polyvar t[1:3]
    vars = [margin; vec(R); t]

    # objective: min margin percentages
    obj = sum(margin)

    # constraints
    ineq = zeros(Polynomial{true, Float64}, 0) # expr ≥ 0
    eq = zeros(Polynomial{true, Float64}, 0)

    # pose uncertainty set
    for i = 1:N
        proj3dto2d = camK*(R*b[:,i] + t)
        # front of camera
        append!(ineq, [proj3dto2d[3]])
        # PURSE
        residual = (I - y[:,i]*[0;0;1]')*proj3dto2d
        # inf-norm
        append!(ineq, (r[i]*margin[i])*proj3dto2d[3] .- residual)
        append!(ineq, (r[i]*margin[i])*proj3dto2d[3] .+ residual)
    end
    # SO(3) constraints
    append!(eq, vec(R'*R - I)) # O(3)
    append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
    append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
    append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

    # bounds
    append!(ineq, margin  .- lowerb) # ≥ 0
    append!(ineq, upperb .- margin) # ≥ 0

    # solve
    pop = [obj; ineq; eq]
    order = 1
    opt, sol, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=silent, solution=true, LorenzoOverride=true)
    sdp_sol,gap,data.flag = TSSOS.approx_sol(opt, data.moment, data.n, data.cliques, data.cql, data.cliquesize, data.supp, data.coe, numeq=data.numeq, tol=data.tol)

    sdp_sol_rounded = sdp_sol
    sdp_sol_rounded[end-3-9+1:end-3] = vec(project2SO3(reshape(sdp_sol[end-3-9+1:end-3],3,3)))

    if !double_local
        sol, refine_status, gap = local_refine_tssos(opt, data; QUIET=silent, startpoint=sdp_sol_rounded)
    else
        start = sdp_sol_rounded
    
        # local solver
        model = Model(Ipopt.Optimizer)
        if silent
            set_silent(model)
        end
        margin = start[1:end-3-9]
        R_est = reshape(start[end-3-9+1:end-3],3,3)
        t_est = start[end-2:end]
        N = size(margin,1)
    
        @variable(model, margin_local[i=1:N], start=margin[i])
        @variable(model, R[i=1:3,j=1:3], start=R_est[i,j])
        @variable(model, t[i=1:3], start=t_est[i])
    
        @objective(model, Min, sum(margin_local))
    
        # constraints
        for i = 1:N
            proj3dto2d = camK*(R*b[:,i] + t)
            # front of camera
            @constraint(model, proj3dto2d[3] >= 0)
            # PURSE
            residual = (I - y[:,i]*[0;0;1]')*proj3dto2d
            # 2-norm
            @constraint(model, margin_local[i]*(r[i])^2*(proj3dto2d[3])^2 - residual'*residual >= 0)
            # inf-norm
            @constraint(model, (r[i]*margin_local[i])*proj3dto2d[3] .- residual >= 0)
            @constraint(model, (r[i]*margin_local[i])*proj3dto2d[3] .+ residual >= 0)
        end
        @constraint(model, vec(R'*R - I) .== 0)
        @constraint(model, R[1:3,3] .== cross(R[1:3,1],R[1:3,2]))
        @constraint(model, R[1:3,1] .== cross(R[1:3,2],R[1:3,3]))
        @constraint(model, R[1:3,2] .== cross(R[1:3,3],R[1:3,1]))
    
        # bounds
        if !isnothing(lowerb)
            @constraint(model, margin_local .>= lowerb)
        end
        @constraint(model, margin_local .<= upperb)
    
        optimize!(model)
    
        refine_status = termination_status(model)
        sol = [value.(margin_local); vec(value.(R)); value.(t)]

        ub = objective_value(model)
        gap = abs(opt-ub)/max(1, abs(ub))

        (sol, refine_status, gap)
    end

    if !silent
        println("SDP status: $(data.SDP_status)")
        println("Loc status: $(refine_status)")
    end

    R_est = project2SO3(reshape(sol[end-3-9+1:end-3],3,3))
    t_est = sol[end-2:end]

    vars_proj = [sol[1:end-3-9]; vec(R_est); t_est]

    # check feasibility
    ineq_subed = [ineq_i(vars=>vars_proj) for ineq_i in ineq]
    feasibility = sum(ineq_subed .< -tol) == 0
    
    return (R_est, t_est), data.SDP_status, refine_status, vars_proj, feasibility, gap
end


function centralpose_percent_linf_LOCAL(y, r, b, camK; lowerb=0.2, upperb=2, tol=1e-3, silent=false, double_local=false)
    N = size(r,1)

    # local solver
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 250)
    if silent
        set_silent(model)
    end

    @variable(model, margin_local[i=1:N])
    @variable(model, R[i=1:3,j=1:3])
    @variable(model, t[i=1:3])

    @objective(model, Min, sum(margin_local))

    # constraints
    for i = 1:N
        proj3dto2d = camK*(R*b[:,i] + t)
        # front of camera
        @constraint(model, proj3dto2d[3] >= 0)
        # PURSE
        residual = (I - y[:,i]*[0;0;1]')*proj3dto2d
        # 2-norm
        @constraint(model, margin_local[i]*(r[i])^2*(proj3dto2d[3])^2 - residual'*residual >= 0)
        # inf-norm
        @constraint(model, (r[i]*margin_local[i])*proj3dto2d[3] .- residual >= 0)
        @constraint(model, (r[i]*margin_local[i])*proj3dto2d[3] .+ residual >= 0)
    end
    @constraint(model, vec(R'*R - I) .== 0)
    @constraint(model, R[1:3,3] .== cross(R[1:3,1],R[1:3,2]))
    @constraint(model, R[1:3,1] .== cross(R[1:3,2],R[1:3,3]))
    @constraint(model, R[1:3,2] .== cross(R[1:3,3],R[1:3,1]))

    # bounds
    if !isnothing(lowerb)
        @constraint(model, margin_local .>= lowerb)
    end
    @constraint(model, margin_local .<= upperb)

    x = all_variables(model)
    for iteration = 1:10
        start = zeros(N + 9 + 3)
        start[end-9-3+1:end-3] = vec(randrotation())
        set_start_value.(x, start)
        optimize!(model)
        if is_solved_and_feasible(model)
            break
        end
    end

    refine_status = termination_status(model)
    sol = [value.(margin_local); vec(value.(R)); value.(t)]

    ub = objective_value(model)
    # gap = abs(opt-ub)/max(1, abs(ub))

    if !silent
        println("Loc status: $(refine_status)")
    end

    R_est = project2SO3(reshape(sol[end-3-9+1:end-3],3,3))
    t_est = sol[end-2:end]

    vars_proj = [sol[1:end-3-9]; vec(R_est); t_est]
    
    return (R_est, t_est), refine_status, refine_status, vars_proj, is_solved_and_feasible(model), ub
end