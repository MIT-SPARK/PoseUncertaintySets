## Functions for pose estimation from keypoints
# Lorenzo Shaikewitz, 6/4/2025

"""
    gaussianpose(y, r, b, camK; silent=true, order=2)

Certifiable PnP with Gaussian noise assumption.

# Arguments:
- `r`: vector of conformal radii (same confidence) [N]
- `y`: keypoint measurements homogenized by 1 [3 x N]
- `b`: 3D canonical keypoint positions [3 x N]
- `camK`: camera calibration matrix [3 x 3]
- `order`: relaxation order to use

# Returns:
- `R`: rotation estimate
- `t`: translation estimate
- `gap`: suboptimality gap
- `status`: SDP solve status
"""
function gaussianpose(y, r, b, camK; silent=true, order=2)
    # objective is scale invariant
    # α_quantile = quantile(Normal(), 1-α)
    # σ = r / α_quantile
    σ = r

    N = size(r,1)
    @polyvar R[1:3,1:3]
    # @polyvar t[1:3]
    vars = vec(R)
    # vars = [vec(R); t]

    # eliminate t
    e3 = [0;0;1]
    U = Array{Any}(undef, N)
    for i = 1:N
        U[i] = (I - y[:,i]*e3')*camK
    end
    H = sum([U[i]'*U[i] / (σ[i]^2) for i = 1:N])
    t = -inv(H)*sum([U[i]'*U[i]*R*b[:,i] / (σ[i]^2) for i = 1:N])

    # objective
    obj = 0.
    for i = 1:N
        term = U[i]*R*b[:,i] + U[i]*t
        obj += term'*term / σ[i]^2
    end

    # constraints
    # expr ≥ 0
    ineq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 
    # expr = 0
    eq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 

    # chirality constraints
    tol = 1e-2
    for i = 1:N
        append!(ineq, [e3'*camK*(R*b[:,i] + t) + tol])
    end

    # SO(3) constraints
    append!(eq, vec(R'*R - I)) # O(3)
    append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
    append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
    append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

    # solve
    pop = [obj; ineq; eq]
    order = order
    opt, sol, data, gap, _ = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=silent, solution=true, refine=false)

    ## Extract solution
    R_est = project2SO3(reshape(sol[1:9],3,3))
    t_est = [ti(vars=>vec(R_est)) for ti in t]

    return R_est, t_est, gap, data.SDP_status
end

function gaussianpose(prob; kwargs...)
    return gaussianpose(prob.y, prob.r, prob.b, prob.camK; kwargs...)
end


"""
    ransagpose(y, r, b, camK; T=1000)

Implements the sampling-based P3P algorithm RANSAG from the paper
"Object Pose Estimation with Statistical Guarantees: Conformal
Keypoint Detection and Geometric Uncertainty Propagation"

# Arguments:
- `r`: vector of conformal radii (same confidence) [N]
- `y`: keypoint measurements homogenized by 1 [3 x N]
- `b`: 3D canonical keypoint positions [3 x N]
- `camK`: camera calibration matrix [3 x 3]
- `T`: number of iterations to search over

# Returns:
- `R`: rotation estimate
- `t`: translation estimate
- `purse_empty`: true if no samples within PURSE found via P3P
"""
function ransagpose(y, r, b, camK; T=1000)
    N = size(r,1)

    # set of feasible poses found
    S_R = []
    S_t = []
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

            if sum(norm.(eachcol(y_p3p - y)) .<= r) == N
                push!(S_R, R)
                push!(S_t, t)
            end
        end
    end
    
    # if empty, fill with random samples ignoring PURSE
    purse_empty = false
    if length(S_R) == 0
        purse_empty = true
        for _ = 1:floor(Int,T/20)
            # perturb from center by (uniformly) random magnitude in random direction
            r_selected = r .* rand(N) # random magnitude
            dir = normalize.(eachcol(randn(2,N))) # random direction
            kpts = y[1:2,:] + reduce(hcat, r_selected .* dir)
            kpts = [kpts; ones(1,N)]
            # run pnp
            R_pnp, t_pnp = gaussianpose(kpts, r, b, camK)
            # R_p3p, t_p3p = p3p(kpts, b[:,idxs], camK)
            for i = axes(t_pnp,2)
                push!(S_R, R_pnp[:,:,i])
                push!(S_t, t_pnp[:,i])
            end
        end
    end

    # if isdefined(Main, :Infiltrator) Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__) end # 🚨 INFILTRATOR 🚨

    # average
    R = project2SO3(sum(S_R))
    t = mean(S_t)

    return R, t, purse_empty
end

ransagpose(prob; kwargs...) = ransagpose(prob.y, prob.r, prob.b, prob.camK; kwargs...)



#################################################################
##
## Functions below kept for reference, not recommended or in use.
## 
#################################################################



"""
    gaussianpose_sdplr(y, r, b, camK; silent=true)

Certifiable PnP with Gaussian noise assumption.

JuMP version with SDPLR optimizer just for fun. Not recommended.

# Arguments:
- `r`: vector of conformal radii (same confidence) [N]
- `y`: keypoint measurements homogenized by 1 [3 x N]
- `b`: 3D canonical keypoint positions [3 x N]
- `camK`: camera calibration matrix [3 x 3]

# Returns:
- `R`: rotation estimate
- `t`: translation estimate
- `gap`: suboptimality gap
- `status`: SDP solve status
"""
function gaussianpose_sdplr(y, r, b, camK; silent=true)
    # objective is scale invariant
    # α_quantile = quantile(Normal(), 1-α)
    # σ = r / α_quantile
    σ = r

    N = size(r,1)
    model = Model(SDPLR.Optimizer)
    if silent
        set_silent(model)
    end
    @variable(model, X[1:10,1:10] ∈ PSDCone()) # vec(R)*vec(R)'

    # eliminate t
    e3 = [0;0;1]
    U = Array{Any}(undef, N)
    for i = 1:N
        U[i] = (I - y[:,i]*e3')*camK
    end
    H = sum([U[i]'*U[i] / (σ[i]^2) for i = 1:N])
    # t = -inv(H)*sum([U[i]'*U[i]*R*b[:,i] / (σ[i]^2) for i = 1:N])
    t_mult = -inv(H)*sum([U[i]'*kron(b[:,i]', U[i]) / (σ[i]^2) for i = 1:N])

    # objective
    A = zeros(9,9)
    for i = 1:N
        kburi = kron(b[:,i]', U[i])

        A += kburi'*kburi / σ[i]^2
        A += (t_mult'*U[i]')*U[i]*t_mult / σ[i]^2
        # cross term
        A += t_mult'*U[i]'*kburi / σ[i]^2
        A += kburi'*U[i]*t_mult / σ[i]^2
        
        # Real value:
        # term = U[i]*R*b[:,i] + U[i]*t_mult*r
        # obj += term'*term / σ[i]^2
    end
    @objective(model, Min, tr(A'*X[1:9,1:9]))

    # constraints
    @constraint(model, X[10,10] == 1)
    q_eqs = SO3_constraints()
    for (i,q) in enumerate(q_eqs)
        Q = Symmetric([q.H[1:9,1:9]  q.c[1:9];  q.c[1:9]'  q.d])
        @constraint(model,  tr(Q*X) == 0.)
    end

    set_attribute(model, "maxrank", (m, n) -> 1)
    optimize!(model)

    if !is_solved_and_feasible(model)
        @warn "Solver did not find an optimal solution!"
    end

    ## Extract solution
    r = value.(X)[1,1:9]
    R_est = project2SO3(reshape(r,3,3))
    t_est = t_mult*vec(R_est)

    return R_est, t_est
end



"""
    R_est, t_est, tight, status = maxmarginpose(y, r, b, camK; lowerb=-1, upperb=10, silent=false)

First order relaxation with l2 form of pose uncertainty set. NOT RECOMMENDED.

Returns pose estimate, optimization status, solution data, JuMP model

# Arguments
- `q_front`: list of quadratics ≤ 0 for chirality constraints
- `q_backproj`: list of quadratics ≤ 0 for backproj constraints
- `q_eqs`: list of quadratics = 0 for SO(3) constraints
## Optional Arguments
- `lowerb=-1`: lower bound the margins
- `upperb=10`: upper bound the margins
- `silent=false`: should we print things?
"""
function maxmarginpose(y, r, b, camK; kwargs...)
    q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
    q_eqs = SO3_constraints()

    return maxmarginpose(q_front, q_backproj, q_eqs; kwargs...)
end

function maxmarginpose(prob; kwargs...)
    # if prob.p == 2
    q_front, q_backproj = uncertaintyset_l2(prob.y, prob.r, prob.b, prob.camK)
    q_eqs = SO3_constraints()
    # else
    #     q_front, q_backproj = uncertaintyset_linf_q(prob.y, prob.r, prob.b, prob.camK)
    #     q_eqs = q_constraints()
    # end
    return maxmarginpose(q_front, q_backproj, q_eqs; kwargs...)
end

function maxmarginpose(q_front, q_backproj, q_eqs; silent=true)
    N = length(q_front)
    dim = length(q_eqs) == 1 ? 8 : 13

    model = Model(Clarabel.Optimizer)
    if silent
        set_silent(model)
    end
    @variable(model, margin[1:N])
    @variable(model, X[1:dim,1:dim] ∈ PSDCone()) # rank 1 of [r, t, 1]
    @constraint(model, X[dim,dim]==1.)

    # objective
    obj = sum(margin)

    # @variable(model, log_margin[1:N])
    # obj = sum(log_margin)
    # for i = 1:N
    #     @constraint(model, [log_margin[i], 1, margin[i]] ∈ MOI.ExponentialCone())
    # end
    @objective(model, Max, obj)

    # constraints
    for (i,q) in enumerate(q_front)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        @constraint(model, tr(Q*X) <= 0.)
    end
    for (i,q) in enumerate(q_backproj)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        if dim == 13
            @constraint(model, tr(Q*X) <= -margin[i])
        else
            @constraint(model, tr(Q*X) <= -margin[floor(Int,(i-1)/4)+1])
        end
    end
    for (i,q) in enumerate(q_eqs)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        @constraint(model,  tr(Q*X) == 0.)
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

    # gap?
    x_round = [vec(R_est); t_est; 1]
    X2 = x_round * x_round'
    margin_round = zeros(N)
    for (i,q) in enumerate(q_backproj)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        margin_round[i] = -tr(Q*X2)
    end

    ub = sum(margin_round)
    gap = abs(ub - opt)/max(1, abs(ub))
    if !silent
        @printf "suboptimality gap of %.2f%%\n" gap*100
    end

    tight = rank(X_val, 1e-4) == 1
    if !tight
        evs = eigvals(X_val)
        if !silent
            @warn "X not rank 1: λ₁ = $(evs[end]), λ₂ = $(evs[end-1])"
        end
    end

    return R_est, t_est, tight, termination_status(model)
end


"""
Compute the "central pose" estimate.

Overrides backproj constraints.
"""
function conformalpose(prob; order=2, silent=false)
    return conformalpose(prob.y, prob.r, prob.b, prob.camK; p=prob.p, order=order, silent=silent)
end

function conformalpose(y, r, b, camK; p=Inf, order=2, silent=false)
    (p == 2 || p == Inf) || error("only accepts `p=2` or `p=Inf`")
    if p == 2
        q_front, _ = uncertaintyset_l2(y, r, b, camK)
        q_eqs = SO3_constraints()
    else
        ## Rotation Version
        # q_front, _ = uncertaintyset_linf_R(y, r, b, camK)
        # q_eqs = SO3_constraints()
        ## Quaternion Version
        q_front, _ = uncertaintyset_linf_q(y, r, b, camK)
        q_eqs = q_constraints()
    end

    N = size(r,1)
    if p == 2
        @polyvar R[1:3,1:3]
    else
        @polyvar R[1:4] # quaternion
    end
    @polyvar t[1:3]
    @polyvar γ[1:N]
    vars = [vec(R); t; γ]

    # Objective
    obj = sum(γ.^2)

    e3 = [0; 0; 1]

    # Constraints
    # expr ≥ 0
    ineq = Vector{TSSOS.Poly{Float64}}()
    # expr = 0
    eq = Vector{TSSOS.Poly{Float64}}()

    # multiplicative margin on backproj constraints
    X = [vec(R); t; 1]*[vec(R); t; 1]'
    for i = 1:N
        iy3 = (I - y[:,i]*e3')

        if p == Inf
            for j = 1:2
                ej = zeros(3); ej[j] = 1
                # TODO: same margin?

                ## Quaternion version
                # positive term
                H = zeros(TSSOS.Poly{Float64},7,7)
                H[1:4, 1:4] = (-Ω1((iy3*camK)'*ej)*Ω2(b[:,i])) - r[i]*γ[i]*(-Ω1(camK'*e3)*Ω2(b[:,i]))
                H += H'
                c = [zeros(4); (ej'*iy3*camK)' - r[i]*γ[i]*(e3'*camK)']
                Q = [H  c;  c'  0.]
                push!(ineq, -tr(Q*X))

                # negative term
                H = zeros(TSSOS.Poly{Float64},7,7)
                H[1:4, 1:4] = -(-Ω1((iy3*camK)'*ej)*Ω2(b[:,i])) - r[i]*γ[i]*(-Ω1(camK'*e3)*Ω2(b[:,i]))
                H += H'
                c = [zeros(4); -(ej'*iy3*camK)' - r[i]*γ[i]*(e3'*camK)']
                Q = [H  c;  c'  0.]
                push!(ineq, -tr(Q*X))
            end
        else # p == 2
            kr_bK = kron(b[:,i]', camK)
            H = zeros(TSSOS.Poly{Float64},12,12)
            H[1:9,1:9] = (iy3*kr_bK)'*(iy3*kr_bK) - γ[i]*r[i]^2*([0 0 1.]*kr_bK)'*([0 0 1.]*kr_bK)
            H[10:12,10:12] = (iy3*camK)'*(iy3*camK) - γ[i]*r[i]^2*([0 0 1.]*camK)'*([0 0 1.]*camK)
            H[10:12,1:9] = (iy3*camK)'*(iy3*kr_bK) - γ[i]*r[i]^2*([0 0 1.]*camK)'*([0 0 1.]*kr_bK)
            H[1:9,10:12] = H[10:12,1:9]'

            Q = [H zeros(12); zeros(13)']
            push!(ineq, -tr(Q*X))
        end
    end

    # no margin on chirality constraints
    slack = 1e-3
    for (i,q) in enumerate(q_front)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        push!(ineq, -tr(Q*X) - slack)
    end

    # no margin on SO(3) constraints
    for (i,q) in enumerate(q_eqs)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        push!(eq, tr(Q*X))
    end

    # bounds on margin
    lowerb = 0.05
    upperb = 10.0
    append!(ineq, γ  .- lowerb) # ≥ 0
    append!(ineq, upperb .- γ) # ≥ 0

    # solve!
    pop = [obj; ineq; eq]
    opt, sol, data, gap = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="block", CS="MD", QUIET=silent, solution=true, refine=true)
    # MOSEK has SLOW PROGRESS


    # Main.@infiltrate

    if data.SDP_status != MOI.OPTIMAL
        @warn "[conformalpose] Returned status $(data.SDP_status). Results may not be lower bound!"
        gap = -1
    end

    R_est = (vars=>sol) .|> R
    if p == Inf
        R_est = quat2rotm(normalize(R_est))
    else
        R_est = project2SO3(R_est)
    end

    t_est = (vars=>sol) .|> t


    return R_est, t_est, gap, data.SDP_status
end


function conformalpose_local(y, r, b, camK; p=Inf, silent=false)
    (p == 2 || p == Inf) || error("only accepts `p=2` or `p=Inf`")
    if p == 2
        q_front, _ = uncertaintyset_l2(y, r, b, camK)
        q_eqs = SO3_constraints()
    else
        ## Rotation Version
        # q_front, _ = uncertaintyset_linf_R(y, r, b, camK)
        # q_eqs = SO3_constraints()
        ## Quaternion Version
        q_front, _ = uncertaintyset_linf_q(y, r, b, camK)
        q_eqs = q_constraints()
    end

    model = Model(Ipopt.Optimizer)

    if silent
        set_silent(model)
    end

    N = size(r,1)
    if p == 2
        @variable(model, R[i=1:3,j=1:3], start = randrotation()[i,j])
    else
        @variable(model, R[i=1:4], start=normalize(randn(4))[i]) # quaternion
    end
    @variable(model, t[1:3])
    @variable(model, γ[1:N])

    # Objective
    @objective(model, Min, sum(γ.^2))

    e3 = [0; 0; 1]

    # Constraints
    # multiplicative margin on backproj constraints
    X = [vec(R); t; 1]*[vec(R); t; 1]'
    for i = 1:N
        iy3 = (I - y[:,i]*e3')

        if p == Inf
            for j = 1:2
                ej = zeros(3); ej[j] = 1
                ## Quaternion version
                # positive term
                H = zeros(AffExpr, 7,7)
                H[1:4, 1:4] = (-Ω1((iy3*camK)'*ej)*Ω2(b[:,i])) - r[i]*γ[i]*(-Ω1(camK'*e3)*Ω2(b[:,i]))
                H += H'
                c = [zeros(4); (ej'*iy3*camK)' - r[i]*γ[i]*(e3'*camK)']
                Q = [H  c;  c'  0.]
                @constraint(model, -tr(Q*X) >= 0)

                # negative term
                H = zeros(AffExpr,7,7)
                H[1:4, 1:4] = -(-Ω1((iy3*camK)'*ej)*Ω2(b[:,i])) - r[i]*γ[i]*(-Ω1(camK'*e3)*Ω2(b[:,i]))
                H += H'
                c = [zeros(4); -(ej'*iy3*camK)' - r[i]*γ[i]*(e3'*camK)']
                Q = [H  c;  c'  0.]
                @constraint(model, -tr(Q*X) >= 0)
            end
        else # p == 2
            kr_bK = kron(b[:,i]', camK)
            H = zeros(AffExpr,12,12)
            H[1:9,1:9] = (iy3*kr_bK)'*(iy3*kr_bK) - γ[i]*r[i]^2*([0 0 1.]*kr_bK)'*([0 0 1.]*kr_bK)
            H[10:12,10:12] = (iy3*camK)'*(iy3*camK) - γ[i]*r[i]^2*([0 0 1.]*camK)'*([0 0 1.]*camK)
            H[10:12,1:9] = (iy3*camK)'*(iy3*kr_bK) - γ[i]*r[i]^2*([0 0 1.]*camK)'*([0 0 1.]*kr_bK)
            H[1:9,10:12] = H[10:12,1:9]'

            Q = [H zeros(12); zeros(13)']
            @constraint(model, -tr(Q*X) >= 0)
        end
    end

    # no margin on chirality constraints
    slack = 1e-3
    for (i,q) in enumerate(q_front)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        @constraint(model, -tr(Q*X) - slack >= 0)
    end

    # no margin on SO(3) constraints
    for (i,q) in enumerate(q_eqs)
        Q = Symmetric([q.H  q.c;  q.c'  q.d])
        @constraint(model, tr(Q*X) == 0)
    end

    # bounds on margin
    lowerb = 0.05
    upperb = 10.0
    @constraint(model, γ  .- lowerb ≥ 0)
    @constraint(model, upperb .- γ ≥ 0)

    # solve!
    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        @warn "[conformalpose] Returned status $(termination_status(model)). Results may not be lower bound!"
        gap = -1
    else
        gap = nothing
    end

    R_est = value.(R)
    if p == Inf
        R_est = quat2rotm(normalize(R_est))
    else
        R_est = project2SO3(R_est)
    end

    t_est = value.(t)


    return R_est, t_est, gap, termination_status(model)
end

function conformalpose_local(prob; silent=false)
    return conformalpose_local(prob.y, prob.r, prob.b, prob.camK; p=prob.p, silent=silent)
end