## Functions for pose estimation from keypoints
# Lorenzo Shaikewitz, 6/4/2025

"""
    gaussianpose(r, y, b, camK; silent=true, order=2)

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
function gaussianpose(r, y, b, camK; silent=true, order=2)
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

"""
    ransagpose(r, y, b, camK; T=1000)

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
function ransagpose(r, y, b, camK; T=1000)
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
            idxs = sample(1:N, 3, replace=false)
            kpts = y[:,idxs]
            # run p3p
            R_p3p, t_p3p = p3p(kpts, b[:,idxs], camK)
            for i = axes(t_p3p,2)
                push!(S_R, R_p3p[:,:,i])
                push!(S_t, t_p3p[:,i])
            end
        end
    end

    # if isdefined(Main, :Infiltrator) Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__) end # 🚨 INFILTRATOR 🚨

    # average
    R = project2SO3(sum(S_R))
    t = mean(S_t) # TODO: check

    return R, t, purse_empty
end


"""
    gaussianpose_sdplr(r, y, b, camK; silent=true)

Certifiable PnP with Gaussian noise assumption.

JuMP version just for fun. Not recommended.

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
function gaussianpose_sdplr(r, y, b, camK; silent=true)
    # objective is scale invariant
    # α_quantile = quantile(Normal(), 1-α)
    # σ = r / α_quantile
    σ = r

    N = size(r,1)
    model = Model(SDPLR.Optimizer)
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