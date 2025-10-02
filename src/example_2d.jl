## A simple 2D ellipse bounding problem
#
# Lorenzo Shaikewitz, 9/29/2025

function bound_2d(center, As, Bs=[]; order=1, silent=false)
    (order == 1) || return bound_2d_higherorder(center, As, Bs; order=order, silent=silent) #error("Not implemented yet!")

    # JuMP model
    model = Model(Mosek.Optimizer)
    if silent
        set_silent(model)
    end
    @variable(model, log_det_H)
    @variable(model, λ[1:length(As)] .>= 0)
    if length(Bs) > 0
        @variable(model, η[1:length(Bs)])
    end
    @variable(model, H[1:2,1:2] ∈ PSDCone())

    # objective
    @objective(model, Max, log_det_H)
    @constraint(model, [log_det_H; 1; triangle_vec(H)] ∈ MOI.LogDetConeTriangle(2))

    # build M
    M = -[center'*H*center-1  (-H*center)'; -H*center  H]
    for (i,A) in enumerate(As)
        M += λ[i]*A
    end
    for (i,B) in enumerate(Bs)
        M += η[i]*B
    end
    @constraint(model, psdcon, M >= 0, PSDCone())

    # solve!
    optimize!(model)

    # display(solution_summary(model))
    # printstyled("\nEigvals\n",underline=true)
    # display(["Dual"  "Dual Dual"; eigvals(value.(M))  eigvals(dual(model[:psdcon]))])

    return value.(H), termination_status(model), value.(M), dual(model[:psdcon])
end

function bound_2d_higherorder(center, As, Bs; order=2, silent=false)
    @polyvar vars[1:2]

    # use proxy objective
    W = [1 -ones(2)'; -ones(2) ones(2,2)]
    obj = -[1;vars]'*W*[1;vars]

    # constraints
    # expr ≥ 0
    ineq = Vector{TSSOS.Poly{Float64}}()
    # expr = 0
    eq = Vector{TSSOS.Poly{Float64}}()

    # As
    for A in As
        push!(ineq, -[1;vars]'*A*[1;vars])
    end
    # Bs
    for B in Bs
        push!(eq, [1;vars]'*B*[1;vars])
    end

    # use TSSOS to generate redundant constraints
    pop = [obj; ineq; eq]
    order = order
    opt, sol, data, gap, model = cs_tssos_first(pop, vars, order, numeq=length(eq), TS=false, CS=false, QUIET=silent, solve=false, solution=false, MomentOne=true)

    if silent
        set_silent(model)
    end

    ## Modify model
    # add shape variable `H` (density must match `Ĥ`)
    @variable(model, H[1:2,1:2] ∈ PSDCone())
    shape_mat = -[(center'*H*center - 1)  center'*H; H*center H] # with -1
    shapeΔ = triangle_vec(shape_mat)
    # update objective to logdet
    @variable(model, logdet_H)
    @objective(model, Max, logdet_H)
    @constraint(model, [logdet_H; 1; triangle_vec(H)] ∈ MOI.LogDetConeTriangle(2))
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
    set_optimizer(model, Mosek.Optimizer)
    optimize!(model)

    # extract dual
    c = [all_constraints(model, t...) for t in list_of_constraint_types(model)]
    M = c[2][1]
    X = dual(c[2][1])

    # temp
    # display(solution_summary(model))
    # printstyled("\nEigvals\n",underline=true)
    # display(["Dual"  "Dual Dual"; eigvals(value.(M))  eigvals(X)])

    return value.(H), termination_status(model), value.(M), X
end


function bound_2d_dual(center, As, Bs=[]; order=1, silent=false)
    (order == 1) || return bound_2d_higherorder(center, As, Bs; order=order, silent=silent) #error("Not implemented yet!")

    # JuMP model
    model = Model(Mosek.Optimizer)
    if silent
        set_silent(model)
    end
    @variable(model, log_det_S)
    @variable(model, S[1:2,1:2] ∈ PSDCone())
    @variable(model, X[1:3,1:3] ∈ PSDCone())

    # objective
    @objective(model, Min, X[1,1] - log_det_S - 2)
    @constraint(model, [log_det_S; 1; triangle_vec(S)] ∈ MOI.LogDetConeTriangle(2))

    # S constraint
    Y = X[2:3,2:3]; y = X[1,2:3]
    @constraint(model, conS, S == Y - (y*center' + center*y') + X[1,1]*center*center')

    # set constraints
    for A in As
        @constraint(model, tr(A'*X) <= 0)
    end
    for B in Bs
        @constraint(model, tr(B'*X) == 0)
    end

    # solve!
    optimize!(model)

    return -dual(model[:conS]), termination_status(model), dual(VariableInSetRef(X)), value.(X)
end