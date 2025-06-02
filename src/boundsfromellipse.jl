## Functions to get angular and translation bounds
#  from S-Lemma ellipse
# Lorenzo Shaikewitz, 5/1/2025

using TSSOS, DynamicPolynomials

# TODO: move to module
include("refine.jl")


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
        ineq = zeros(Polynomial{true, Float64}, 0) # expr ≥ 0
        push!(ineq, 1 - (vec(R) - vec(Rc))'*H_r*(vec(R) - vec(Rc)))
        
        # c > 0 forces to be within π/2 of center--this is an assumption
        # but if it does not hold these bounds are the wrong approach anyways
        push!(ineq, c)

        eq = [s^2 + c^2 - 1]

        # Solve with TSSOS
        pop = [obj; ineq; eq]
        opt, sol, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=silent, solution=true, LorenzoOverride=true)
        sdp_sol,gap,data.flag = TSSOS.approx_sol(opt, data.moment, data.n, data.cliques, data.cql, data.cliquesize, data.supp, data.coe, numeq=data.numeq, tol=data.tol)
        sol, refine_status, gap = local_refine_tssos(opt, data; QUIET=silent, startpoint=sdp_sol)

        if !silent
            println("SDP status: $(data.SDP_status)")
            println("Loc status: $(refine_status)")
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