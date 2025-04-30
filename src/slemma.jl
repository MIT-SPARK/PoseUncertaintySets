## Functions to find a bounding ellipse for the set
# Lorenzo Shaikewitz, 4/25/2025

using Printf
using LinearAlgebra
using JuMP, Clarabel, MosekTools

# TODO: move to submodule
include("utils.jl")

"""
    bounding_ellipse(center, y, r, b, K[; solver=Clarabel, silent=false])

S-Lemma to outer bound pose uncertainty set. `(x-center)'*H*(x-center) ≤ 1`

Returns ellipse matrix `H`, optimization status.

# Arguments
- `center`: center of ellipse [vec(R), t]
- `y`: pixel keypoints [3 x N] (homogenized)
- `r`: l2 radii for each keypoint [N]
- `b`: 3D canonical keypoint frame, meters [3 x N]
- `K`: camera calibration matrix [3 x 3]
## Optional Arguments
- `solver=Clarabel.Optimizer`: what solver to use
- `silent=false`: should we print things?
"""
function bounding_ellipse(center, q_front, q_backproj, q_eqs; solver=Clarabel.Optimizer, silent=false)
    q_ineqs = [q_front; q_backproj]

    # JuMP model
    model = Model(solver)
    if silent
        set_silent(model)
    end
    @variable(model, log_det_H0)
    @variable(model, λ[1:length(q_ineqs)] .>= 0)
    @variable(model, η[1:length(q_eqs)])

    # full 12 x 12
    @variable(model, H0[1:12,1:12] ∈ PSDCone())

    # rotation and position independent (breaks tightness)
    # @variable(model, H0_r[1:9,1:9] ∈ PSDCone())
    # @variable(model, H0_p[1:3,1:3] ∈ PSDCone())
    # H0 = [H0_r zeros(9,3); zeros(3,9) H0_p]
    # H0 = [zeros(9,12); zeros(3,9) H0_p]

    # diagonal (breaks tightness)
    # @variable(model, r_H0[1:12] .>= 0)
    # H0 = diagm(r_H0)

    # objective: max logdet(H0)
    # use auxillary variable
    @objective(model, Max, log_det_H0)
    @constraint(model, [log_det_H0; 1; vec(H0)] in MOI.LogDetConeSquare(12))

    # build and constrain M
    # q0 = x'*H0*x + 2(-H0*c)'*x + c'*H0*c <= 1
    M = -[H0  -H0*center;  (-H0*center)'  center'*H0*center-1]
    for (i,q) in enumerate(q_ineqs)
        M += [λ[i]*q.H  λ[i]*q.c;  λ[i]*q.c'  λ[i]*q.d]
    end
    for (i,q) in enumerate(q_eqs)
        M += [η[i]*q.H  η[i]*q.c;  η[i]*q.c'  η[i]*q.d]
    end
    @constraint(model, M >= 0, PSDCone())

    # Solve with JuMP
    optimize!(model)

    if !silent && !is_solved_and_feasible(model)
        printstyled("Solver did not find an optimal solution!\n",color=:red)
    end
    H0_val = value.(H0)

    return (H0_val, termination_status(model))
end