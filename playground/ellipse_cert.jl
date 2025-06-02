## Get bounding ellipse via S-Lemma
# Lorenzo Shaikewitz 4/29/2025

using Serialization
using Printf
using LinearAlgebra
using JuMP, Clarabel, MosekTools

# TODO: move to module
include("../src/uncertaintyset.jl")
include("../src/maxmargin.jl")
include("../src/slemma.jl")

# Parameters
object_id = 9

analytic = false
lowerb = -1
upperb = 10
silent = true

# Load data
cadpath = "./data/lmo/models_eval/"
datapath = "./data/lmo/l2_01.dat"
all_data = deserialize(datapath)
camK = all_data["camK"]
img_names = all_data["img"]
num_frames = length(img_names)

frame = 1

r = all_data["radii"][frame][object_id]
y = all_data["pixel_measurements"][frame][object_id]
b = all_data["canonical_kpts"][frame][object_id]

# build uncertainty set
q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
q_eqs = SO3_constraints()

# find center
println("----------MAX MARGIN----------")
sdp_pose, status, data, model = center_l2(q_front, q_backproj, q_eqs; analytic=analytic, lowerb=lowerb, upperb=upperb*r, silent=silent)

# local refinement + gap
tight = data[1]
vars_proj = data[3]
if tight
    gap = 0.
    est_pose = sdp_pose
else
    est_pose, loc_status, vars_proj, gap = local_refine(q_front, q_backproj, q_eqs, data; analytic=analytic, lowerb=lowerb, upperb=upperb*r, silent=silent)
end

# check feasibility
feas = check_feas(q_front, q_backproj, q_eqs, vars_proj; tol=1e-3, silent=silent)

println("Solved with status: $status")

if !feas
    println("Failed to find feasible solution!")
end

println("----------S LEMMA----------")
center = [vec(est_pose[1]); est_pose[2]]

println("----------PRIMAL----------")
# JuMP model
model = Model(Mosek.Optimizer)
@variable(model, X[1:13,1:13] ∈ PSDCone())

# objective
W = [I -center; -center' center'*center]
obj = tr(W'*X)
@objective(model, Max, obj)

# constraints
for (i,q) in enumerate(q_backproj)
    @constraint(model, tr([q.H  q.c;  q.c'  q.d]'*X) <= 0)
end
for (i,q) in enumerate(q_front)
    @constraint(model, tr([q.H  q.c;  q.c'  q.d]'*X) <= 0)
end
for (i,q) in enumerate(q_eqs)
    @constraint(model, tr([q.H  q.c;  q.c'  q.d]'*X) == 0)
end
@constraint(model, tr(X) == 13)
# @constraint(model, X[13,13] == 1)

# tightening constraints?
# @constraint(model, tr([diagm(ones(12)) zeros(12); zeros(12)' -3.5]'*X) <= 0)

# Solve with JuMP
optimize!(model)

if !silent && !is_solved_and_feasible(model)
    printstyled("Solver did not find an optimal solution!\n",color=:red)
end

println("Solved with status: $(termination_status(model))")
println("Primal value: $(value(obj))")


# println("----------DUAL----------")
# # JuMP model
# model = Model(Mosek.Optimizer)
# @variable(model, η)
# @variable(model, λ[1:(length(q_backproj) + length(q_front))] .>= 0)
# @variable(model, μ[1:length(q_eqs)])

# # objective
# n = 13
# obj = n*η
# @objective(model, Min, obj)

# # constraints
# W = [I -center; -center' center'*center]
# Q = W - η*diagm(ones(n))
# for (i_bp,q) in enumerate(q_backproj)
#     global Q
#     i = i_bp
#     Q -= λ[i]*[q.H  q.c;  q.c'  q.d]
# end
# for (i_fr,q) in enumerate(q_front)
#     global Q
#     i = i_fr + length(q_backproj)
#     Q -= λ[i]*[q.H  q.c;  q.c'  q.d]
# end
# for (i,q) in enumerate(q_eqs)
#     global Q
#     Q -= μ[i]*[q.H  q.c;  q.c'  q.d]
# end
# @constraint(model, -Q >= 0, PSDCone())

# # Solve with JuMP
# optimize!(model)

# if !silent && !is_solved_and_feasible(model)
#     printstyled("Solver did not find an optimal solution!\n",color=:red)
# end

# println("Solved with status: $(termination_status(model))")
# println("Dual value: $(value(obj))")