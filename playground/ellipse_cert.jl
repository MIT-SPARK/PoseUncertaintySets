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

model = Model(Clarabel.Optimizer)
# set_silent(model)
@variable(model, r2)
@variable(model, λ[1:(length(q_backproj) + length(q_front))] .>= 0)
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
@objective(model, Min, r2)

# build and constrain M
# q0 = x'*H0*x + 2(-H0*c)'*x + c'*H0*c <= 1
H0 = diagm(ones(12))
M = -[H0  -H0*center;  (-H0*center)'  center'*H0*center-r2]
for (i_bp,q) in enumerate(q_backproj)
    global M
    i = i_bp
    M += [λ[i]*q.H  λ[i]*q.c;  λ[i]*q.c'  λ[i]*q.d]
end
for (i_fc,q) in enumerate(q_front)
    global M
    i = i_fc + length(q_backproj)
    M += [λ[i]*q.H  λ[i]*q.c;  λ[i]*q.c'  λ[i]*q.d]
end
for (i,q) in enumerate(q_eqs)
    global M
    M += [η[i]*q.H  η[i]*q.c;  η[i]*q.c'  η[i]*q.d]
end
@constraint(model, M >= 0, PSDCone())

# Solve with JuMP
optimize!(model)

println("Solved with status: $(termination_status(model))")
println("Radius: $(value(r2))")

r2_val = value(r2)


println("----------PRIMAL----------")
# JuMP model
model = Model(Clarabel.Optimizer)
@variable(model, X[1:13,1:13] ∈ PSDCone())
set_silent(model)

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



println("----------TSSOS----------")
using TSSOS, DynamicPolynomials

@polyvar R[1:3,1:3]
@polyvar t[1:3]
vars = [vec(R); t]

# objective
obj = -[vars;1]'*W*[vars;1]

# constraints
# expr ≥ 0
ineq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 
# expr = 0
eq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 

for (i,q) in enumerate(q_backproj)
    push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
end
for (i,q) in enumerate(q_front)
    push!(ineq, -[vars;1]'*[q.H  q.c;  q.c'  q.d]*[vars;1])
end

# SO(3) constraints
append!(eq, vec(R'*R - I)) # O(3)
append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

# solve
pop = [obj; ineq; eq]
order = 2
opt, sol, gap, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=false, solution=true, refine=false)



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


### PLOT

using Plots

function generate_ellipse(A, c, num_points=100)
    # Eigenvalue decomposition of A
    λ, V = eigen(Symmetric(A))
        
    # Generate points on unit sphere
    u = LinRange(0, 2π, num_points)
    v = LinRange(0, π, num_points)
    x = zeros(num_points, num_points)
    y = zeros(num_points, num_points)
    z = zeros(num_points, num_points)
    for (i,ui) in enumerate(u)
        # Parametric equations of the unit sphere
        x[:,i] = cos(ui)*sin.(v)
        y[:,i] = sin(ui)*sin.(v)
        z[:,i] = cos.(v)
    end
    sphere = [vec(x) vec(y) vec(z)]' # 3 x num_points^2

    # Transform sphere to ellipsoid
    axes_lengths = 1. ./ sqrt.(λ)
    surf = V * diagm(axes_lengths) * sphere
    surf .+= c
    return surf
end
# plot
Plots.plot([center[10]],[center[11]],[center[12]], seriestype=:scatter, label="center")
surf = generate_ellipse(diagm(ones(3)/-opt), center[10:12], 100)
Plots.scatter!([0],[0],[0],label="camera")
p2 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:])#, msw=0., alpha = 1)
Plots.plot!(label="Ellipse", xlabel="X", ylabel="Y", zlabel="Z", title="Projected")