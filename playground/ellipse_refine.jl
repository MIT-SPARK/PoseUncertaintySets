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

model = Model(Mosek.Optimizer)
set_silent(model)
@variable(model, log_det_H0)
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
@objective(model, Max, log_det_H0)
@constraint(model, [log_det_H0; 1; vec(H0)] in MOI.LogDetConeSquare(12))

# build and constrain M
# q0 = x'*H0*x + 2(-H0*c)'*x + c'*H0*c <= 1
M = -[H0  (-H0*center);  (-H0*center)'  center'*H0*center-1]
for (i_bp,q) in enumerate(q_backproj)
    # quadratic only
    global M
    i = i_bp
    M += [λ[i]*q.H  λ[i]*q.c;  λ[i]*q.c'  λ[i]*q.d]
end
for (i_fc,q) in enumerate(q_front)
    # linear only
    global M
    i = i_fc + length(q_backproj)
    M += [λ[i]*q.H  λ[i]*q.c;  λ[i]*q.c'  λ[i]*q.d]
end
for (i,q) in enumerate(q_eqs)
    # quadratic and linear
    global M
    M += [η[i]*q.H  η[i]*q.c;  η[i]*q.c'  η[i]*q.d]
end
@constraint(model, M >= 0, PSDCone())

# Solve with JuMP
optimize!(model)

println("Solved with status: $(termination_status(model))")

H = value.(H0)


## REFINEMENT
println("----------TSSOS----------")

for axis = 1:12
    global H

    V = eigvecs(H)' # H = V'*diagm(λ)*V
    λ = eigvals(H)

    using TSSOS, DynamicPolynomials

    @polyvar R[1:3,1:3]
    @polyvar t[1:3]
    vars = [vec(R); t]
    varsV = V*vars
    centerV = V*center

    # objective
    obj = -(varsV[axis] - centerV[axis])^2

    # constraints
    # expr ≥ 0
    ineq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 
    # expr = 0
    eq = zeros(Polynomial{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder}, Float64}, 0) 

    # pretty sure these are equivalent to not doing the V thing
    for (i,q) in enumerate(q_backproj)
        push!(ineq, -[varsV;1]'*[V*q.H*V'  V*q.c;  (V*q.c)'  q.d]*[varsV;1])
    end
    for (i,q) in enumerate(q_front)
        push!(ineq, -[varsV;1]'*[V*q.H*V'  V*q.c;  (V*q.c)'  q.d]*[varsV;1])
    end

    # SO(3) constraints
    for (i,q) in enumerate(q_eqs)
        push!(eq, [varsV;1]'*[V*q.H*V'  V*q.c;  (V*q.c)'  q.d]*[varsV;1])
    end
    # append!(eq, vec(R'*R - I)) # O(3)
    # append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
    # append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
    # append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

    # solve
    pop = [obj; ineq; eq]
    order = 1
    opt, sol, gap, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=false, solution=true, refine=false)

    γ = -obj(vars=>sol)
    λ[axis] = (length(λ)-2)/γ
    H = V'*diagm(λ)*V
end



### PLOT

P = [zeros(3,9) diagm(ones(3))]
H_t = inv(P*inv(H)*P')

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
# surf = generate_ellipse(diagm(ones(3)/-opt), center[10:12], 100)
surf = generate_ellipse(H_t, center[10:12], 100)
Plots.scatter!([0],[0],[0],label="camera")
p2 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:])#, msw=0., alpha = 1)
Plots.plot!(label="Ellipse", xlabel="X", ylabel="Y", zlabel="Z", title="Projected")