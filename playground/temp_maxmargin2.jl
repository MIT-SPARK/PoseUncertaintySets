


include("../src/uncertaintyset.jl")
include("../src/centralpose.jl")

# Parameters
object_id = 11

analytic = false
silent = false

# Load data
cadpath = "./data/lmo/models_eval/"
datapath = "./data/lmo/l2_04_real.dat"
all_data = deserialize(datapath)
camK = all_data["camK"]
img_names = all_data["img"]
num_frames = length(img_names)


frame = 28

r = all_data["radii"][frame][object_id] .+ 1e-3 # make sure 0 radii doesn't happen
y = all_data["pixel_measurements"][frame][object_id]
b = all_data["canonical_kpts"][frame][object_id]

# build uncertainty set
q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
q_eqs = SO3_constraints()

# r .*= 0.1

lowerb = -10*r
upperb = 10*r

# ----------------
N = length(q_front)
model = Model(Clarabel.Optimizer)
if silent
    set_silent(model)
end
@variable(model, margin[1:N])
@variable(model, X[1:13,1:13] ∈ PSDCone()) # rank 1 of [r, t, 1]
@constraint(model, X[13,13]==1.)

# objective
obj = sum(margin)
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
@constraint(model, margin .>= lowerb)
@constraint(model, margin .<= upperb)

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
sdp_status = termination_status(model)

# time321=@timed begin
#     start = sdp_sol_rounded

#     # local solver
#     model = Model(NLPModelsJuMP.Optimizer)
#     set_attribute(model, "solver", Percival.PercivalSolver)
#     margin = start[1:end-3-9]
#     R_est = reshape(start[end-3-9+1:end-3],3,3)
#     t_est = start[end-2:end]
#     N = size(margin,1)

#     @variable(model, margin_local[i=1:N], start=margin[i])
#     @variable(model, R[i=1:3,j=1:3], start=R_est[i,j])
#     @variable(model, t[i=1:3], start=t_est[i])

#     @objective(model, Max, sum(margin_local))

#     # constraints
#     for i = 1:N
#         proj3dto2d = camK*(R*b[:,i] + t)
#         # front of camera
#         @constraint(model, proj3dto2d[3] >= 0)
#         # PURSE
#         residual = (I - y[:,i]*[0;0;1]')*proj3dto2d
#         # 2-norm
#         @constraint(model, (r[i])^2*(proj3dto2d[3])^2 - residual'*residual - margin_local[i] >= 0)
#     end
#     @constraint(model, vec(R'*R - I) .== 0)
#     @constraint(model, R[1:3,3] .== cross(R[1:3,1],R[1:3,2]))
#     @constraint(model, R[1:3,1] .== cross(R[1:3,2],R[1:3,3]))
#     @constraint(model, R[1:3,2] .== cross(R[1:3,3],R[1:3,1]))

#     # bounds
#     if !isnothing(lowerb)
#         @constraint(model, margin_local .>= lowerb)
#     end
#     @constraint(model, margin_local .<= upperb)

#     optimize!(model)

#     refine_status = termination_status(model)
#     sol = [value.(margin_local); vec(value.(R)); value.(t)]
# end


p, d, v, g = (R_est, t_est), sdp_status, vars_proj, gap
# ------------------------


gt = all_data["gt_poses"][frame][5]
R_gt = project2SO3(gt[1])
t_gt = gt[2] / 1000. # [m]
R_est = p[1]
t_est = p[2]
err_R = roterror(R_gt, R_est)
err_t = norm(t_est - t_gt)*1000
@printf "R error: %.2f deg\n" err_R
@printf "t error: %.2f mm\n" err_t