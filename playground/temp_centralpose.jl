


include("../src/uncertaintyset.jl")
include("../src/centralpose.jl")

# Parameters
object_id = 9

analytic = false
silent = false

# Load data
cadpath = "./data/lmo/models_eval/"
datapath = "./data/lmo/l2_04_real.dat"
all_data = deserialize(datapath)
camK = all_data["camK"]
img_names = all_data["img"]
num_frames = length(img_names)


frame = 1

r = all_data["radii"][frame][object_id] .+ 1e-3 # make sure 0 radii doesn't happen
y = all_data["pixel_measurements"][frame][object_id]
b = all_data["canonical_kpts"][frame][object_id]

# build uncertainty set
q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
q_eqs = SO3_constraints()


lowerb = -0.8*r
upperb = 0.8*r

# ----------------
N = length(q_front)
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

time123=@timed sol, refine_status = local_refine_tssos(opt, data; QUIET=silent, startpoint=sdp_sol_rounded)

time321=@timed begin
    start = sdp_sol_rounded

    # local solver (gurobi?)
    model = Model(Ipopt.Optimizer)
    margin = start[1:end-3-9]
    R_est = reshape(start[end-3-9+1:end-3],3,3)
    t_est = start[end-2:end]
    N = size(margin,1)

    @variable(model, margin_local[i=1:N], start=margin[i])
    @variable(model, R[i=1:3,j=1:3], start=R_est[i,j])
    @variable(model, t[i=1:3], start=t_est[i])

    @objective(model, Max, sum(margin_local))

    # constraints
    for i = 1:N
        proj3dto2d = camK*(R*b[:,i] + t)
        # front of camera
        @constraint(model, proj3dto2d[3] >= 0)
        # PURSE
        residual = (I - y[:,i]*[0;0;1]')*proj3dto2d
        # 2-norm
        @constraint(model, (r[i] - margin_local[i])^2*(proj3dto2d[3])^2 - residual'*residual >= 0)
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
end




if !silent
    println("SDP status: $(data.SDP_status)")
    println("Loc status: $(refine_status)")
end

R_est = project2SO3(reshape(sol[end-3-9+1:end-3],3,3))
t_est = sol[end-2:end]

vars_proj = [sol[1:end-3-9]; vec(R_est); t_est]

p, d, v, g = (R_est, t_est), data.SDP_status, vars_proj, gap
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