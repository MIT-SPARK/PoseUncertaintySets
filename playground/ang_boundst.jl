## Get angular bounds--version without marginalized t

using TSSOS, DynamicPolynomials

include("../src/refine.jl")

@polyvar s
@polyvar c
@polyvar t[1:3]
vars = [s; c; t]

Rx = [1 0 0; 0 c -s; 0 s c]
Ry = [c 0 s; 0 1 0; -s 0 c]
Rz = [c s 0; -s c 0; 0 0 1]

R = Rz*Rc

Rc = reshape(center[1:9],3,3)
obj = c

# constraints
ineq = zeros(Polynomial{true, Float64}, 0) # expr ≥ 0
push!(ineq, 1 - ([vec(R); t] - center)'*H*([vec(R); t]- center))

r = all_data["radii"][frame][object_id]
y = all_data["pixel_measurements"][frame][object_id]
b = all_data["canonical_kpts"][frame][object_id]
for pt_idx = 1:size(b,2)
    proj3dto2d = camK*(R*b[:,pt_idx] + t)
    # front of camera: proj3dto2d[3] >= 0
    push!(ineq, proj3dto2d[3])
end

# push!(ineq, -s)
push!(ineq, c)
# push!(ineq, sin(π/4) - s)

eq = zeros(Polynomial{true, Float64}, 0) # expr == 0
# R ∈ SO(3)
push!(eq, s^2 + c^2 - 1)

# TSSOS stuff
pop = [obj; ineq; eq]
order = 2
opt, sol, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=false, solution=true, LorenzoOverride=true)
sdp_sol,gap,data.flag = TSSOS.approx_sol(opt, data.moment, data.n, data.cliques, data.cql, data.cliquesize, data.supp, data.coe, numeq=data.numeq, tol=data.tol)
sol, refine_status = local_refine_tssos(opt, data; QUIET=true, startpoint=sdp_sol)


println(opt)
R_est = reshape([r(vars=>sol) for r in vec(R)],3,3)
roterror(R_est, Rc)