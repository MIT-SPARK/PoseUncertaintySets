## Get angular bounds?

using TSSOS, DynamicPolynomials

include("../src/refine.jl")

@polyvar t[1:3]
vars = t

obj = -t[3]

# constraints
ineq = zeros(Polynomial{true, Float64}, 0) # expr ≥ 0
push!(ineq, 1 - (vars - center[10:12])'*H_t*(vars - center[10:12]))

# Add front of camera constraints
# This will make bounds asymmetric
# r = all_data["radii"][frame][object_id]
# y = all_data["pixel_measurements"][frame][object_id]
# b = all_data["canonical_kpts"][frame][object_id]
# for pt_idx = 1:size(b,2)
#     proj3dto2d = camK*(R*b[:,pt_idx] + t)
#     # front of camera: proj3dto2d[3] >= 0
#     push!(ineq, proj3dto2d[3])
# end

eq = zeros(Polynomial{true, Float64}, 0) # expr == 0
# R ∈ SO(3)
# append!(eq, vec(R'*R - I)) # O(3)
# append!(eq, vec(R*R' - I)) # O(3) (redundant)
# append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
# append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
# append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

# TSSOS stuff
pop = [obj; ineq; eq]
order = 1
opt, sol, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=false, solution=true, LorenzoOverride=true)
sdp_sol,gap,data.flag = TSSOS.approx_sol(opt, data.moment, data.n, data.cliques, data.cql, data.cliquesize, data.supp, data.coe, numeq=data.numeq, tol=data.tol)
sol, refine_status = local_refine_tssos(opt, data; QUIET=true, startpoint=sdp_sol)


println(sol - center[10:12])
# R_est = reshape([r(vars=>sol) for r in vec(R)],3,3)
# roterror(R_est*Rc, Rc)