## Get angular bounds?

using TSSOS, DynamicPolynomials

@polyvar R[1:3,1:3]
vars = vec(R)

Rc = reshape(center[1:9],3,3)
cosθ = (tr(R'*Rc) - 1)/2
obj = cosθ # = cos(∠(R,Rc))

# constraints
ineq = zeros(Polynomial{true, Float64}, 0) # expr ≥ 0
# push!(ineq, 1 - (R[:,1] - Rc[:,1])'*H_r1*(R[:,1] - Rc[:,1]))
# push!(ineq, 1 - (R[:,2] - Rc[:,2])'*H_r2*(R[:,2] - Rc[:,2]))
# push!(ineq, 1 - (R[:,3] - Rc[:,3])'*H_r3*(R[:,3] - Rc[:,3]))
push!(ineq, 1 - (vec(R) - vec(Rc))'*H_r*(vec(R) - vec(Rc)))

push!(ineq, cosθ - cos(π/4))

# Add front of camera constraints
# for pt_idx = 1:size(b,2)
#     proj3dto2d = camK*(R*b[:,pt_idx] + t)
#     # front of camera: proj3dto2d[3] >= 0
#     push!(ineq, proj3dto2d[3])
# end

eq = zeros(Polynomial{true, Float64}, 0) # expr == 0
# R ∈ SO(3)
append!(eq, vec(R'*R - I)[[1,2,3,5,6,9]]) # O(3)
append!(eq, vec(R*R' - I)[[2,3,6]]) # O(3)
# append!(eq, R[1:3,3] .- cross(R[1:3,1],R[1:3,2]))
# append!(eq, R[1:3,1] .- cross(R[1:3,2],R[1:3,3]))
# append!(eq, R[1:3,2] .- cross(R[1:3,3],R[1:3,1]))

# TSSOS stuff
pop = [obj; ineq; eq]
order = 1
opt, sol, data = cs_tssos_first(pop, vars, order, numeq=length(eq), TS="MD", QUIET=false, solution=true, LorenzoOverride=true)
sdp_sol,gap,data.flag = TSSOS.approx_sol(opt, data.moment, data.n, data.cliques, data.cql, data.cliquesize, data.supp, data.coe, numeq=data.numeq, tol=data.tol)
sol, refine_status = local_refine_tssos(opt, data; QUIET=true, startpoint=sdp_sol)

println(opt)
roterror(project2SO3(reshape(sol,3,3)), Rc)