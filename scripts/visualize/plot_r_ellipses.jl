## Plot 1st & 2nd order ellipses (rotation)
#
# Lorenzo Shaikewitz, 8/5/2025

using LinearAlgebra
import Plots
using PoseUncertaintySets
using SimpleRotations

# settings used for figure
object_id = 9
frame = 352
α = 0.1
p = Inf

# load data and calibrate
keypoint_data, gt = load_keypoint_data(calibrate_lp, p=p, α=α)
prob = get_problem(keypoint_data, object_id, frame)

# get pose estimate and ellipse
R_est, t_est, gap_pose, status_pose = gaussianpose(prob; silent=true, order=2)
q_est = rotm2quat(R_est)
# R ellipse order 1
H_o1, gap_o1, status_o1 = bounding_ellipse([vec(R_est); t_est], prob; silent=true, order=1)
P1 = [diagm(ones(9)) zeros(9,3)]
Hr_o1 = kron(R_est',diagm(ones(3)))'*inv(P1*pinv(H_o1)*P1')*kron(R_est',diagm(ones(3)))
Pθ = zeros(3,9)
Pθ[1,6] = 1; Pθ[1,8] = -1; Pθ[2,7] = 1; Pθ[2,3] = -1; Pθ[3,2] = 1; Pθ[3,4] = -1
Hθ_o1 = 1/2*inv(Pθ*pinv(Hr_o1)*Pθ')
# R ellipse order 2
H_o2, gap_o2, status_o2 = bounding_ellipse([vec(R_est); t_est], prob; silent=false, order=2)
Hr_o2 = kron(R_est',diagm(ones(3)))'*inv(P1*pinv(H_o2)*P1')*kron(R_est',diagm(ones(3)))
Hθ_o2 = 1/2*inv(Pθ*pinv(Hr_o2)*Pθ')
# q ellipse order 2
H_o2, gap_o2, status_o2 = bounding_ellipse_quat([q_est; t_est], prob; silent=true, order=2)
P2 = [diagm(ones(4)) zeros(4,3)]
Hr_o2 = Ω2(q_est)'*inv(P2*pinv(H_o2)*P2')*Ω2(q_est)
Pθ = [zeros(3,1) diagm(ones(3))]
Hθq_o2 = inv(Pθ*pinv(Hr_o2)*Pθ')
# TODO

# get samples
S_R, S_t = sample_set(prob; method="ransag", T=1000)
# R = R₀*R̄ ⟹ R₀ = R*R̄'
axangs = rotm2axang.([R*R_est' for R in S_R])
ωsinθ = reduce(hcat,[ω*sin(θ) for (ω, θ) in axangs])
ωsinθ2 = reduce(hcat,[ω*sin(θ/2) for (ω, θ) in axangs])

## plot!
Plots.plotlyjs()
Plots.scatter([0],[0],[0],label="center")

# order 1 ellipse
surf = ellipse_to_surf(Hθ_o1, zeros(3), 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="order 1")
surf = ellipse_to_surf(diagm(ones(3)), zeros(3), 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="unit")
surf = ellipse_to_surf(Hθ_o2, zeros(3), 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="order 2")
# samples
plot_R = Plots.scatter3d!(eachrow(ωsinθ)..., label="samples")

## Quaternion plot
# order 2 ellipse
Plots.scatter([0],[0],[0],label="center")
surf = ellipse_to_surf(Hθq_o2, zeros(3), 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="order 2")
# surf = ellipse_to_surf(diagm(ones(3)), zeros(3), 100)
# Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="unit")
# samples
plot_q = Plots.scatter3d!(eachrow(ωsinθ2)..., label="samples")