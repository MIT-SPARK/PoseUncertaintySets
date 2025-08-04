## Plot 1st & 2nd order ellipses (translation)
#
# Lorenzo Shaikewitz, 8/1/2025

using LinearAlgebra
import Plots
using PoseUncertaintySets
using SimpleRotations

Plots.plotlyjs()

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
H_o1, gap_o1, status_o1 = bounding_ellipse([vec(R_est); t_est], prob; silent=true, order=1)
P1 = [zeros(3,9) diagm(ones(3))]
Ht_o1 = inv(P1*pinv(H_o1)*P1')
H_o2, gap_o2, status_o2 = bounding_ellipse_quat([rotm2quat(R_est); t_est], prob; silent=true, order=2)
P2 = [zeros(3,4) diagm(ones(3))]
Ht_o2 = inv(P2*pinv(H_o2)*P2')

# get samples
S_R, S_t = sample_set(prob; method="ransag", T=1000)
S_t = reduce(hcat, S_t)
# S_R, S_t = sample_set(prob; method="grid", Ht=Ht_o2)

## plot!
Plots.plot([t_est[1]], [t_est[2]], [t_est[3]], seriestype=:scatter, label="center")
Plots.scatter!([0],[0],[0],label="camera")

# order 1 ellipse
surf = ellipse_to_surf(Ht_o1, t_est, 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="order 1")
# order 2 ellipse
surf = ellipse_to_surf(Ht_o2, t_est, 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="order 2")
# samples
Plots.scatter3d!(eachrow(S_t)..., label="samples")