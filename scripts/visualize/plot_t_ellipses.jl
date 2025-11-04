## Plot 1st & 2nd order ellipses (translation)
#
# Lorenzo Shaikewitz, 8/1/2025

using LinearAlgebra
using Statistics
import Plots
using PoseUncertaintySets
using SimpleRotations

# settings used for figure
dataset = "lmo"
object_id = 9
frame = 352
α = 0.1
p = Inf

# load data and calibrate
keypoint_data, gt = load_keypoint_data(calibrate_lp, dataset, p=p, α=α)
prob = get_problem(keypoint_data, object_id, frame)

# get pose estimate and ellipse
R_est, t_est, gap_pose, status_pose = gaussianpose(prob; silent=true, order=2)
H_o1, gap_o1, status_o1 = bounding_ellipse([vec(R_est); t_est], prob; silent=true, order=1)
P1 = [zeros(3,9) diagm(ones(3))]
Ht_o1 = inv(P1*pinv(H_o1)*P1')
H_o2, gap_o2, status_o2 = bounding_ellipse_quat([rotm2quat(R_est); t_est], prob; silent=true, order=2)
P2 = [zeros(3,4) diagm(ones(3))]
Ht_o2 = inv(P2*pinv(H_o2)*P2')
# takes a while
# H_o3, gap_o3, status_o3 = bounding_ellipse_quat([rotm2quat(R_est); t_est], prob; silent=true, order=3)
# Ht_o3 = inv(P2*pinv(H_o3)*P2')

# RANSAG
trans_bound1, trans_gap1, ang_bound1, ang_gap1, status1 = purse_bounds([vec(R_est); t_est], prob.y, prob.r, prob.b, prob.camK; order=1, silent=true)
trans_bound, trans_gap, ang_bound, ang_gap, status = purse_bounds([vec(R_est); t_est], prob.y, prob.r, prob.b, prob.camK; order=2, silent=true)

# get samples
S_R, S_t = sample_set(prob; method="ransag", T=1000)
S_t = reduce(hcat, S_t)
# S_R, S_t = sample_set(prob; method="grid", Ht=Ht_o2)

## plot!
ms_plot = 0.5

Plots.gr()
surfs = zeros(3,0)
# RANSAG order 1
surf = ellipse_to_surf(diagm(ones(3))/trans_bound1.^2, t_est, 100)
Plots.scatter3d(surf[1,:], surf[2,:], surf[3,:], label="RANSAG (order 1)", msw=0., c=5, ms=ms_plot)
surfs = hcat(surfs, surf)
# order 1 ellipse
surf = ellipse_to_surf(Ht_o1, t_est, 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="order 1", msw=0.,c=4, ms=ms_plot)
surfs = hcat(surfs, surf)
# RANSAG order 2
surf = ellipse_to_surf(diagm(ones(3))/trans_bound.^2, t_est, 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="RANSAG (order 2)", msw=0., c=1, ms=ms_plot)
surfs = hcat(surfs, surf)
# order 2 ellipse
surf = ellipse_to_surf(Ht_o2, t_est, 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="order 2", msw=0.,c=3, ms=ms_plot)
surfs = hcat(surfs, surf)
# samples
Plots.scatter3d!(eachrow(S_t)..., label="samples", c="royalblue4", msw=0., ms=ms_plot)
# center and camera
Plots.plot!([t_est[1]], [t_est[2]], [t_est[3]], seriestype=:scatter, label="center",c=1, ms=2)
plot_static = Plots.scatter!([0],[0],[0],label="camera",c="grey", xlabel="x",ylabel="y",zlabel="z")
# Plots.savefig(plot_static, "tellipse_10.svg")

x12, y12, z12 = extrema(surfs[1,:]), extrema(surfs[2,:]), extrema(surfs[3,:])
d = maximum([diff([x12...]),diff([y12...]),diff([z12...])])[1] / 2
xm, ym, zm = mean(x12),  mean(y12),  mean(z12)
Plots.plot!(xlims=(xm-d,xm+d), ylims=(ym-d,ym+d), zlims=(zm-d,zm+d), aspect_ratio=1)
Plots.plot!(camera=(50,20))



Plots.plotlyjs()
Plots.plot([t_est[1]], [t_est[2]], [t_est[3]], seriestype=:scatter, label="center")
Plots.scatter!([0],[0],[0],label="camera")

# order 1 ellipse
surf = ellipse_to_surf(Ht_o1, t_est, 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="order 1")
# order 2 ellipse
surf = ellipse_to_surf(Ht_o2, t_est, 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="order 2")
# order 3 ellipse
# surf = ellipse_to_surf(Ht_o3, t_est, 100)
# Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="order 3")
# RANSAG
surf = ellipse_to_surf(diagm(ones(3))/trans_bound.^2, t_est, 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="RANSAG")
# samples
plot_interactive = Plots.scatter3d!(eachrow(S_t)..., label="samples")
Plots.plot!(xlim=[t_est[1]-2.5, t_est[1]+2.5], ylim=[t_est[2]-2.5,t_est[2]+2.5], zlim=[t_est[3]-2.5,t_est[3]+2.5])