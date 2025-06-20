## Uncertainty bounds on a single frame
# Lorenzo Shaikewitz, 6/11/2025

using LinearAlgebra
using Printf
using Clarabel, MosekTools
import Plots

using PoseUncertaintySets
using SimpleRotations

object_id = 12
frame = 701

## Load data
keypoint_data, gt = load_keypoint_data(calibrate_l2, α=0.1)
camK = keypoint_data["K"]

r = keypoint_data["r"][frame][object_id]
y = keypoint_data["y"][frame][object_id]
b = keypoint_data["b"][object_id]

# eliminate missing measurements
y = y[1:2,r .>= 0]
y = [y[1:2,:]; ones(size(y,2))']
b = b[:, r .>= 0]
r = r[r .>= 0]

## Pose estimate
# R_est, t_est, purse_empty = ransagpose(r, y, b, camK)
R_est, t_est, gap, SDP_status = gaussianpose(r, y, b, camK; silent=true, order=2)

## S-Lemma
center = [vec(R_est); t_est]
# H, status = bounding_ellipse(center, y, r, b, camK; solver=Mosek.Optimizer, silent=true)
trans_bound, trans_gap, ang_bound, ang_gap, status = purse_bounds(center, y, r, b, camK; order=2, silent=false)
# rad, status2 = bounding_sphere(center, y, r, b, camK; order=1, silent=false)

error("Lorenzo was here")

## Angular Bounds
# Δθs, status_angbounds, gaps = angular_bounds(center, H)


## Refinement Ideas
# marginalize via projection
P = [zeros(3,9) diagm(ones(3))]
H_t = inv(P*inv(H)*P')

# bounding box approach
# H_t = diagm(ones(3))
bounds, gaps, statuses = refine_bbox(center, H, H_t, y, r, b, camK; mode=2, order=1, silent=true)
bounds2, gaps2, statuses2 = refine_bbox(center, H, H_t, y, r, b, camK; mode=3, order=1, silent=true)


## Visualize
surf = ellipse_to_surf(H_t, center[10:12], 100)
# plot ellipse
Plots.plot([center[10]],[center[11]],[center[12]], seriestype=:scatter, label="center")
Plots.scatter!([0],[0],[0],label="camera")
p2 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="s-lemma")#, msw=0., alpha = 1)

# plot PURSE bound
# surf = ellipse_to_surf(diagm(ones(3)) ./ trans_bound, center[10:12], 100)
# Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:])

# plot sphere bound
# surf = ellipse_to_surf(diagm(ones(3)) ./ rad, center[10:12], 100)
# Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:])


plot_bbox!(p2, center, H_t, bounds; label="direct")
plot_bbox!(p2, center, H_t, bounds2; label="ellipse")



# ## Refinement
# l = eigvals(H_t)
# V = eigvecs(H_t)
# l[1] = (V'*center[10:12])[1]
# H_t2 = V*diagm(l)*V'


# sol, status_feas = check_feasibility(center, H_t2, y, r, b, camK; order=1, silent=false)
# surf = ellipse_to_surf(H_t2, center[10:12], 100)
# Plots.plot([center[10]],[center[11]],[center[12]], seriestype=:scatter, label="center")
# Plots.scatter!([0],[0],[0],label="camera")
# Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:])
# p3 = Plots.scatter!([sol[10]],[sol[11]],[sol[12]],label="sample?")