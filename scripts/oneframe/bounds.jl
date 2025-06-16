## Uncertainty bounds on a single frame
# Lorenzo Shaikewitz, 6/11/2025

using LinearAlgebra
using Printf
using Clarabel, MosekTools
import Plots

using PoseUncertaintySets
using SimpleRotations

object_id = 9
frame = 10

## Load data
keypoint_data, gt = load_keypoint_data(calibrate_l2)
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
H, status = bounding_ellipse(center, y, r, b, camK; solver=Clarabel.Optimizer, silent=false)
trans_bound, trans_gap, ang_bound, ang_gap = purse_bounds(center, y, r, b, camK; order=2, silent=false)

r, status2 = bounding_sphere(center, y, r, b, camK; order=2, silent=false)

## Angular Bounds
# Δθs, status_angbounds, gaps = angular_bounds(center, H)

## Visualize

P = [zeros(3,9) diagm(ones(3))]
H_t = inv(P*inv(H)*P')

surf = ellipse_to_surf(H_t, center[10:12], 100)
# plot ellipse
Plots.plot([center[10]],[center[11]],[center[12]], seriestype=:scatter, label="center")
Plots.scatter!([0],[0],[0],label="camera")
p2 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:])#, msw=0., alpha = 1)

# plot PURSE bound
surf = ellipse_to_surf(diagm(ones(3)) ./ trans_bound, center[10:12], 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:])

# plot PURSE bound
surf = ellipse_to_surf(diagm(ones(3)) ./ r, center[10:12], 100)
Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:])
