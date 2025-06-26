## Uncertainty bounds on a single frame
# Lorenzo Shaikewitz, 6/11/2025

using LinearAlgebra
using Printf
using Clarabel, MosekTools
import Plots

using PoseUncertaintySets
using SimpleRotations

object_id = 5
frame = 13

## Load data
keypoint_data, gt = load_keypoint_data(calibrate_lp, p=Inf, α=0.1)
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
# R_est, t_est, purse_empty = ransagpose(y, r, b, camK)
R_est, t_est, gap, SDP_status = gaussianpose(y, r, b, camK; silent=true, order=2)

## S-Lemma
center = [rotm2quat(R_est); t_est]
# center = [vec(R_est); t_est]
# Hinf, status = bounding_ellipse(center, y, r, b, camK; p=Inf, solver=Mosek.Optimizer, silent=false)
rad, status2 = bounding_sphere(center, y, r, b, camK; p=Inf, order=2, silent=false)
# trans_bound, trans_gap, ang_bound, ang_gap, status = purse_bounds(center, y, r, b, camK; p=Inf, order=2, silent=false)
# center = [vec(R_est); t_est]
# H2, status2 = bounding_ellipse(center, y, r, b, camK; p=2, solver=Mosek.Optimizer, silent=true)

## Angular Bounds
# Δθs, status_angbounds, gaps = angular_bounds(center, H)


# ## Refinement
# # marginalize via projection
# P = [zeros(3,4) diagm(ones(3))]
# Hinf_t = inv(P*inv(Hinf)*P')
# P = [zeros(3,9) diagm(ones(3))]
# H2_t = inv(P*inv(H2)*P')


# ## Visualize
# surfinf = ellipse_to_surf(Hinf_t, center[10:12], 100)
# surf2 = ellipse_to_surf(H2_t, center[10:12], 100)
# # plot ellipse
# Plots.plot([center[10]],[center[11]],[center[12]], seriestype=:scatter, label="center")
# Plots.scatter!([0],[0],[0],label="camera")
# p2 = Plots.scatter3d!(surfinf[1,:], surfinf[2,:], surfinf[3,:], label="Inf")
# p2 = Plots.scatter3d!(surf2[1,:], surf2[2,:], surf2[3,:], label="2")