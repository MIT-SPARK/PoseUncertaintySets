## Uncertainty bounds on a single frame
# Lorenzo Shaikewitz, 6/11/2025

using Serialization
using LinearAlgebra
using Printf
using Clarabel, MosekTools

using PoseUncertaintySets
using SimpleRotations

keypoints_file = "../data/lmo/l2_01_real.dat"
object_id = 9
frame = 100

## Load data
keypoints_data = deserialize(keypoints_file)
camK = keypoints_data["camK"]

r = keypoints_data["radii"][frame][object_id]
y = keypoints_data["pixel_measurements"][frame][object_id]
b = keypoints_data["canonical_kpts"][frame][object_id]

## Pose estimate
# R_est, t_est, purse_empty = ransagpose(r, y, b, camK)
R_est, t_est, gap, SDP_status = gaussianpose(r, y, b, camK; silent=true, order=2)

## S-Lemma
center = [vec(R_est); t_est]
H, status = bounding_ellipse(center, y, r, b, camK; solver=Clarabel.Optimizer, silent=false)

## Angular Bounds
Δθs, status_angbounds, gaps = angular_bounds(center, H)

## Visualize

P = [zeros(3,9) diagm(ones(3))]
H_t = inv(P*inv(H)*P')

surf = ellipse_to_surf(H_t, center[10:12], 100)
# plot ellipse
Plots.plot([center[10]],[center[11]],[center[12]], seriestype=:scatter, label="center")
Plots.scatter!([0],[0],[0],label="camera")
p2 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:])#, msw=0., alpha = 1)