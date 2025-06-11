## Pose estimation on a single frame
# Lorenzo Shaikewitz, 6/11/2025

using Serialization
using LinearAlgebra
using Printf

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

## Pose estimation
R_est, t_est, purse_empty = ransagpose(r, y, b, camK)
# R_est, t_est, gap, SDP_status = gaussianpose(r, y, b, camK; silent=true, order=2)

## Check against gt
gt = keypoints_data["gt_poses"][frame][object_id]
R_gt = project2SO3(gt[1])
t_gt = gt[2] / 1000. # [m]
err_R = roterror(R_gt, R_est)
err_t = norm(t_est - t_gt)*1000
@printf "R error: %.2f deg\n" err_R
@printf "t error: %.2f mm\n" err_t