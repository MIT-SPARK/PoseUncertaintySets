## Pose estimation on a single frame
# Lorenzo Shaikewitz, 6/11/2025

using Serialization
using LinearAlgebra
using Printf

using PoseUncertaintySets
using SimpleRotations

object_id = 9
frame = 101

## Load data
keypoint_data, gt = load_keypoint_data(calibrate_lp, p=2, α=0.1)
camK = keypoint_data["K"]

r = keypoint_data["r"][frame][object_id]
y = keypoint_data["y"][frame][object_id]
b = keypoint_data["b"][object_id]

# eliminate missing measurements
y = y[1:2,r .>= 0]
y = [y[1:2,:]; ones(size(y,2))']
b = b[:, r .>= 0]
r = r[r .>= 0]

## Pose estimation
# R_est, t_est, purse_empty = ransagpose(y, r, b, camK)
# R_est, t_est, gap, SDP_status = gaussianpose(y, r, b, camK; silent=true, order=2)
R_est, t_est, gap, SDP_status = maxmarginpose(y, r, b, camK)

## Check against gt
gt = gt[frame][object_id]
R_gt = project2SO3(gt[1])
t_gt = gt[2]
err_R = roterror(R_gt, R_est)
err_t = norm(t_est - t_gt)*1000
@printf "R error: %.2f deg\n" err_R
@printf "t error: %.2f mm\n" err_t