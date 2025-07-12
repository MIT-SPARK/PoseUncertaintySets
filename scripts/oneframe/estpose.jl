## Pose estimation on a single frame
# Lorenzo Shaikewitz, 6/11/2025

using Serialization
using LinearAlgebra
using Printf

using PoseUncertaintySets
using SimpleRotations

object_id = 5
frame = 80

# Load data
keypoint_data2, gt2 = load_keypoint_data(calibrate_lp, p=2  , α=0.1)
keypoint_datai, gti = load_keypoint_data(calibrate_lp, p=Inf, α=0.1)

prob2 = get_problem(keypoint_data2, object_id, frame)
probi = get_problem(keypoint_datai, object_id, frame)

# Pose estimation
# R_est, t_est, purse_empty = ransagpose(prob2)
# R_est, t_est, gap, SDP_status = gaussianpose(prob2; silent=true, order=2)
R_est, t_est, gap, SDP_status = maxmarginpose(prob2)
# R_est, t_est, gap, SDP_status = conformalpose(probi; order=2, silent=false)
# R_est, t_est, gap, SDP_status = conformalpose_local(probi; silent=true)

## Check against gt
gt = gt2[frame][object_id]
R_gt = project2SO3(gt[1])
t_gt = gt[2]
err_R = roterror(R_gt, R_est)
err_t = norm(t_est - t_gt)*1000
@printf "R error: %.2f deg\n" err_R
@printf "t error: %.2f mm\n" err_t