# Quick script for comparison

using Serialization
using DataFrames
using PoseUncertaintySets

dataset = "lmo"
object_id = 9
frame = 111

α = 0.1
p = 2
pose = "pnp2"
posepath = "../data/$dataset/pose_$(pose)_$(round(Int,α*100))_$(string(p)).dat"

## Load data
keypoint_data, gt = load_keypoint_data(calibrate_lp, dataset; p=p, α=α)
pose_data = deserialize(posepath)["solns"]

## Run RANSAG
camK = keypoint_data["K"]
# frame-specific data
r = keypoint_data["r"][frame][object_id]
y = keypoint_data["y"][frame][object_id]
b = keypoint_data["b"][object_id]

# eliminate missing measurements
y = y[1:2,r .>= 0]
y = [y[1:2,:]; ones(size(y,2))']
b = b[:, r .>= 0]
r = r[r .>= 0]

# pose data
R_est = pose_data[object_id][1][frame]
t_est = pose_data[object_id][2][frame]
center = [vec(R_est); t_est]

# RANSAG
order = 2
# trans_bound, trans_gap, ang_bound, ang_gap, status = purse_bounds(center, y, r, b, camK; order=order, silent=false)
q_front, q_backproj = PoseUncertaintySets.uncertaintyset_l2(y, r, b, camK)
q_eqs = PoseUncertaintySets.SO3_constraints()
trans_bound, ang_bound = grcc_bounds(center, q_front, q_backproj, q_eqs; order=2, silent=false)