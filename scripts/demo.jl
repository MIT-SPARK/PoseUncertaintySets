## A simple demo of pose & uncertainty estimation
# 
# Lorenzo Shaikewitz, 9/24/2025

using LinearAlgebra
using Printf
using Clarabel, MosekTools
import Plots

using PoseUncertaintySets
using SimpleRotations

dataset = "lmo" # ["lmo", "ycbv", "cast"]
object_id = 9
frame = 11
estorder = 1

# load data
keypoint_datai, gt = load_keypoint_data(calibrate_lp, "lmo"; p=Inf, α=0.1)
prob = get_problem(keypoint_datai, object_id, frame)

# pose estimation
R, t, gap_pose, status_pose = gaussianpose(probi; silent=true, order=2)

# joint uncertainty ellipse
H, gap, status = bounding_ellipse_quat(centeri, probi; order=order, silent=false)