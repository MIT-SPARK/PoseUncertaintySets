## Generate successively higher order bounds
# this script currently does not work (need to return X in bounding_ellipse_quat).
# 
# Lorenzo Shaikewitz, 8/14/2025

using LinearAlgebra
using Printf
using Clarabel, MosekTools
import Plots

using PoseUncertaintySets
using SimpleRotations

object_id = 9
frame = 11

# load data
keypoint_datai, gt = load_keypoint_data(calibrate_lp, "lmo"; p=Inf, α=0.1)
probi = get_problem(keypoint_datai, object_id, frame)

Ri, ti, gapi, statusi = gaussianpose(probi; silent=true, order=2)

# Second order
centeri = [rotm2quat(Ri); ti]

H_trace = []
status_trace = []
X_trace = []
for order = 2:5
    H, gap, status, X = bounding_ellipse_quat(centeri, probi; order=order, silent=false)
    push!(H_trace, H)
    push!(status_trace, status)
    push!(X_trace, X)
end

data = Dict("X"=>X_trace, "H"=>H_trace)
serialize("bounds_higher", data)