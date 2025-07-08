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

function load_and_bound(p)
    ## Load data
    keypoint_data, gt = load_keypoint_data(calibrate_lp, p=p, α=0.1)
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
    R_est, t_est, gap, SDP_status = gaussianpose(y, r, b, camK; silent=true, order=2)

    ## S-Lemma
    # l2 approach
    if p == 2
        center = [vec(R_est); t_est]
    else
        center = [rotm2quat(R_est); t_est]
    end
    rad, status = bounding_sphere(center, y, r, b, camK; p=p, order=2, silent=false)

    @infiltrate

    return rad, status
end


rad2, status2 = load_and_bound(2)
radi, statusi = load_and_bound(Inf)

# Conclusion: Inf norm works and is generally tighter and faster than 2 norm!
# Next: second order version of S-Lemma for Inf norm (experience shows it will have only small penalty)