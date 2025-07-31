## Plot pose uncertainty set on the image
# used in main keypoints/uncertainty figure
# 
# Lorenzo Shaikewitz, 7/31/2025

import Plots
using SimpleRotations
using PoseUncertaintySets

image_parent = "../data/bop/lmo/test_all/000002/rgb"
cadpath = "../data/bop/lmo/models_eval/"

# settings used for figure
object_id = 9
frame = 352
α = 0.1
p = Inf

# load data and calibrate
keypoint_data, gt = load_keypoint_data(calibrate_lp, p=p, α=α)
prob = get_problem(keypoint_data, object_id, frame)

# get pose estimate and ellipse
R_est, t_est, gap_pose, status_pose = gaussianpose(prob; silent=true, order=2)
center = [rotm2quat(R_est); t_est]
H, gap_slem, status_slem = bounding_ellipse_quat(center, prob; silent=true, order=2)

# plot pose estimate
p, img = plot_image(image_parent, frame)
(p, seg) = plot_mask!(p, img, cadpath, object_id, (R_est, t_est), camK; lazy=false)
plot_outline!(p, img, seg)

# plot uncertainty
# TODO: how do I sample feasible rotations from the ellipse?
# easiest bet: sample translations from marg. translation ellipse and rotations from marg. rotation ellipse.