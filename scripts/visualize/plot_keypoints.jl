## Plot keypoint detections and uncertainty box on a single image.
# Used to make keypoint figure 
#
# Lorenzo Shaikewitz, 7/31/2025

import Plots
using PoseUncertaintySets

image_parent = "../data/bop/lmo/test_all/000002/rgb"
cadpath = "../data/bop/lmo/models_eval/"

# settings used for figure
object_id = 9
frame = 352
α = 0.1
p = Inf
plot_gt = false

# load data and calibrate
keypoint_data, gt = load_keypoint_data(calibrate_lp, p=p, α=α)
prob = get_problem(keypoint_data, object_id, frame)

# plot
p, img = plot_image(image_parent, frame)
plot_keypoints!(p, prob.y, prob.r, p=Inf)
if plot_gt
    plot_3d_keypoints!(p, gt[frame][object_id], prob.b, prob.camK)
end

# save
# Plots.savefig(p, "keypoints.svg")

p