## Plot specific image
# Lorenzo Shaikewitz, 6/13/2025

import Plots

using PoseUncertaintySets

object_id = 9
frame = 102
image_parent = "../data/bop/lmo/test_all/000002/rgb"
cadpath = "../data/bop/lmo/models_eval/"

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

## Plot
p, img = plot_image(image_parent, frame)
plot_keypoints!(p, y, r)
plot_3d_keypoints!(p, gt[frame][object_id], b, camK)

# add in pose estimate
# R_est, t_est, gap, SDP_status = gaussianpose(r, y, b, camK; silent=true, order=2)
R_est, t_est, gap, SDP_status = maxmarginpose(y, r, b, camK)
plot_3d_keypoints!(p, (R_est, t_est), b, camK; msc=:blue)

# ground truth
gt = gt[frame][object_id]
R_gt = project2SO3(gt[1])
t_gt = gt[2]

## plot
p2 = Plots.plot(img, axis=false, title="Frame $frame", grid=false)
# plot_mask!(p2, img, cadpath, object_id, (R_est, t_est), camK; lazy=true)
# plot_mask!(p2, img, cadpath, object_id, (R_gt, t_gt), camK; lazy=true)
(p2, seg) = plot_mask!(p2, img, cadpath, object_id, (R_est, t_est), camK; lazy=false)
# (p2, seg) = plot_mask!(p2, img, cadpath, object_id, (R_gt, t_gt), camK; lazy=false)
plot_outline!(p2, img, seg)