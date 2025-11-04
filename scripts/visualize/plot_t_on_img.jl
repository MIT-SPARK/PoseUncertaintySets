## Plot pose uncertainty set on the image
# used in main keypoints/uncertainty figure
# 
# Lorenzo Shaikewitz, 7/31/2025

import Plots
import FileIO, GeometryBasics, Images
import JSON
using Printf
using LinearAlgebra
using SimpleRotations
using PoseUncertaintySets
using ColorSchemes, Colors



# settings used for figure
# dataset = "lmo"
# object_id = 9
# frame = 352
# dataset = "ycbv"
# object_id = 14
# frame = 480083 # 501125
dataset = "cast"
object_id = 1
frame = 1000 # 568, 1000, 703
α = 0.1
p = Inf

image_parent = "../data/$dataset/test"

# load data and calibrate
keypoint_data, gt = load_keypoint_data(calibrate_lp, dataset, p=p, α=α)
data_cast = JSON.parsefile("../data/$dataset/detections_test.json")
prob = get_problem(keypoint_data, object_id, frame); camK = prob.camK

# get pose estimate and ellipse
R_est, t_est, gap_pose, status_pose = gaussianpose(prob; silent=true, order=2)
center = [rotm2quat(R_est); t_est]
H, gap_slem, status_slem = bounding_ellipse_quat(center, prob; silent=true, order=2)
# marginalize
(Ht, Hθ), (boundst, boundsθ) = project_ellipse(H, R_est)

# PURSE version
# trans_bound, trans_gap, ang_bound, ang_gap, status = purse_bounds([vec(R_est); t_est], prob.y, prob.r, prob.b, prob.camK; order=2, silent=true)
# Ht = diagm(ones(3))/trans_bound.^2

# plot image with frame
p, img = plot_image(image_parent, frame, data=data_cast)
plot_frame!(p, (R_est, t_est), camK; gt=false)
# plot_frame!(p, gt[frame][1], camK; gt=true)
# project(p) = (camK * (p))[1:2] ./ (camK * (p))[3]
# o2d = project(t_est)
# Plots.scatter!([o2d[2]], [o2d[1]], ms=5)

# plot keypoints
plot_keypoints!(p, prob.y, prob.r, p=Inf)

# plot t uncertainty ellipse
surf = ellipse_to_surf(Ht, t_est, 500)
# sample from interior too

# project 3D -> 2D
surf_px = [camK] .* eachcol(surf)
depths = reduce(hcat, surf_px)[3,:]
surf_px = reduce(hcat, [s[1:2] / s[3] for s in surf_px])
surf_px = round.(Int,surf_px)
# draw on mask
dmin, dmax = extrema(depths)
norm_depths = (depths .- dmin) ./ (dmax - dmin)
scheme = reverse(ColorSchemes.Greens_4)

sample_mask = Images.RGBA.(copy(img)).*0
sample_depths = ones(size(img))
for (i,col) in enumerate(eachcol(surf_px))
    # only override for closer depths
    if sample_depths[col[2], col[1]] > norm_depths[i]
        sample_mask[col[2], col[1]] = alphacolor(get(scheme, norm_depths[i]), 0.8)
        sample_depths[col[2], col[1]] = norm_depths[i]
    end
end
Plots.plot!(sample_mask)

Plots.title!("Translation Ellipse (frame $frame)")