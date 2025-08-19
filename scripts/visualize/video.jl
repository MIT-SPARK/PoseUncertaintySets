## Make video of detections
# Lorenzo Shaikewitz, 8/14/2025

import Plots, JSON
using Serialization
using DataFrames
using PoseUncertaintySets

# settings
dataset = "ycbv"
object_id = 13
α = 0.1
p = 2#Inf
plot_gt = !false
plot_pose = true # false to show keypoints
plot_frame = !true

# load data and calibrate
image_parent = "../data/$dataset/test"
cadpath = "../data/$dataset/models_eval/"
posepath = "../data/$dataset/pose_pnp2_$(round(Int,α*100))_$(string(p)).dat"

keypoint_data, gt = load_keypoint_data(calibrate_lp, dataset; p=p, α=α)
if dataset == "cast"
    data_cast = JSON.parsefile("../data/$dataset/detections_test.json")
end
if plot_pose
    poses = deserialize(posepath)["solns"]
end

anim = Plots.@animate for frame in sort(collect(keys(keypoint_data["y"])))
    if !(object_id in keys(keypoint_data["y"][frame]))
        continue
    end
    if plot_pose && !(frame in keys(poses[object_id][1]))
        continue
    end

    prob = get_problem(keypoint_data, object_id, frame)
    if dataset != "cast"
        p, img = plot_image(image_parent, frame)
    else
        p, img = plot_image(image_parent, frame; data=data_cast)
    end

    if !plot_pose
        plot_keypoints!(p, prob.y, prob.r, p=Inf)
        if plot_gt
            plot_3d_keypoints!(p, gt[frame][object_id], prob.b, prob.camK)
        end
    end


    if plot_pose
        R_est = poses[object_id][1][frame]
        t_est = poses[object_id][2][frame]
        if plot_frame
            plot_frame!(p, (R_est, t_est), prob.camK)
        else
            (p, seg) = plot_mask!(p, img, cadpath, object_id, (R_est, t_est), prob.camK; lazy=true)
        end
        if plot_gt && plot_frame
            plot_frame!(p, gt[frame][object_id], prob.camK; gt=true)
        end
    end

    if frame % 10 == 0
        print("$frame ")
    end
end
println("\nSaving...")
Plots.gif(anim, "$(dataset)_$object_id.mp4", fps=15)

# TODO: only works for LMO
# need to work for YCB-V (multiple folders) and drone (weird format)