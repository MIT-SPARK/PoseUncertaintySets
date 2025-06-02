## Conformal coverage of poses and keypoints
# Lorenzo Shaikewitz, 5/6/2025

using Serialization
using Statistics
using Printf

# TODO: move to module
include("../src/uncertaintyset.jl")

# PARAMETERS
l2 = true # alt is linf
α = 0.4
real_cal = !false
exclude_bop200 = false
object_ids = [1,5,6,9,8,10,11,12]

begin
    using Serialization, JuMP, Printf
    datafile1 = "ellipse40_fromlinf"
    ellipse_data = deserialize(datafile1)
    datafile2 = "linf_04_percent01"
    all_status = deserialize(datafile2)["status"]

    num_frames = size(all_status[5]["all_feas"],1)
end

calibrated_frames = open("./data/lmo/valid_test_frames.txt") do f
    readlines(f) |> (s-> parse.(Int,s))
end

println(datafile1)
println(datafile2)

all_visible = []
all_covered = []
for object_id in object_ids
    all_H = ellipse_data[object_id]["all_H"]

    poses_covered = -ones(Int, num_frames)
    for frame = 1:num_frames
        if exclude_bop200
            if frame in calibrated_frames
                continue
            end
        end
        if all_status[object_id]["all_feas"][frame] == -1
            continue
        end

        Rc = all_status[object_id]["R_ests"][frame]
        tc = all_status[object_id]["t_ests"][frame]
        xc = [vec(Rc); tc]

        # pose coverage
        gt = all_data["gt_poses"][frame][object_id]
        R_gt = project2SO3(gt[1])
        t_gt = gt[2] / 1000. # [m]
        
        H = all_H[frame]
        x = [vec(R_gt); t_gt]
        poses_covered[frame] = (x - xc)'*H*(x - xc) <= 1
    end
    visible_in_frames = sum(poses_covered .!= -1)
    covered = sum(poses_covered .== 1)
    @printf "[%02d] Poses: %d/%d frames (%.2f%%)\n" object_id covered visible_in_frames sum(poses_covered .== 1)/visible_in_frames*100


    push!(all_visible, visible_in_frames)
    push!(all_covered, covered)
end

println("--------------")
@printf "Pose coverage: %.2f%% (%.2f%% -- %.2f%%)\n" sum(all_covered)/sum(all_visible)*100 minimum(all_covered ./ all_visible)*100 maximum(all_covered ./ all_visible)*100