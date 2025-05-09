## Conformal coverage of poses and keypoints
# Lorenzo Shaikewitz, 5/6/2025

using Serialization
using Statistics
using Printf

# TODO: move to module
include("../src/uncertaintyset.jl")

# PARAMETERS
l2 = true # alt is linf
α = 0.01
real_cal = !true
exclude_bop200 = !true
object_ids = [1,5,6,8,9,10,11,12]

# load data
datapath = @sprintf "./data/lmo/l%s_%02d%s.dat" (l2 ? "2" : "inf") Int(α*10) (real_cal ? "_real" : "")
all_data = deserialize(datapath)
camK = all_data["camK"]
img_names = all_data["img"]
num_frames = length(img_names)

calibrated_frames = open("./data/lmo/valid_test_frames.txt") do f
    readlines(f) |> (s-> parse.(Int,s))
end

println(datapath)

all_visible = []
all_covered = []
all_keypoints = []
for object_id in object_ids
    poses_covered = -ones(Int, num_frames)
    keypoints_covered = -ones(num_frames)
    for frame = 1:num_frames
        if exclude_bop200
            if frame in calibrated_frames
                continue
            end
        end
        if !(object_id in keys(all_data["radii"][frame]))
            continue
        end

        r = all_data["radii"][frame][object_id] .+ 1e-3 # make sure 0 radii doesn't happen
        y = all_data["pixel_measurements"][frame][object_id]
        b = all_data["canonical_kpts"][frame][object_id]

        # build uncertainty set
        if l2
            q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
        else
            q_front, q_backproj = uncertaintyset_linf(y, r, b, camK)
        end
        q_eqs = SO3_constraints()

        # pose coverage
        gt = all_data["gt_poses"][frame][object_id]
        R_gt = project2SO3(gt[1])
        t_gt = gt[2] / 1000. # [m]
        
        vars_proj = [zeros(size(r,1)); vec(R_gt); t_gt]
        poses_covered[frame] = check_feas(q_front, q_backproj, q_eqs, vars_proj; tol=1e-3, silent=true)

        # keypoint coverage
        y_gt = camK*(R_gt*b .+ t_gt)
        y_gt = reduce(hcat, eachcol(y_gt) ./ y_gt[3,:])
        keypoints_covered[frame] = mean(norm.(eachcol(y_gt - y)) .<= r)
    end
    visible_in_frames = sum(poses_covered .!= -1)
    covered = sum(poses_covered .== 1)
    @printf "[%02d] Poses: %d/%d frames (%.2f%%)\n" object_id covered visible_in_frames sum(poses_covered .== 1)/visible_in_frames*100
    @printf "     Kypts: %.2f%%\n" mean(keypoints_covered[poses_covered .!= -1])*100


    push!(all_visible, visible_in_frames)
    push!(all_covered, covered)
    push!(all_keypoints, mean(keypoints_covered[poses_covered .!= -1]))
end

println("--------------")
@printf "Pose coverage: %.2f%% (%.2f%% -- %.2f%%)\n" sum(all_covered)/sum(all_visible)*100 minimum(all_covered ./ all_visible)*100 maximum(all_covered ./ all_visible)*100
@printf "Kypt coverage: %.2f%% (%.2f%% -- %.2f%%)\n" mean(all_keypoints)*100 minimum(all_keypoints)*100 maximum(all_keypoints)*100