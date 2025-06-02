## Conformalize poses directly
# Lorenzo Shaikewitz, 5/14/2025

using Serialization
using JuMP
using Printf

using Statistics

include("../src/utils.jl")

# Load centers and keypoint data
all_data = deserialize("data/lmo/linf_04_real.dat")
all_status = deserialize("linf_04_percent01")["status"]
calibrated_frames = open("./data/lmo/valid_test_frames.txt") do f
    readlines(f) |> (s-> parse.(Int,s))
end

object_ids = [1,5,6,9,8,10,11,12]

# Calibrate
all_ang_scores = Dict()
all_t_scores = Dict()
for object_id in object_ids
    num_frames = length(all_status[object_id]["all_feas"])

    ang_scores = []
    t_scores = []
    for frame in calibrated_frames
        if all_status[object_id]["all_feas"][frame] != 1
            continue
        end

        # pose estimate
        R_est = all_status[object_id]["R_ests"][frame]
        t_est = all_status[object_id]["t_ests"][frame]

        # ground truth
        R_gt = project2SO3(all_data["gt_poses"][frame][object_id][1])
        t_gt = all_data["gt_poses"][frame][object_id][2] / 1000

        # calibration
        append!(ang_scores, roterror(R_est, R_gt)) # deg
        append!(t_scores, norm(t_est - t_gt)) # m
    end
    # print calibration data
    @printf "[%02d] α = 0.1: ± %.2f deg\t± %.2f mm (%d frames)\n" object_id quantile(ang_scores, 1-0.1) quantile(t_scores, 1-0.1)*1000 length(t_scores)
    @printf "     α = 0.6: ± %.2f deg\t± %.2f mm\n" quantile(ang_scores, 1-0.6) quantile(t_scores, 1-0.6)*1000

    all_ang_scores[object_id] = ang_scores
    all_t_scores[object_id] = t_scores
end



