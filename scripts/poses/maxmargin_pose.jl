## Compute pose of each frame using a specific method
# 
# Lorenzo Shaikewitz, 7/31/2025

using Serialization
using Printf
using Statistics
using DataFrames, TexTables

using PoseUncertaintySets

# parameters
dataset = "ycbv"
α = 0.1
p = 2

object_ids = [1,5,6,9,8,10,11,12]
cadpath = "../data/$dataset/models_eval/"
savepath = "../data/$dataset/pose_maxmargin_$(round(Int,α*100))_$(string(p)).dat"

# load data
keypoint_data, gt = load_keypoint_data(calibrate_lp, dataset; p=p, α=α)


# run method!
println("Starting Max Margin...")
errs = DataFrame()
solns = Dict()
for object_id in object_ids
    println("\n------------$object_id------------")
    # solve!
    Rs, ts, gaps, times, statuses = dataset_pose_est(keypoint_data, object_id, maxmarginpose; silent=true)
    # errors
    R_errs, t_errs = calc_pose_errors(Rs, ts, gt, object_id)
    proj_errs = calc_projection_errors(Rs, ts, keypoint_data, gt, object_id, cadpath)
    # save
    errs_obj = DataFrame(R=collect(values(R_errs)), t=collect(values(t_errs)), proj=collect(values(proj_errs)), gap=collect(values(gaps)), time=collect(values(times)), frame=collect(keys(R_errs)), id=object_id)
    global errs, solns
    errs = [errs;errs_obj]
    solns[object_id] = (Rs, ts)
end

# save!
pose_dict = Dict("solns"=>solns, "errs"=>errs)
serialize(savepath, pose_dict)