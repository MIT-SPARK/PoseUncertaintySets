## Compute explicit translation and rotations bounds with specified method
# 
# Lorenzo Shaikewitz, 7/31/2025

using Serialization
using Statistics
using DataFrames, TexTables

using PoseUncertaintySets

# parameters
dataset = "lmo"
α = 0.1
p = 2 # cannot do Inf
sdporder = 1 # GRCC compares against order 1, RANSAG uses order 2!
pose = "ransag"

object_ids = [1,5,6,9,8,11,12]
cadpath = "../data/$dataset/models_eval/"
posepath = "../data/$dataset/pose_$(pose)_$(round(Int,α*100))_$(string(p)).dat"
savepath = "../data/$dataset/bounds_ransag_o$(sdporder)_$(round(Int,α*100))_$(string(p)).dat"

# load data
keypoint_data, gt = load_keypoint_data(calibrate_lp, dataset; p=p, α=α)
poses = deserialize(posepath)["solns"]


println("Starting RANSAG Baseline...")
bounds = DataFrame()
for object_id in object_ids
    println("\n------------$object_id------------")
    # solve!
    angles, trans, statuses, times, gaps = dataset_ransag_bounds(keypoint_data, poses, object_id; order=sdporder)
    
    # save
    frames = collect(keys(angles))
    (d::Dict)(k) = d[k] # make dictionary callable
    bounds_obj = DataFrame(frame=frames, id=object_id, time=times.(frames),
                θ=angles.(frames), t=trans.(frames), 
                status_t=statuses[1].(frames), gap_t=gaps[1].(frames), 
                status_θ=statuses[2].(frames), gap_θ=gaps[2].(frames)
    )
    global bounds
    bounds = [bounds; bounds_obj]
end

# save!
out_dict = Dict("bounds"=>bounds)
serialize(savepath, out_dict)