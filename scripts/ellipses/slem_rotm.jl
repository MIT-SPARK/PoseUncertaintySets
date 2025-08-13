## Compute bounding ellipse at each frame using specified method
# 
# Lorenzo Shaikewitz, 7/31/2025

# TODO:
# add order to rotation slemma
# fix inf norm slemma extra constraints

using Serialization
using Statistics
using DataFrames, TexTables
using JuMP

using PoseUncertaintySets

# parameters
dataset = "lmo"
object_ids = [1,5,6,9,8,10,11,12]
# dataset = "ycbv"
# object_ids = 1:21
# dataset = "cast"
# object_ids = [1]
α = 0.1
p = Inf
order=1
pose = "pnp2"

cadpath = "../data/$dataset/models_eval/"
posepath = "../data/$dataset/pose_$(pose)_$(round(Int,α*100))_$(string(p)).dat"
savepath = "../data/$dataset/ellipse_slem_rotm_o$(order)_$(round(Int,α*100))_$(string(p)).dat"

# load data
keypoint_data, gt = load_keypoint_data(calibrate_lp, dataset; p=p, α=α)
poses = deserialize(posepath)["solns"]


println("\nStarting S-Lemma (rotation matrix)")
data = DataFrame()
ellipses = Dict()
for object_id in object_ids
    println("\n------------$object_id------------")
    # solve!
    Hs, statuses, times, gaps = dataset_slem(keypoint_data, poses, object_id; order=order, quat=false)

    # save
    frames = collect(keys(Hs))
    (d::Dict)(k) = d[k] # make dictionary callable
    data_obj = DataFrame(frame=frames, id=object_id, time_s=times.(frames),
                optimal=(MOI.OPTIMAL .== statuses.(frames)) .|| (MOI.ALMOST_OPTIMAL .== statuses.(frames)),
                gaps=gaps.(frames)
    )
    global data, ellipses
    data = [data; data_obj]
    ellipses[object_id] = Hs
end
println("")

# save!
out_dict = Dict("ellipses"=>ellipses, "data"=>data)
serialize(savepath, out_dict)