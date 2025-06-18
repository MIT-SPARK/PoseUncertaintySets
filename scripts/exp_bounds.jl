## Experiment: bounds given a pose
# Lorenzo Shaikewitz, 6/4/2025

using Serialization
using Printf
using Statistics
using DataFrames, TexTables
using JuMP

using PoseUncertaintySets

## parameters
α = 0.1
save_path = "../data/bounds_alpha$(round(Int,α*100)).dat"
keypoint_path = "../data/pose_alpha$(round(Int,α*100)).dat"
object_ids = [1,5,6,9,8,10,11,12]
# cadnames = Dict(1=>"ape", 5=>"can", 6=>"cat", 8=>"driller", 9=>"duck", 10=>"eggbox", 11=>"glue", 12=>"holepuncher")

## Load data
keypoint_data, gt = load_keypoint_data(calibrate_l2, α=α)

pose_dict = deserialize(keypoint_path)
(solns_g1, errs_g1) = pose_dict["g1"]
(solns_g2, errs_g2) = pose_dict["g2"]
(solns_r, errs_r) = pose_dict["r"]

## OURS
println("\nStarting S-Lemma + Refinement..")
bounds_slem = DataFrame()
for object_id in object_ids
    println("\n------------$object_id------------")
    # solve!
    Hs, Δθs, Δts, statuses, times = dataset_slem_bounds(keypoint_data, solns_g2, object_id)

    angles = Float64.(reduce(hcat,collect(values(Δθs))))
    times = reduce(hcat, collect(values(times)))
    trans = reduce(hcat,collect(values(Δts)))
    
    # save
    bounds_obj = DataFrame(frame=collect(keys(Hs)), id=object_id, time_s=times[1,:],
                θx=angles[1,:], θy=angles[1,:], θz=angles[3,:], time_r=times[2,:],
                tu1=trans[1,1:3:end], tu2=trans[1,2:3:end], tu3=trans[1,3:3:end],
                tl1=trans[2,1:3:end], tl2=trans[2,2:3:end], tl3=trans[2,3:3:end], time_t=times[3,:],
                bad=[MOI.SLOW_PROGRESS in statuses[frame][2:end] for frame in keys(statuses)])
    global bounds_slem
    bounds_slem = [bounds_slem; bounds_obj]
end
println("")

## RANSAG BASELINE
println("\nStarting RANSAG Baseline...")
bounds_slem = DataFrame()
for object_id in object_ids
    println("\n------------$object_id------------")
    # solve!
    Hs, Δθs, Δts, statuses, times = dataset_bounds(keypoint_data, solns_g2, object_id)

    angles = Float64.(reduce(hcat,collect(values(Δθs))))
    times = reduce(hcat, collect(values(times)))
    trans = reduce(hcat,collect(values(Δts)))
    
    # save
    bounds_obj = DataFrame(frame=collect(keys(Hs)), id=object_id, time_s=times[1,:],
                θx=angles[1,:], θy=angles[1,:], θz=angles[3,:], time_r=times[2,:],
                tu1=trans[1,1:3:end], tu2=trans[1,2:3:end], tu3=trans[1,3:3:end],
                tl1=trans[2,1:3:end], tl2=trans[2,2:3:end], tl3=trans[2,3:3:end], time_t=times[3,:],
                bad=[MOI.SLOW_PROGRESS in statuses[frame][2:end] for frame in keys(statuses)])
    global bounds_slem
    bounds_slem = [bounds_slem; bounds_obj]
end
println("")



# save data for later
