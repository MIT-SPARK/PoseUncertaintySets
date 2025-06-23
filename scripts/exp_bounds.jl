## Experiment: bounds given a pose
# Lorenzo Shaikewitz, 6/4/2025

using Serialization
using Printf
using Statistics
using DataFrames, TexTables
using JuMP
import Plots

using PoseUncertaintySets

## parameters
α = 0.1
save_path = "../data/bounds_alpha$(round(Int,α*100)).dat"
keypoint_path = "../data/pose_alpha$(round(Int,α*100)).dat"
object_ids = [1,5,6,9,8,10,11,12]
# cadnames = Dict(1=>"ape", 5=>"can", 6=>"cat", 8=>"driller", 9=>"duck", 10=>"eggbox", 11=>"glue", 12=>"holepuncher")

## Load data
keypoint_data, gt = load_keypoint_data(calibrate_lp, p=2, α=α)

pose_dict = deserialize(keypoint_path)
(solns_g1, errs_g1) = pose_dict["g1"]
(solns_g2, errs_g2) = pose_dict["g2"]
(solns_r, errs_r) = pose_dict["r"]

## OURS
println("\nStarting S-Lemma + Refinement..")
bounds_slem = DataFrame()
data_slem = Dict()
for object_id in object_ids
    println("\n------------$object_id------------")
    # solve!
    Hs, Δθs, Δts, statuses, times, gaps = dataset_slem_bounds(keypoint_data, solns_g2, object_id)

    angles = Float64.(reduce(hcat,collect(values(Δθs))))
    times = reduce(hcat, collect(values(times)))
    trans = reduce(hcat,collect(values(Δts)))
    
    # save
    bounds_obj = DataFrame(frame=collect(keys(Hs)), id=object_id, time_s=times[1,:],
                θx=angles[1,:], θy=angles[2,:], θz=angles[3,:], time_r=times[2,:],
                tu1=trans[1,1:3:end], tu2=trans[1,2:3:end], tu3=trans[1,3:3:end],
                tl1=trans[2,1:3:end], tl2=trans[2,2:3:end], tl3=trans[2,3:3:end], time_t=times[3,:],
                optimal=[MOI.OPTIMAL in statuses[frame][2:end] || MOI.ALMOST_OPTIMAL in statuses[frame][2:end] for frame in keys(statuses)])
    global bounds_slem, data_slem
    bounds_slem = [bounds_slem; bounds_obj]
    data_slem[object_id] = (statuses, gaps)
end
println("")

## RANSAG BASELINE
println("\nStarting RANSAG Baseline...")
bounds_ransag = DataFrame()
data_ransag = Dict()
for object_id in object_ids
    println("\n------------$object_id------------")
    # solve!
    Δθs, Δts, statuses, times, gaps = dataset_ransag_bounds(keypoint_data, solns_g2, object_id)
    
    angles = collect(values(Δθs))
    times = collect(values(times))
    trans = collect(values( Δts ))
    
    # save
    bounds_obj = DataFrame(frame=collect(keys(Δθs)), id=object_id, time=times,
                θ=angles, t=trans, 
                optimal=[MOI.OPTIMAL in statuses[frame] || MOI.ALMOST_OPTIMAL in statuses[frame] for frame in keys(statuses)])
    global bounds_ransag, data_ransag
    bounds_ransag = [bounds_ransag; bounds_obj]
    data_ransag[object_id] = (statuses, gaps)
end
println("")



# save data for later
bounds_dict = Dict("slem"=>bounds_slem, "ransag"=>bounds_ransag, "slem_data"=>data_slem, "ransag_data"=>data_ransag)
serialize(save_path, bounds_dict)

## Display results
# bounds_dict = deserialize(save_path)
bounds_slem = bounds_dict["slem"]
bounds_ransag = bounds_dict["ransag"]

# runtime comparison
bounds_slem.time = bounds_slem[:, :time_s] + bounds_slem[:, :time_r] + bounds_slem[:, :time_t]
df_slem = summarize_by(bounds_slem, :id, [:time], stats=("SLEM"=> x->mean(skipmissing(x))*1000))
df_ransag = summarize_by(bounds_ransag, :id, [:time], stats=("RANS"=> x->mean(skipmissing(x))*1000))
runtime_results = [df_slem df_ransag]

df_mslem = summarize(bounds_slem, [:time], stats=("SLEM"=> x->mean(skipmissing(x))*1000))
df_mransag = summarize(bounds_ransag, [:time], stats=("RANS"=> x->mean(skipmissing(x))*1000))
runtime_results = [runtime_results; df_mslem df_mransag]

# runtime breakdown (S-Lemma)
breakdown_slem = summarize_by(bounds_slem, :id, [:time_s, :time_r, :time_t, :time], stats=("SLEM"=> x->mean(skipmissing(x))*1000))
df_mslem = summarize(bounds_slem, [:time_s, :time_r, :time_t, :time], stats=("SLEM"=> x->mean(skipmissing(x))*1000))
breakdown_slem = [breakdown_slem; df_mslem]

# rot bounds CDF 
# TODO:
# - add RANSAG, be consistent in frames (filter by RANSAG status too)
# - Filter out object 10
filterval = bounds_slem.optimal .== true .&& bounds_ransag.optimal .== true
p_rcdf = Plots.plot(ylabel="CDF", title="Rotation Bounds")
plot_cdf!(p_rcdf, bounds_slem.θx[filterval]; label="x")
plot_cdf!(p_rcdf, bounds_slem.θy[filterval]; label="y")
plot_cdf!(p_rcdf, bounds_slem.θz[filterval]; label="z")
plot_cdf!(p_rcdf, bounds_ransag.θ[filterval]; label="RANSAG")


# trans range CDF (note this is 2x the radius)
p_tcdf = Plots.plot(ylabel="CDF", title="Translation Range")
plot_cdf!(p_tcdf, (bounds_slem.tu1 - bounds_slem.tl1)[filterval]; label="1")
plot_cdf!(p_tcdf, (bounds_slem.tu2 - bounds_slem.tl2)[filterval]; label="2")
plot_cdf!(p_tcdf, (bounds_slem.tu3 - bounds_slem.tl3)[filterval]; label="3")
plot_cdf!(p_tcdf, (2*bounds_ransag.t)[filterval .&& bounds_ransag.t .> 0]; label="RANSAG")
Plots.plot!(xscale=:log10)
