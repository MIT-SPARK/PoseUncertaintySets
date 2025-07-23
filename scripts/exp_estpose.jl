## Experiment: pose estimation
# Lorenzo Shaikewitz, 6/4/2025

using Serialization
using Printf
using Statistics
using DataFrames, TexTables

using PoseUncertaintySets

# parameters
α = 0.1
save_path = "../data/pose_alpha$(round(Int,α*100)).dat"
cadpath = "../data/bop/lmo/models_eval/"
object_ids = [1,5,6,9,8,10,11,12]
# cadnames = Dict(1=>"ape", 5=>"can", 6=>"cat", 8=>"driller", 9=>"duck", 10=>"eggbox", 11=>"glue", 12=>"holepuncher")

keypoint_data, gt = load_keypoint_data(calibrate_lp, p=2, α=α)

# GAUSSIAN, ORDER 1
println("\nStarting Gaussian Order 1...")
errs_g1 = DataFrame()
solns_g1 = Dict()
for object_id in object_ids
    println("\n------------$object_id------------")
    # solve!
    Rs, ts, gaps, times, statuses = dataset_pose_est(keypoint_data, object_id, gaussianpose; silent=true, order=1)
    # errors
    R_errs, t_errs = calc_pose_errors(Rs, ts, gt, object_id)
    proj_errs = calc_projection_errors(Rs, ts, keypoint_data, gt, object_id, cadpath)
    # save
    errs_obj = DataFrame(R=collect(values(R_errs)), t=collect(values(t_errs)), proj=collect(values(proj_errs)), gap=collect(values(gaps)), time=collect(values(times)), frame=collect(keys(R_errs)), id=object_id)
    global errs_g1, solns_g1
    errs_g1 = [errs_g1;errs_obj]
    solns_g1[object_id] = (Rs, ts)
end
println("")

# GAUSSIAN, ORDER 2
println("\nStarting Gaussian Order 2...")
errs_g2 = DataFrame()
solns_g2 = Dict()
for object_id in object_ids
    println("\n------------$object_id------------")
    # solve!
    Rs, ts, gaps, times, statuses = dataset_pose_est(keypoint_data, object_id, gaussianpose; silent=true, order=2)
    # errors
    R_errs, t_errs = calc_pose_errors(Rs, ts, gt, object_id)
    proj_errs = calc_projection_errors(Rs, ts, keypoint_data, gt, object_id, cadpath)
    # save
    errs_obj = DataFrame(R=collect(values(R_errs)), t=collect(values(t_errs)), proj=collect(values(proj_errs)), gap=collect(values(gaps)), time=collect(values(times)), frame=collect(keys(R_errs)), id=object_id)
    global errs_g2, solns_g2
    errs_g2 = [errs_g2;errs_obj]
    solns_g2[object_id] = (Rs, ts)
end
println("")

# RANSAG
println("\nStarting RANSAG...")
errs_r = DataFrame()
solns_r = Dict()
for object_id in object_ids
    println("\n------------$object_id------------")
    # solve!
    Rs, ts, purse_emptys, times = dataset_pose_est(keypoint_data, object_id, ransagpose)
    # errors
    R_errs, t_errs = calc_pose_errors(Rs, ts, gt, object_id)
    proj_errs = calc_projection_errors(Rs, ts, keypoint_data, gt, object_id, cadpath)
    # save
    errs_obj = DataFrame(R=collect(values(R_errs)), t=collect(values(t_errs)), proj=collect(values(proj_errs)), gap=collect(values(purse_emptys)), time=collect(values(times)), frame=collect(keys(R_errs)), id=object_id)
    global errs_r, solns_r
    errs_r = [errs_r;errs_obj]
    solns_r[object_id] = (Rs, ts)
end
println("")

# MAX MARGIN
println("\nStarting Max Margin...")
errs_mm = DataFrame()
solns_mm = Dict()
for object_id in object_ids
    println("\n------------$object_id------------")
    # solve!
    Rs, ts, gaps, times, statuses = dataset_pose_est(keypoint_data, object_id, maxmarginpose; silent=true)
    # errors
    R_errs, t_errs = calc_pose_errors(Rs, ts, gt, object_id)
    proj_errs = calc_projection_errors(Rs, ts, keypoint_data, gt, object_id, cadpath)
    # save
    errs_obj = DataFrame(R=collect(values(R_errs)), t=collect(values(t_errs)), proj=collect(values(proj_errs)), gap=collect(values(gaps)), time=collect(values(times)), frame=collect(keys(R_errs)), id=object_id)
    global errs_mm, solns_mm
    errs_mm = [errs_mm;errs_obj]
    solns_mm[object_id] = (Rs, ts)
end
println("")

# save data for later
pose_dict = Dict("g1"=>(solns_g1, errs_g1), "g2"=>(solns_g2, errs_g2), "r"=>(solns_r,errs_r), "mm"=>(solns_mm,errs_mm))
serialize(save_path, pose_dict)

## Display results
# pose_dict = deserialize(save_path)
(solns_g1, errs_g1) = pose_dict["g1"]
(solns_g2, errs_g2) = pose_dict["g2"]
(solns_r, errs_r) = pose_dict["r"]
(solns_mm, errs_mm) = pose_dict["mm"]

# tightness
df_g1 = summarize_by(errs_g1, :id, [:gap], stats=("G1"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
df_g2 = summarize_by(errs_g2, :id, [:gap], stats=("G2"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
df_r  = summarize_by(errs_r , :id, [:gap], stats=("RA"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
df_mm = summarize_by(errs_mm, :id, [:gap], stats=("MM"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
gap_results = [df_g2 df_r df_mm]

df_mg1 = summarize(errs_g1, [:gap], stats=("G1"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
df_mg2 = summarize(errs_g2, [:gap], stats=("G2"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
df_mr  = summarize(errs_r , [:gap], stats=("RA"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
df_mmm = summarize(errs_mm, [:gap], stats=("MM"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
gap_results = [gap_results; df_mg2 df_mr df_mmm]

# times
df_g1 = summarize_by(errs_g1, :id, [:time], stats=("G1"=> x->mean(filter(!ismissing,x)*1000)))
df_g2 = summarize_by(errs_g2, :id, [:time], stats=("G2"=> x->mean(filter(!ismissing,x)*1000)))
df_r  = summarize_by(errs_r , :id, [:time], stats=("RA"=> x->mean(filter(!ismissing,x)*1000)))
df_mm = summarize_by(errs_mm, :id, [:time], stats=("MM"=> x->mean(filter(!ismissing,x)*1000)))
time_results = [df_g2 df_r df_mm]

df_mg1 = summarize(errs_g1, [:time], stats=("G1"=> x->mean(filter(!ismissing,x)*1000)))
df_mg2 = summarize(errs_g2, [:time], stats=("G2"=> x->mean(filter(!ismissing,x)*1000)))
df_mr  = summarize(errs_r , [:time], stats=("RA"=> x->mean(filter(!ismissing,x)*1000)))
df_mmm = summarize(errs_mm, [:time], stats=("MM"=> x->mean(filter(!ismissing,x)*1000)))
time_results = [time_results; df_mg2 df_mr df_mmm]

# projection errors
df_g1 = summarize_by(errs_g1, :id, [:proj], stats=("G1"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
df_g2 = summarize_by(errs_g2, :id, [:proj], stats=("G2"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
df_r  = summarize_by(errs_r , :id, [:proj], stats=("RA"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
df_mm = summarize_by(errs_mm, :id, [:proj], stats=("MM"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
proj_results = [df_g2 df_r df_mm]

df_mg1 = summarize(errs_g1, [:proj], stats=("G1"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
df_mg2 = summarize(errs_g2, [:proj], stats=("G2"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
df_mr  = summarize(errs_r , [:proj], stats=("RA"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
df_mmm = summarize(errs_mm, [:proj], stats=("MM"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
proj_results = [proj_results; df_mg2 df_mr df_mmm]

tab = join_table("Proj" => proj_results, "Time" => time_results, "Gap" => gap_results)
# to_tex(tab) |> print