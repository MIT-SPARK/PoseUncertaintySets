## Summarize results into a table / plots
#
# Lorenzo Shaikewitz, 7/31/2025

using Serialization
using Statistics
using DataFrames, TexTables

using PoseUncertaintySets

# parameters
dataset = "ycbv"
α = 0.1
p = 2#Inf


# load pose data
datapath(method) = "../data/$dataset/pose_$(method)_$(round(Int,α*100))_$(string(p)).dat"
pose_dict = deserialize(datapath("pnp2"))
solns_g2 = pose_dict["solns"]; errs_g2 = pose_dict["errs"]
pose_dict = deserialize(datapath("ransag"))
solns_r = pose_dict["solns"]; errs_r = pose_dict["errs"]
# pose_dict = deserialize(datapath("maxmargin"))
# solns_mm = pose_dict["solns"]; errs_mm = pose_dict["errs"]

# tightness
df_g2 = summarize_by(errs_g2, :id, [:gap], stats=("PnP"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
df_r  = summarize_by(errs_r , :id, [:gap], stats=("Ran"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
# df_mm = summarize_by(errs_mm, :id, [:gap], stats=("MM"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
gap_results = [df_g2 df_r]# df_mm]

df_mg2 = summarize(errs_g2, [:gap], stats=("PnP"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
df_mr  = summarize(errs_r , [:gap], stats=("Ran"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
# df_mmm = summarize(errs_mm, [:gap], stats=("MM"=> x->100*sum(skipmissing(x).<=1e-3) / length(collect(skipmissing(x)))))
gap_results = [gap_results; df_mg2 df_mr]# df_mmm]

# times
df_g2 = summarize_by(errs_g2, :id, [:time], stats=("PnP"=> x->mean(filter(!ismissing,x)*1000)))
df_r  = summarize_by(errs_r , :id, [:time], stats=("Ran"=> x->mean(filter(!ismissing,x)*1000)))
# df_mm = summarize_by(errs_mm, :id, [:time], stats=("MM"=> x->mean(filter(!ismissing,x)*1000)))
time_results = [df_g2 df_r]# df_mm]

df_mg2 = summarize(errs_g2, [:time], stats=("PnP"=> x->mean(filter(!ismissing,x)*1000)))
df_mr  = summarize(errs_r , [:time], stats=("Ran"=> x->mean(filter(!ismissing,x)*1000)))
# df_mmm = summarize(errs_mm, [:time], stats=("MM"=> x->mean(filter(!ismissing,x)*1000)))
time_results = [time_results; df_mg2 df_mr]# df_mmm]

# projection errors
df_g2 = summarize_by(errs_g2, :id, [:proj], stats=("PnP"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
df_r  = summarize_by(errs_r , :id, [:proj], stats=("Ran"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
# df_mm = summarize_by(errs_mm, :id, [:proj], stats=("MM"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
proj_results = [df_g2 df_r]# df_mm]

df_mg2 = summarize(errs_g2, [:proj], stats=("PnP"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
df_mr  = summarize(errs_r , [:proj], stats=("Ran"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
# df_mmm = summarize(errs_mm, [:proj], stats=("MM"=> x->100*sum(skipmissing(x).<=5.) / length(collect(skipmissing(x)))))
proj_results = [proj_results; df_mg2 df_mr]# df_mmm]

tab = join_table("Proj" => proj_results, "Time" => time_results, "Gap" => gap_results)
# to_tex(tab) |> print