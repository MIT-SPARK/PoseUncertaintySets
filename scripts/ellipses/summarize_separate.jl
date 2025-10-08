## Summarize bound results (separate ellipsoids)
# 
# Lorenzo Shaikewitz, 8/1/2025

using Serialization
using MAT
using Statistics
using LinearAlgebra
using DataFrames, TexTables
using JuMP
import Plots

using SimpleRotations
using PoseUncertaintySets

## global parameters
# dataset = "lmo"
# object_ids = [1,5,6,9,8,11,12] # omit 10
# dataset = "ycbv"
# object_ids =  [1:12;14;15] # omit 13, 16:21
dataset = "cast"
object_ids = [1]
α = 0.1
p = Inf
sdporder = 1
pose = "pnp2"


## Load data
# load slem separate
save_path = "../data/$dataset/ellipse_slemsep_rotm_o$(sdporder)_$(round(Int,α*100))_$(string(p)).dat"
slem_sep_dict = deserialize(save_path)
H_slem_sep = slem_sep_dict["ellipses"]
# load slem
save_path = "../data/$dataset/ellipse_slem_rotm_o$(sdporder)_$(round(Int,α*100))_$(string(p)).dat"
slem_dict = deserialize(save_path)
H_slem = slem_dict["ellipses"]
# load poses
posepath = "../data/$dataset/pose_$(pose)_$(round(Int,α*100))_$(string(p)).dat"
poses = deserialize(posepath)["solns"]


## Preprocess data: remove bad points
# preprocess slem separate
data_slem_sep = filter(:id => f-> f in object_ids, slem_sep_dict["data"])
filter_slem_sep = data_slem_sep.optimal_r .&& data_slem_sep.optimal_t
println("Separate filtered: $(sum(filter_slem_sep))/$(size(data_slem_sep,1))")
# preprocess slem separate
data_slem = filter(:id => f-> f in object_ids, slem_dict["data"])
filter_slem = data_slem.optimal
println("Together filtered: $(sum(filter_slem))/$(size(data_slem,1))")

# combine filters
filter_combined = filter_slem .& filter_slem_sep
println("Combined filter: $(sum(filter_combined))/$(size(data_slem,1))")


# separate: get explicit bounds
slem_sep_t = []
slem_sep_r = []
for object_id in object_ids
    frames = filter(:id => x->x==object_id, data_slem_sep[filter_combined,:]).frame
    for (i,H) in enumerate(H_slem_sep[object_id].(frames))
        R_est = poses[object_id][1][frames[i]]
        _, (boundst, boundsθ) = project_ellipse(H, R_est)
        push!(slem_sep_t, boundst)
        push!(slem_sep_r, boundsθ)
    end
end
slem_sep_t = reduce(hcat, slem_sep_t)
slem_sep_r = reduce(hcat, slem_sep_r)
# togther: get explicit bounds
slem_t = []
slem_r = []
for object_id in object_ids
    frames = filter(:id => x->x==object_id, data_slem[filter_combined,:]).frame
    for (i,H) in enumerate(H_slem[object_id].(frames))
        R_est = poses[object_id][1][frames[i]]
        _, (boundst, boundsθ) = project_ellipse(H, R_est)
        push!(slem_t, boundst)
        push!(slem_r, boundsθ)
    end
end
slem_t = reduce(hcat, slem_t)
slem_r = reduce(hcat, slem_r)


## Summarize / plot!
println("")
println("t. vol (together): $(mean(4/3*π*slem_t[1,:].*slem_t[2,:].*slem_t[3,:])) m^3")
println("t. vol (separate): $(mean(4/3*π*slem_sep_t[1,:].*slem_sep_t[2,:].*slem_sep_t[3,:])) m^3")
println("r. vol (together): $(mean(4/3*π*min.(90,slem_r[1,:]).*min.(90.,slem_r[2,:]).*min.(90,slem_r[3,:]))) deg^3")
println("r. vol (separate): $(mean(4/3*π*min.(90,slem_sep_r[1,:]).*min.(90.,slem_sep_r[2,:]).*min.(90,slem_sep_r[3,:]))) deg^3")
println("time (together): $(mean(data_slem.time_s[filter_combined]))")
println("time (separate): $(mean(data_slem_sep.time_s[filter_combined]))")


# translation CDF (volume)
p_tcdf = Plots.plot(ylabel="CDF", title="Translation Uncertainty", xlabel="Volume (m^3)")
plot_cdf!(p_tcdf, 4/3*π*slem_sep_t[1,:].*slem_sep_t[2,:].*slem_sep_t[3,:], label="separate")
plot_cdf!(p_tcdf, 4/3*π*slem_t[1,:].*slem_t[2,:].*slem_t[3,:], label="together")

Plots.plot!(xscale=:log10)
Plots.plot!(yticks=0:0.2:1, ylim=[0,1])
Plots.plot!(p_tcdf,xlim=[1e-5,200],xticks=[1e-4,1e-2,1,100])
Plots.plot!(fontfamily="helvetica")

# angular CDF (volume)
p_acdf = Plots.plot(ylabel="CDF", title="Angular Uncertainty", xlabel="Volume (deg^3)")
plot_cdf!(p_acdf, 4/3*π*min.(90,slem_sep_r[1,:]).*min.(90.,slem_sep_r[2,:]).*min.(90,slem_sep_r[3,:]), label="separate")
plot_cdf!(p_acdf, 4/3*π*min.(90,slem_r[1,:]).*min.(90.,slem_r[2,:]).*min.(90,slem_r[3,:]), label="together")

Plots.plot!(xscale=:log10)
Plots.plot!(yticks=0:0.2:1, ylim=[0,1])
Plots.plot!(p_acdf,xlim=[500,1e7],xticks=[1e3,1e4,1e5,1e6])
Plots.plot!(fontfamily="helvetica")

Plots.plot(p_tcdf, p_acdf)