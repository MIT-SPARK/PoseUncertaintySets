## Summarize bound results
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
dataset = "lmo"
object_ids = [1,5,6,9,8,11,12] # omit 10
# dataset = "ycbv"
# object_ids =  [1:12;14;15] # omit 13, 16:21
# dataset = "cast"
# object_ids = [1]
α = 0.1
p = Inf
sdporder = 2
if sdporder == 2
    quat = true
else
    quat = false
end
pose = "pnp2"


## Load data
# load grcc
if sdporder == 2
    save_path = "../data/$dataset/$(dataset)_grcc.mat"
    file = matopen(save_path)
    t_grcc = real.(read(file, "translations"))
    r_grcc = read(file, "rotations")
    close(file)
end
# load ransag
save_path = "../data/$dataset/bounds_ransag_o$(sdporder)_$(round(Int,α*100))_2.dat" # p=2 only
bounds_ransag = deserialize(save_path)["bounds"]
# load slem
if !quat
    save_path = "../data/$dataset/ellipse_slem_rotm_o$(sdporder)_$(round(Int,α*100))_$(string(p)).dat"
else
    save_path = "../data/$dataset/ellipse_slem_quat_o$(sdporder)_$(round(Int,α*100))_$(string(p)).dat"
end
slem_dict = deserialize(save_path)
H_slem = slem_dict["ellipses"]
# load poses
posepath = "../data/$dataset/pose_$(pose)_$(round(Int,α*100))_$(string(p)).dat"
poses = deserialize(posepath)["solns"]


## Preprocess data: remove bad points
# preprocess grcc
# not using the same filter because format is different
if sdporder == 2
    filter_grcc = (r_grcc .!= 0) .& (t_grcc .!= 0)
    println("GRCC filtered: $(sum(filter_grcc))/$(size(filter_grcc,1))")
end
# preprocess ransag
filter!(:id => f-> f in object_ids, bounds_ransag)
filter_t = bounds_ransag.status_t .== MOI.OPTIMAL .|| bounds_ransag.status_t .== MOI.ALMOST_OPTIMAL .|| bounds_ransag.status_t .== MOI.SLOW_PROGRESS
println("RANSAG translations: $(sum(filter_t))/$(size(bounds_ransag,1))")
filter_θ = bounds_ransag.status_θ .== MOI.OPTIMAL .|| bounds_ransag.status_θ .== MOI.ALMOST_OPTIMAL .|| bounds_ransag.status_θ .== MOI.SLOW_PROGRESS
println("RANSAG rotations: $(sum(filter_θ))/$(size(bounds_ransag,1))")
# preprocess slem
data_slem = filter(:id => f-> f in object_ids, slem_dict["data"])
filter_slem = data_slem.optimal
println("S-Lemma filtered: $(sum(filter_slem))/$(size(data_slem,1))")

# combine filters
filter_combined = filter_t .& filter_θ# .& filter_slem
println("Combined filter: $(sum(filter_combined))/$(size(data_slem,1))")

# s-lemma: get explicit bounds
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
# println("Certificate: $(sum(slem_dict["data"][:,:gaps] .== 1))/$(size(slem_dict["data"],1))")

# translation CDF (volume)
if sdporder == 2
    p_tcdf = Plots.plot(ylabel="CDF", title="Translation Uncertainty", xlabel="Volume (m^3)")
end
plot_cdf!(p_tcdf, 4/3*π*(bounds_ransag.t)[filter_combined .&& bounds_ransag.t .> 0].^3; label="RANSAG")
plot_cdf!(p_tcdf, 4/3*π*slem_t[1,:].*slem_t[2,:].*slem_t[3,:], label="order $sdporder")
if sdporder == 2
    plot_cdf!(p_tcdf, 4/3*π*(t_grcc)[filter_grcc].^3; label="GRCC")
end
Plots.plot!(xscale=:log10)
Plots.plot!(yticks=0:0.2:1, ylim=[0,1])
Plots.plot!(p_tcdf,xlim=[1e-5,200],xticks=[1e-4,1e-2,1,100])
Plots.plot!(fontfamily="helvetica")

# angular CDF (volume)
if sdporder == 2
    p_acdf = Plots.plot(ylabel="CDF", title="Angular Uncertainty", xlabel="Volume (deg^3)")
end
plot_cdf!(p_acdf, 4/3*π*min.((bounds_ransag.θ)[filter_combined .&& bounds_ransag.θ .> 0], 90.).^3; label="RANSAG")
# plot_cdf!(p_acdf, slem_r[1,:]; label="\\theta_3",c=color_ours,lw=2)
# plot_cdf!(p_acdf, slem_r[2,:]; label="\\theta_2",c=color_ours,lw=2)
# plot_cdf!(p_acdf, slem_r[3,:]; label="\\theta_1",c=color_ours,lw=2)
plot_cdf!(p_acdf, 4/3*π*min.(90,slem_r[1,:]).*min.(90.,slem_r[2,:]).*min.(90,slem_r[3,:]), label="order $sdporder")
if sdporder == 2
    plot_cdf!(p_acdf, 4/3*π*min.((r_grcc)[filter_grcc], 90.).^3; label="GRCC")
end
Plots.plot!(xscale=:log10)
Plots.plot!(yticks=0:0.2:1, ylim=[0,1])
Plots.plot!(p_acdf,xlim=[500,1e7],xticks=[1e3,1e4,1e5,1e6])
Plots.plot!(fontfamily="helvetica")


# ang range CDF
# filterval = bounds_ransag.status_θ .== MOI.OPTIMAL .|| bounds_ransag.status_θ .== MOI.ALMOST_OPTIMAL .|| bounds_ransag.status_θ .== MOI.SLOW_PROGRESS

# p_acdf = Plots.plot(ylabel="CDF", title="Angular Bounds")
# plot_cdf!(p_acdf, (bounds_ransag.θ)[filterval .&& bounds_ransag.θ .> 0]; label="RANSAG")

Plots.plot(p_tcdf, p_acdf)