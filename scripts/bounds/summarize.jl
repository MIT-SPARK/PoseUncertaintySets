## Summarize bound results
# 
# Lorenzo Shaikewitz, 8/1/2025

using Serialization
using Statistics
using DataFrames, TexTables
using JuMP
import Plots

using PoseUncertaintySets

# global parameters
dataset = "lmo"
α = 0.1

# load RANSAG
method = "ransag"
p = 2
sdporder = 1
save_path = "../data/$dataset/bounds_$(method)_o$(sdporder)_$(round(Int,α*100))_$(string(p)).dat"
bounds_ransag = deserialize(save_path)["bounds"]



# trans range CDF (note this is just the radius)
filterval = bounds_ransag.status_t .== MOI.OPTIMAL .|| bounds_ransag.status_t .== MOI.ALMOST_OPTIMAL .|| bounds_ransag.status_t .== MOI.SLOW_PROGRESS

p_tcdf = Plots.plot(ylabel="CDF", title="Translation Radii")
plot_cdf!(p_tcdf, (bounds_ransag.t)[filterval .&& bounds_ransag.t .> 0]; label="RANSAG")
Plots.plot!(xscale=:log10)


# ang range CDF
filterval = bounds_ransag.status_θ .== MOI.OPTIMAL .|| bounds_ransag.status_θ .== MOI.ALMOST_OPTIMAL .|| bounds_ransag.status_θ .== MOI.SLOW_PROGRESS

p_acdf = Plots.plot(ylabel="CDF", title="Angular Bounds")
plot_cdf!(p_acdf, (bounds_ransag.θ)[filterval .&& bounds_ransag.θ .> 0]; label="RANSAG")