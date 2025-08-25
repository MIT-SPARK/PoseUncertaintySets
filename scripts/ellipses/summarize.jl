## Summarize bound results
# 
# Lorenzo Shaikewitz, 8/1/2025

using Serialization
using Statistics
using LinearAlgebra
using DataFrames, TexTables
using JuMP
import Plots

using SimpleRotations
using PoseUncertaintySets

# global parameters
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
filter_combined = filter_t .& filter_θ .& filter_slem
println("Combined filter: $(sum(filter_combined))/$(size(data_slem,1))")

# continue preprocessing slem
slem_t = []
slem_r = []
if quat
    Pt = [zeros(3,4) I]
    Pr = [I zeros(4,3)]
else
    Pt = [zeros(3,9) I]
    Pr = [I zeros(9,3)]
end
for object_id in object_ids
    frames = filter(:id => x->x==object_id, data_slem[filter_combined,:]).frame
    for (i,H) in enumerate(H_slem[object_id].(frames))
        # project to translations
        Ht_inv = Pt*pinv(H)*Pt'
        push!(slem_t, sqrt.(eigvals(Ht_inv)))

        # project to rotations
        Hr = inv(Pr*pinv(H)*Pr')
        R_est = poses[object_id][1][frames[i]]
        if quat
            q_est = rotm2quat(R_est)
            Hr_centered = Ω2(q_est)'*Hr*Ω2(q_est)
            Pθ = [zeros(3,1) I]
            Hθ_inv = Pθ*pinv(Hr_centered)*Pθ'
            push!(slem_r, abs.(2*asin.(min.(1.,sqrt.(eigvals(Hθ_inv)))))*180/π)
        else
            Hr_centered = kron(R_est',diagm(ones(3)))'*Hr*kron(R_est',diagm(ones(3)))
            Pθ = zeros(3,9)
            Pθ[1,6] = 1; Pθ[1,8] = -1; Pθ[2,7] = 1; Pθ[2,3] = -1; Pθ[3,2] = 1; Pθ[3,4] = -1
            Hθ_inv = 1/4*Pθ*pinv(Hr_centered)*Pθ'
            push!(slem_r, abs.(asin.(min.(1.,sqrt.(eigvals(Hθ_inv)))))*180/π)
        end
    end
end

slem_t = reduce(hcat, slem_t)
slem_r = reduce(hcat, slem_r)

# translation CDF (radius)
p_tcdf = Plots.plot(ylabel="CDF", title="Translation Radii (Order $sdporder)")
plot_cdf!(p_tcdf, (bounds_ransag.t)[filter_combined .&& bounds_ransag.t .> 0]; label="RANSAG")
plot_cdf!(p_tcdf, slem_t[1,:]; label="1")
plot_cdf!(p_tcdf, slem_t[2,:]; label="2")
plot_cdf!(p_tcdf, slem_t[3,:]; label="3")
Plots.plot!(xscale=:log10)

# angular CDF (radius)
p_acdf = Plots.plot(ylabel="CDF", title="Angular Bounds (Order $sdporder)")
plot_cdf!(p_acdf, (bounds_ransag.θ)[filter_combined .&& bounds_ransag.θ .> 0]; label="RANSAG")
plot_cdf!(p_acdf, slem_r[1,:]; label="1")
plot_cdf!(p_acdf, slem_r[2,:]; label="2")
plot_cdf!(p_acdf, slem_r[3,:]; label="3")


# ang range CDF
# filterval = bounds_ransag.status_θ .== MOI.OPTIMAL .|| bounds_ransag.status_θ .== MOI.ALMOST_OPTIMAL .|| bounds_ransag.status_θ .== MOI.SLOW_PROGRESS

# p_acdf = Plots.plot(ylabel="CDF", title="Angular Bounds")
# plot_cdf!(p_acdf, (bounds_ransag.θ)[filterval .&& bounds_ransag.θ .> 0]; label="RANSAG")