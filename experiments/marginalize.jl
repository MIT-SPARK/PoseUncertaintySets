## Marginalize bounding ellipse to translation and angular bounds
# Requires from ellipse:
# - all_H
# Requires from central pose:
# - all_feas
# - R_ests
# Lorenzo Shaikewitz, 4/29/2025

using Serialization
using MosekTools

# TODO: move to module
include("../src/boundsfromellipse.jl")

# Parameters
object_id = 9

silent = true

# Load data
cadpath = "./data/lmo/models_eval/"
datapath = "./data/lmo/l2_01_real.dat"
all_data = deserialize(datapath)
camK = all_data["camK"]
img_names = all_data["img"]
num_frames = length(img_names)

# Run all frames
all_status_rot = Array{MOI.TerminationStatusCode}(undef, 3, num_frames)
all_gaps_rot = Array{Any}(undef, 3, num_frames)
all_R_bounds = Array{Any}(undef, 3, num_frames)
all_times_rot = -ones(num_frames)
for frame = 1:num_frames
    if (all_feas[frame] == -1)
        continue
    end

    # pull center from max margin
    center = [vec(R_ests[frame]); t_ests[frame]]
    # pull ellipse from s-lemma
    H = all_H[frame]

    # Rotation bounds
    out = @timed angular_bounds(center, H; silent=true)
    Δθs, status_rot, gaps_rot = out.value
    time_rot = out.time - out.compile_time
        
    # save
    all_status_rot[:,frame] = status_rot
    all_gaps_rot[:,frame] = gaps_rot
    all_R_bounds[:,frame] = Δθs
    all_times_rot[frame] = time_rot
    
    print("$frame ")
end

ang_bounds = mean.(eachrow(all_R_bounds[:,all_feas.==1]))

if sum(sum.(eachrow(all_status_rot[:,all_feas .== 1] .== OPTIMAL))) != 3*sum(all_feas .== 1)
    @warn "Not all solutions optimal--some bounds may not be accurate!"
end

# CDF Plots
Plots.plot(sort(all_R_bounds[1,all_feas.!=-1])  .+ 1e-6, (1:visible_in_frames)./visible_in_frames, label="x")
Plots.plot!(sort(all_R_bounds[2,all_feas.!=-1]) .+ 1e-6, (1:visible_in_frames)./visible_in_frames, label="y")
Plots.plot!(sort(all_R_bounds[3,all_feas.!=-1]) .+ 1e-6, (1:visible_in_frames)./visible_in_frames, label="z")
p2=Plots.plot!(ylabel="CDF", xlabel="Error Bound (deg)")

# Plots.plot(sort(all_R_bounds[1,all_feas.==1])  .+ 1e-6, (1:num_feas)./num_feas, label="x")
# Plots.plot!(sort(all_R_bounds[2,all_feas.==1]) .+ 1e-6, (1:num_feas)./num_feas, label="y")
# Plots.plot!(sort(all_R_bounds[3,all_feas.==1]) .+ 1e-6, (1:num_feas)./num_feas, label="z")
# p2=Plots.plot!(xscale=:log10, ylabel="CDF", xlabel="Error Bound (deg)")