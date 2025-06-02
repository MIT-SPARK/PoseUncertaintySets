## Marginalize bounding ellipse to translation and angular bounds
# Requires from ellipse:
# - all_H
# Requires from central pose:
# - all_feas
# - R_ests
# Lorenzo Shaikewitz, 4/29/2025

using Serialization
using MosekTools
using Printf, JuMP

# TODO: move to module
include("../src/boundsfromellipse.jl")
include("../src/utils.jl")

# Parameters
# object_id = 9

silent = true
savename = "angbounds40_fromlinf"

# Load data
# datapath = "./data/lmo/l2_01_real.dat"
# all_data = deserialize(datapath)
# camK = all_data["camK"]
# img_names = all_data["img"]
# num_frames = length(img_names)

begin
    using Serialization, JuMP, Printf
    datafile = "linf_04_percent01"
    out = deserialize(datafile)
    println("***********$datafile************")
    status_dict = out["status"]
    object_ids = sort(Int.(keys(status_dict)))
    params = out["params"]
    datapath = @sprintf "./data/lmo/l%s_%02d%s.dat" (params.l2 ? "2" : "inf") Int(params.α*10) (params.real_cal ? "_real" : "")

    all_data = deserialize(datapath)
    camK = all_data["camK"]
    img_names = all_data["img"]
    num_frames = length(img_names)

    ellipsefile = "ellipse40_fromlinf"
    ellipse_dict = deserialize(ellipsefile)
end

marginal_dict = Dict()
for object_id in object_ids
    all_status = status_dict[object_id]["all_status"]
    all_feas = status_dict[object_id]["all_feas"]
    all_gaps = status_dict[object_id]["all_gaps"]
    all_times = status_dict[object_id]["all_times"]

    R_ests = status_dict[object_id]["R_ests"]
    t_ests = status_dict[object_id]["t_ests"]

    all_H = ellipse_dict[object_id]["all_H"]

    println("\n-------$object_id-------")

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
        
        print("[$object_id] $frame ")
    end

    marginal_dict[object_id] = Dict("all_status_rot"=>all_status_rot,
        "all_gaps_rot"=>all_gaps_rot, "all_R_bounds"=>all_R_bounds, 
        "all_times_rot"=>all_times_rot)
end

serialize(savename, marginal_dict)

all_R_bounds_allobj = zeros(Float64, 3, 0)
all_status_rot_allobj = Array{MOI.TerminationStatusCode}(undef, 3, 0)
all_feas_allobj = []
all_times_allobj = []
all_gaps_rot_allobj = zeros(Float64, 3, 0)
for (object_id, val) in marginal_dict
    global all_R_bounds_allobj, all_status_rot_allobj
    if object_id == 10
        continue
    end
    all_feas = status_dict[object_id]["all_feas"]
    all_R_bounds = val["all_R_bounds"]
    all_status_rot = val["all_status_rot"]

    append!(all_feas_allobj, all_feas)
    all_R_bounds_allobj = [all_R_bounds_allobj all_R_bounds]
    all_status_rot_allobj = [all_status_rot_allobj all_status_rot]
    all_gaps_rot_allobj = [all_gaps_rot_allobj val["all_gaps_rot"]]

    append!(all_times_allobj, val["all_times_rot"])
end
visible_in_frames = sum(all_feas_allobj .!= -1)
num_feas = sum(all_feas_allobj .==1)

# mean(all_times_allobj[all_feas_allobj .!= -1])

# ang_bounds = mean.(eachrow(all_R_bounds[:,all_feas.==1]))

if sum(sum.(eachrow(all_status_rot_allobj[:,all_feas_allobj .== 1] .== OPTIMAL))) != 3*sum(all_feas_allobj .== 1)
    @warn "Not all solutions optimal--some bounds may not be accurate!"
end
import Plots

# Gap plots
gaps_filtered = all_gaps_rot_allobj[:,all_feas_allobj .!= -1]
plot_gaps = Plots.plot(sort(vec(gaps_filtered)), (1:visible_in_frames*3)./(3*visible_in_frames), label="x")
Plots.plot!(xscale=:log10, xticks=[1e-10, 1e-6, 1e-3, 1], xlim=[1e-12,1], xlabel="Suboptimality Gap", ylabel="CDF", fontfamily="Helvetica")
Plots.vline!([1e-3])

# CDF Plots
Plots.plot(sort(all_R_bounds_allobj[1,all_feas_allobj.!=-1])  .+ 1e-6, (1:visible_in_frames)./visible_in_frames, label="x")
Plots.plot!(sort(all_R_bounds_allobj[2,all_feas_allobj.!=-1]) .+ 1e-6, (1:visible_in_frames)./visible_in_frames, label="y")
Plots.plot!(sort(all_R_bounds_allobj[3,all_feas_allobj.!=-1]) .+ 1e-6, (1:visible_in_frames)./visible_in_frames, label="z")
p3=Plots.plot!(ylabel="CDF", xlabel="Error Bound (deg)")

Plots.plot(sort(all_R_bounds_allobj[1,all_feas_allobj.==1])  .+ 1e-6, (1:num_feas)./num_feas, label="x")
Plots.plot!(sort(all_R_bounds_allobj[2,all_feas_allobj.==1]) .+ 1e-6, (1:num_feas)./num_feas, label="y")
Plots.plot!(sort(all_R_bounds_allobj[3,all_feas_allobj.==1]) .+ 1e-6, (1:num_feas)./num_feas, label="z")
p4=Plots.plot!(ylabel="CDF", xlabel="Error Bound (deg)")


# to match Hank's plots:
Plots.plot!(p4, xlim=[0,100], ylim=[0,1], yticks=[0,0.2,0.4,0.6,0.8,1], xticks=[0,20,40,60,80,100], fontfamily="Helvetica")