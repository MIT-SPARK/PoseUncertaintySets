## Compute the bounding ellipse (l2).
# Requires from central pose:
# - all_feas
# - R_ests, t_ests
# Lorenzo Shaikewitz, 4/29/2025

using Serialization
using MosekTools

# TODO: move to module
include("../src/uncertaintyset.jl")
include("../src/slemma.jl")

# Parameters
# object_id = 9

silent = true
savename = "ellipse10_fromlinf"

# Load data
# datapath = "./data/lmo/l2_01_real.dat"
# all_data = deserialize(datapath)
# camK = all_data["camK"]
# img_names = all_data["img"]
# num_frames = length(img_names)

begin
    using Serialization, JuMP, Printf
    datafile = "linf_01_percent01"
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
end


ellipse_dict = Dict()
for object_id in object_ids
    all_status = status_dict[object_id]["all_status"]
    all_feas = status_dict[object_id]["all_feas"]
    all_gaps = status_dict[object_id]["all_gaps"]
    all_times = status_dict[object_id]["all_times"]

    R_ests = status_dict[object_id]["R_ests"]
    t_ests = status_dict[object_id]["t_ests"]

    println("\n-------$object_id-------")

    # Run all frames
    all_status_ellipse = Vector{MOI.TerminationStatusCode}(undef, num_frames)
    all_times_ellipse = -ones(num_frames)
    all_H = Vector{Any}(undef, num_frames)
    for frame = 1:num_frames
        if (all_feas[frame] == -1)
            continue
        end

        r = all_data["radii"][frame][object_id]
        y = all_data["pixel_measurements"][frame][object_id]
        b = all_data["canonical_kpts"][frame][object_id]

        # build uncertainty set
        q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
        q_eqs = SO3_constraints()

        # pull center from max margin
        center = [vec(R_ests[frame]); t_ests[frame]]

        # S-Lemma
        out = @timed bounding_ellipse(center, q_front, q_backproj, q_eqs; solver=Mosek.Optimizer, silent=true)
        H, status = out.value
        time = out.time - out.compile_time

        # save
        all_status_ellipse[frame] = status
        all_times_ellipse[frame] = time

        all_H[frame] = H
        
        println("[$object_id] $frame: $status")
    end

    ## Compute marginalization / cdf
    num_feas = sum(all_feas.==1)

    all_H_ts = Vector{Any}(undef, num_frames)
    trans_principal_axes = -ones(3, num_frames)
    trans_principal_axes_fake = -ones(3,num_frames)
    for frame = 1:num_frames
        if all_feas[frame] != 1
            continue
        end

        # project to translations
        P = [zeros(3,9) diagm(ones(3))]
        H_t = inv(P*inv(all_H[frame])*P')
        all_H_ts[frame] = H_t
        trans_principal_axes[:,frame] = 1 ./ eigvals(H_t) # [m]

        # just take constant rotation
        P = [zeros(3,9) diagm(ones(3))]
        trans_principal_axes_fake[:,frame] = 1 ./ eigvals(all_H[frame][10:12,10:12]) # [m]
    end

    ellipse_dict[object_id] = Dict("all_status_ellipse"=>all_status_ellipse,
        "all_times_ellipse"=>all_times_ellipse, "all_H"=>all_H, "all_H_ts"=>all_H_ts,
        "trans_principal_axes"=>trans_principal_axes, "trans_principal_axes_fake"=> trans_principal_axes_fake)
end


serialize(savename, ellipse_dict)

trans_principal_axes_allobj = zeros(Float64, 3, 0)
trans_principal_axes_fake_allobj = zeros(Float64, 3, 0)
all_feas_allobj = []
for (object_id, val) in ellipse_dict
    if object_id == 10
        continue
    end
    all_feas = status_dict[object_id]["all_feas"]
    trans_principal_axes = val["trans_principal_axes"]
    trans_principal_axes_fake = val["trans_principal_axes_fake"]

    append!(all_feas_allobj, all_feas)
    trans_principal_axes_allobj = [trans_principal_axes_allobj trans_principal_axes]
    trans_principal_axes_fake_allobj = [trans_principal_axes_fake_allobj trans_principal_axes_fake]
end
num_feas = sum(all_feas_allobj .==1)

# Plot
import Plots

Plots.plot(sort(trans_principal_axes_fake_allobj[1,all_feas_allobj.==1]), (1:num_feas)./num_feas, label="1")
Plots.plot!(sort(trans_principal_axes_fake_allobj[2,all_feas_allobj.==1]), (1:num_feas)./num_feas, label="2")
Plots.plot!(sort(trans_principal_axes_fake_allobj[3,all_feas_allobj.==1]), (1:num_feas)./num_feas, label="3")
p1=Plots.plot!(xscale=:log10, ylabel="CDF", xlabel="Error Bound (m)")

Plots.plot(sort(trans_principal_axes_allobj[1,all_feas_allobj.==1]), (1:num_feas)./num_feas, label="1")
Plots.plot!(sort(trans_principal_axes_allobj[2,all_feas_allobj.==1]), (1:num_feas)./num_feas, label="2")
Plots.plot!(sort(trans_principal_axes_allobj[3,all_feas_allobj.==1]), (1:num_feas)./num_feas, label="3")
p2=Plots.plot!(xscale=:log10, xticks=[1e-5, 1e-1,1,10,100,1000], ylabel="CDF", xlabel="Error Bound (m)")
