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
all_status_ellipse = Vector{MOI.TerminationStatusCode}(undef, num_frames)
all_times_ellipse = -ones(num_frames)
all_H = Vector{Any}(undef, num_frames)
for frame = 1:num_frames
    if (all_feas[frame] != 1)
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
    
    println("$frame: $status")
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

# Plot
import Plots

Plots.plot(sort(trans_principal_axes_fake[1,all_feas.==1]), (1:num_feas)./num_feas, label="1")
Plots.plot!(sort(trans_principal_axes_fake[2,all_feas.==1]), (1:num_feas)./num_feas, label="2")
Plots.plot!(sort(trans_principal_axes_fake[3,all_feas.==1]), (1:num_feas)./num_feas, label="3")
p1=Plots.plot!(xscale=:log10, ylabel="CDF", xlabel="Error Bound (m)")

Plots.plot(sort(trans_principal_axes[1,all_feas.==1]), (1:num_feas)./num_feas, label="1")
Plots.plot!(sort(trans_principal_axes[2,all_feas.==1]), (1:num_feas)./num_feas, label="2")
Plots.plot!(sort(trans_principal_axes[3,all_feas.==1]), (1:num_feas)./num_feas, label="3")
p2=Plots.plot!(xscale=:log10, xticks=[1e-5, 1e-1,1,10,100,1000], ylabel="CDF", xlabel="Error Bound (m)")
