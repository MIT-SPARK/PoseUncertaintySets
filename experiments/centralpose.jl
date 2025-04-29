## Compute the central pose
# Lorenzo Shaikewitz, 4/29/2025

using Serialization

# TODO: move to module
include("../src/uncertaintyset.jl")
include("../src/center_l2.jl")

# Parameters
object_id = 9

analytic = false
lowerb = -1
upperb = 10
silent = true

# Load data
datapath = "./data/lmo/l2.dat"
all_data = deserialize(datapath)
camK = all_data["camK"]
img_names = all_data["img"]
num_frames = length(img_names)

# Run all frames
all_status = Vector{MOI.TerminationStatusCode}(undef, num_frames)
all_feas = -ones(Int, num_frames)
all_gaps = -ones(num_frames)
t_ests = Vector{Any}(undef, num_frames)
R_ests = Vector{Any}(undef, num_frames)
for frame = 1:num_frames
    if !(object_id in keys(all_data["radii"][frame]))
        continue
    end

    r = all_data["radii"][frame][object_id]
    y = all_data["pixel_measurements"][frame][object_id]
    b = all_data["canonical_kpts"][frame][object_id]

    # build uncertainty set
    q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
    q_eqs = SO3_constraints()

    # find center
    sdp_pose, status, data, model = center_l2(q_front, q_backproj, q_eqs; analytic=analytic, lowerb=lowerb, upperb=upperb, silent=silent)

    # local refinement + gap
    est_pose, loc_status, gap = local_refine(q_front, q_backproj, q_eqs, data; analytic=analytic, lowerb=lowerb, upperb=upperb, silent=silent)

    # check feasibility
    feas = 

    all_status[frame] = status
    all_feas[frame] = feas
    all_gaps[frame] = gap

    # save estimated pose
    R_est = est_pose[1]
    t_est = est_pose[2]
    t_ests[frame] = t_est
    R_ests[frame] = R_est
    
    println("$frame: $status")
end