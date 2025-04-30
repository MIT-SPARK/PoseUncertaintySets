## Sample from pose uncertainty set using max margin
# Run central pose first!
# Lorenzo Shaikewitz

using Serialization
using JuMP

# TODO: move to module
include("../src/uncertaintyset.jl")
include("../src/center_l2.jl")

# Parameters
object_id = 9

analytic = false
lowerb = 0
upperb = 10
silent = true

# Load data
cadpath = "./data/lmo/models_eval/"
datapath = "./data/lmo/l2.dat"
all_data = deserialize(datapath)
camK = all_data["camK"]
img_names = all_data["img"]
num_frames = length(img_names)

# select frame
frame = 1

## Get center
r = all_data["radii"][frame][object_id]
y = all_data["pixel_measurements"][frame][object_id]
b = all_data["canonical_kpts"][frame][object_id]

# build uncertainty set
q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
q_eqs = SO3_constraints()

# grid size
grid_size = 0.005 # [m]
max_offset = 0.1 # [m]
pts_per_axis = Int(round(max_offset/grid_size))

# find center
sdp_pose, status, data, model = center_l2(q_front, q_backproj, q_eqs; analytic=analytic, lowerb=lowerb, upperb=upperb*r, silent=silent)

# local refinement + gap
tight = data[1]
vars_proj = data[3]
if tight
    gap = 0.
    est_pose = sdp_pose
else
    est_pose, loc_status, vars_proj, gap = local_refine(q_front, q_backproj, q_eqs, data; analytic=analytic, lowerb=lowerb, upperb=upperb*r, silent=silent)
end

## Sample
# build grid
t_center = t_ests[frame]
tmin = t_center .- max_offset
tmax = t_center .+ max_offset
ranges = LinRange.(tmin, tmax, pts_per_axis)
test_centers = []
for i = 1:pts_per_axis
    for j = 1:pts_per_axis
        for k = 1:pts_per_axis
            push!(test_centers, [ranges[1][i]; ranges[2][j]; ranges[3][k]])
        end
    end
end

margins = -ones(length(test_centers))
poses = []
println("Taking $(length(test_centers)) samples!")
for (i,center) in enumerate(test_centers)   
    # find center
    sdp_pose, status, data, model = center_l2(q_front, q_backproj, q_eqs; analytic=analytic, lowerb=lowerb, upperb=upperb*r, silent=silent, 
        tlower=center .- grid_size, tupper=center .+ grid_size)

    # local refinement + gap
    tight = data[1]
    vars_proj = data[3]
    if tight
        gap = 0.
        est_pose = sdp_pose
    else
        est_pose, loc_status, vars_proj, gap = local_refine(q_front, q_backproj, q_eqs, data; analytic=analytic, lowerb=lowerb, upperb=upperb*r, silent=silent, 
            tlower=center .- grid_size, tupper=center .+ grid_size)
    end

    tol = 1e-3
    feas = check_feas(q_front, q_backproj, q_eqs, vars_proj; tol=tol, silent=silent)
    t = vars_proj[end-2:end]
    feas = feas && sum(t.+tol.>= center .- grid_size)==3 && sum(t.-tol .<= center .+ grid_size)==3
        
    if feas
        if analytic
            margins[i] = sum(log.(vars_proj[1:end-3-9]))
        else
            margins[i] = sum(vars_proj[1:end-3-9])
        end
        push!(poses, est_pose)
    end
    if mod(i, 100) == 0
        print("$i ")
    end
end

## Plot
import Plots
good_centers = reduce(hcat, [p[2] for p in poses])

# good_centers = reduce(hcat, test_centers[margins .>= 0])
good_margins = margins[margins .>= 0]

p1=Plots.plot(eachrow(good_centers)...,seriestype=:scatter,zcolor=good_margins, color=:blues)
Plots.plot!(xlims=[t_center[1]-max_offset, t_center[1]+max_offset],
            ylims=[t_center[2]-max_offset, t_center[2]+max_offset],
            zlims=[t_center[3]-max_offset, t_center[3]+max_offset])

println("Found $(length(good_centers)) good centers.")
p1