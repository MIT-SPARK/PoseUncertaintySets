## Compute the central pose
# Lorenzo Shaikewitz, 5/7/2025

using Serialization

# TODO: move to module
include("../src/uncertaintyset.jl")
include("../src/centralpose.jl")

# PARAMETERS
l2 = true # alt is linf
α = 0.1
real_cal = true
exclude_bop200 = false
silent = true

percent = true
lowerb = 0.1 # -0.9
upperb = 2 # 0.7
params = Params(l2, α, real_cal, percent, lowerb, upperb)
savename = "l2_01_percent0.1"

# object_id = 9
object_ids = [1,5,6,8,9,10,11,12]

# load data
cadpath = "./data/lmo/models_eval/"
datapath = @sprintf "./data/lmo/l%s_%02d%s.dat" (l2 ? "2" : "inf") Int(α*10) (real_cal ? "_real" : "")
# datapath = @sprintf "./data/lmo/l%s_%03d%s.dat" (l2 ? "2" : "inf") Int(α*100) (real_cal ? "_real" : "")
all_data = deserialize(datapath)
camK = all_data["camK"]
img_names = all_data["img"]
num_frames = length(img_names)

calibrated_frames = open("./data/lmo/valid_test_frames.txt") do f
    readlines(f) |> (s-> parse.(Int,s))
end

status_dict = Dict()
for object_id in object_ids
    println("-------$object_id-------")

    all_status = Vector{MOI.TerminationStatusCode}(undef, num_frames)
    all_feas = -ones(Int, num_frames)
    all_gaps = -ones(num_frames)
    all_times = -ones(num_frames)
    t_ests = Vector{Any}(undef, num_frames)
    R_ests = Vector{Any}(undef, num_frames)
    for frame = 1:num_frames
        if exclude_bop200
            if frame in calibrated_frames
                continue
            end
        end
        if !(object_id in keys(all_data["radii"][frame]))
            continue
        end
        
        r = all_data["radii"][frame][object_id] .+ 1e-3 # make sure 0 radii doesn't happen
        y = all_data["pixel_measurements"][frame][object_id]
        b = all_data["canonical_kpts"][frame][object_id]

        # find center
        if l2
            if percent
                out = @timed centralpose_percent_l2(y, r, b, camK; lowerb=lowerb, upperb=upperb, silent=silent)
            else
                out = @timed centralpose_l2(y, r, b, camK; upperb=upperb*r, lowerb=lowerb*r, silent=silent)
            end
        else
            if percent
                out = @timed centralpose_percent_linf(y, r, b, camK; lowerb=lowerb, upperb=upperb, silent=silent)
            else
                out = @timed centralpose_linf(y, r, b, camK; upperb=upperb*r, lowerb=lowerb*r, silent=silent)
            end
        end
        sdp_pose, status, vars_proj, feas, gap = out.value
        time = out.time - out.compile_time

        # save
        all_status[frame] = status
        all_feas[frame] = feas
        all_gaps[frame] = gap
        all_times[frame] = time

        R_ests[frame] = project2SO3(sdp_pose[1])
        t_ests[frame] = sdp_pose[2]
        
        println("[$object_id] $frame: $status")
    end
    status_dict[object_id] = Dict("all_status"=>all_status, "all_feas"=>all_feas, "all_gaps"=>all_gaps, 
        "all_times"=>all_times, "R_ests"=>R_ests, "t_ests"=>t_ests)
end



print(datapath)
println(" done!")

serialize(savename, Dict("status" => status_dict, "params"=> params))

# ## TODO: MOVE SUMMARY
# ## Compute errors
# import GeometryBasics
# using FileIO, MeshIO

# cad = FileIO.load(cadpath*(@sprintf "obj_%06d.ply" object_id))
# cad_m = GeometryBasics.Mesh(GeometryBasics.coordinates(cad)/1000, cad.faces)

# t_errors = -ones(num_frames)
# R_errors = -ones(num_frames)
# proj_errors = -ones(num_frames)
# for frame = 1:num_frames
#     if all_feas[frame] != -1
#         gt = all_data["gt_poses"][frame][object_id]
#         R_gt = project2SO3(gt[1])
#         t_gt = gt[2] / 1000. # [m]
#         R_est = R_ests[frame]
#         t_est = t_ests[frame]

#         # pose error
#         err_R = roterror(R_gt, R_est)
#         err_t = norm(t_est - t_gt)
#         t_errors[frame] = err_t*1000
#         R_errors[frame] = err_R

#         # 2D projection error
#         # https://openaccess.thecvf.com/content_cvpr_2016/papers/Brachmann_Uncertainty-Driven_6D_Pose_CVPR_2016_paper.pdf supplement
#         proj_error = 0.
#         for coord in GeometryBasics.coordinates(cad_m)
#             pixel_est = camK*(R_est*coord + t_est)
#             pixel_est ./= pixel_est[3]
#             pixel_gt = camK*(R_gt*coord + t_gt)
#             pixel_gt ./= pixel_gt[3]
#             proj_error += norm(pixel_est - pixel_gt)
#         end
#         proj_error /= length(GeometryBasics.coordinates(cad_m))
#         proj_errors[frame] = proj_error
#     end
# end

# ## Print metrics
# using Statistics

# visible_in_frames = sum(.!(all_feas .≈ -1.0))
# @printf "Feasible for %d/%d frames (%.2f%%)\n" sum(all_feas .== 1) visible_in_frames sum(all_feas .== 1)/visible_in_frames*100
# @printf "Reasonable for %d/%d frames (%.2f%%)\n" sum(t_errors[all_feas .!= -1] .< 1e3) visible_in_frames sum(t_errors[all_feas .!= -1] .< 1e3)/visible_in_frames*100
# @printf "Tight for %d/%d frames (%.2f%%)\n" sum(all_gaps[all_feas .!= -1] .< 1e-3) visible_in_frames sum(all_gaps[all_feas .!= -1] .< 1e-3)/visible_in_frames*100
# println("")

# function summarize(e)
#     @printf "Mean: %.1f ± %.1f\n" mean(e) std(e)
#     @printf "Median: %.1f (%.1f, %.1f)\n" median(e) quantile(e,0.25) quantile(e,0.75)
# end

# t_filtered = t_errors[all_feas .!= -1]
# printstyled("t errors (mm):\n",underline=true)
# summarize(t_filtered[t_filtered .< 1e3])

# R_filtered = R_errors[all_feas .!= -1]
# printstyled("R errors (deg):\n",underline=true)
# summarize(R_filtered[t_filtered .< 1e3])

# proj_filtered = proj_errors[all_feas .!= -1]
# printstyled("Proj errors (px):",underline=true)
# @printf " %.2f%% under 5 px\n" sum(proj_filtered .< 5)/visible_in_frames*100
# summarize(proj_filtered[t_filtered .< 1e3])