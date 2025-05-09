## Run after centralpose to summarize results
# Lorenzo Shaikewitz, 5/8/2025

function summarize(e)
    @printf "Mean: %.1f ± %.1f\n" mean(e) std(e)
    @printf "Median: %.1f (%.1f, %.1f)\n" median(e) quantile(e,0.25) quantile(e,0.75)
end

# linf_01

include("../src/utils.jl")

# to load data
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

cadpath = "./data/lmo/models_eval/" # _eval
cadnames = Dict(1=>"ape", 5=>"can", 6=>"cat", 8=>"driller", 9=>"duck", 10=>"eggbox", 11=>"glue", 12=>"holepuncher")

println(datapath)

all_proj_errors = Dict()
for object_id in object_ids
    all_status = status_dict[object_id]["all_status"]
    all_feas = status_dict[object_id]["all_feas"]
    all_gaps = status_dict[object_id]["all_gaps"]
    all_times = status_dict[object_id]["all_times"]

    R_ests = status_dict[object_id]["R_ests"]
    t_ests = status_dict[object_id]["t_ests"]


    println("\n-------$object_id ($(cadnames[object_id]))-------")
    ## Compute errors
    import GeometryBasics
    using FileIO, MeshIO

    cad = FileIO.load(cadpath*(@sprintf "obj_%06d.ply" object_id))
    cad_m = GeometryBasics.Mesh(GeometryBasics.coordinates(cad)/1000, cad.faces)

    t_errors = -ones(num_frames)
    R_errors = -ones(num_frames)
    proj_errors = -ones(num_frames)
    for frame = 1:num_frames
        if all_feas[frame] != -1
            gt = all_data["gt_poses"][frame][object_id]
            R_gt = project2SO3(gt[1])
            t_gt = gt[2] / 1000. # [m]
            R_est = R_ests[frame]
            t_est = t_ests[frame]

            # pose error
            err_R = roterror(R_gt, R_est)
            err_t = norm(t_est - t_gt)
            t_errors[frame] = err_t*1000
            R_errors[frame] = err_R

            # 2D projection error
            # https://openaccess.thecvf.com/content_cvpr_2016/papers/Brachmann_Uncertainty-Driven_6D_Pose_CVPR_2016_paper.pdf supplement
            proj_error = 0.
            for coord in GeometryBasics.coordinates(cad_m)
                pixel_est = camK*(R_est*coord + t_est)
                pixel_est ./= pixel_est[3]
                pixel_gt = camK*(R_gt*coord + t_gt)
                pixel_gt ./= pixel_gt[3]
                proj_error += norm(pixel_est - pixel_gt)
            end
            proj_error /= length(GeometryBasics.coordinates(cad_m))
            proj_errors[frame] = proj_error
        end
    end

    ## Print metrics
    using Statistics

    visible_in_frames = sum(.!(all_feas .≈ -1.0))
    @printf "Feasible for %d/%d frames (%.2f%%)\n" sum(all_feas .== 1) visible_in_frames sum(all_feas .== 1)/visible_in_frames*100
    @printf "Reasonable for %d/%d frames (%.2f%%)\n" sum(t_errors[all_feas .!= -1] .< 1e3) visible_in_frames sum(t_errors[all_feas .!= -1] .< 1e3)/visible_in_frames*100
    @printf "Tight for %d/%d frames (%.2f%%)\n" sum(all_gaps[all_feas .!= -1] .< 1e-3) visible_in_frames sum(all_gaps[all_feas .!= -1] .< 1e-3)/visible_in_frames*100
    println("")

    t_filtered = t_errors[all_feas .!= -1]
    printstyled("t errors (mm):\n",underline=true)
    summarize(t_filtered[t_filtered .< 1e3])

    R_filtered = R_errors[all_feas .!= -1]
    printstyled("R errors (deg):\n",underline=true)
    summarize(R_filtered[t_filtered .< 1e3])

    proj_filtered = proj_errors[all_feas .!= -1]
    printstyled("Proj errors (px):",underline=true)
    @printf " %.2f%% under 5 px\n" sum(proj_filtered .< 5)/visible_in_frames*100
    summarize(proj_filtered[t_filtered .< 1e3])

    ## Save
    all_proj_errors[object_id] = proj_filtered
end

println("\n$datapath averages:")

sum_proj = 0.
tot_frames = 0
mean_projs = []
for (key, val) in all_proj_errors
    global sum_proj += sum(val .< 5)
    global tot_frames += length(val)
    global mean_projs
    push!(mean_projs, sum(val .< 5)/length(val))
end
printstyled("Correct proj errors (px):",underline=true)
@printf " %.2f%% under 5 px\n" sum_proj/tot_frames*100

printstyled("Mean proj errors (px):",underline=true)
@printf " %.2f%% under 5 px\n" mean(mean_projs)*100