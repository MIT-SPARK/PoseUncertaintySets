## Functions to perform pose estimation on datasets
# Lorenzo Shaikewitz, 6/4/2025


"""
    dataset_pose_est(keypoint_data, object_id, method; kwargs...)

Run pose estimation using `method` on all frames given `keypoint_data` and `object_id`.

# Returns
- `Rs`: rotation estimates (vector of matrices)
- `ts`: translation estimates (vector of vectors)
- `gaps`: suboptimality gaps (vector)
- `times`: runtimes (vector)
- `extras`: vector of any extra information returned by `method` (SDP status, etc.)
"""
function dataset_pose_est(keypoint_data, object_id, method; kwargs...)
    # setup
    camK = keypoint_data["camK"]
    num_frames = length(keypoint_data["radii"])

    Rs = Vector{Any}(missing, num_frames)
    ts = Vector{Any}(missing, num_frames)
    gaps = Vector{Union{Float64, Missing}}(missing, num_frames)
    times = Vector{Union{Float64, Missing}}(missing, num_frames)
    extras = Vector{Any}(missing, num_frames)

    println("Starting $num_frames frames...")
    for frame = 1:num_frames
        if !(object_id in keys(keypoint_data["radii"][frame]))
            continue
        end

        # frame-specific data
        r = keypoint_data["radii"][frame][object_id]
        y = keypoint_data["pixel_measurements"][frame][object_id]
        b = keypoint_data["canonical_kpts"][frame][object_id]

        # solve
        out = @timed method(r, y, b, camK; kwargs...)
        R_est, t_est, gap = out.value[1:3]
        extra = out.value[4:end]
        time = out.time - out.compile_time
        
        # save
        Rs[frame] = R_est
        ts[frame] = t_est
        gaps[frame] = gap
        times[frame] = time
        extras[frame] = extra

        if mod(frame, 10) == 0
            print("$frame ")
        end
    end

    return Rs, ts, gaps, times, extras
end

"""
    calc_pose_errors(Rs, ts, keypoints_data, object_id)

Compute angular error (degrees) and translation error (mm) of estimate.
"""
function calc_pose_errors(Rs, ts, keypoints_data, object_id)
    num_frames = length(keypoints_data["radii"])

    R_errs = Vector{Union{Float64, Missing}}(missing, num_frames)
    t_errs = Vector{Union{Float64, Missing}}(missing, num_frames)
    for frame = 1:num_frames
        if !(object_id in keys(keypoints_data["radii"][frame]))
            continue
        end
        gt = keypoints_data["gt_poses"][frame][object_id]
        R_gt = project2SO3(gt[1])
        t_gt = gt[2] / 1000. # [m]

        # errors
        R_err = roterror(R_gt, Rs[frame])
        t_err = norm(ts[frame] - t_gt)*1000

        # save
        R_errs[frame] = R_err
        t_errs[frame] = t_err
    end

    return R_errs, t_errs
end

"""
    calc_projection_errors(Rs, ts, keypoints_data, object_id, cadpath)

Compute 2D projection error given estimated rotation.
"""
function calc_projection_errors(Rs, ts, keypoints_data, object_id, cadpath)
    num_frames = length(keypoints_data["radii"])
    camK = keypoints_data["camK"]

    cad = FileIO.load(cadpath*(@sprintf "obj_%06d.ply" object_id))
    cad_m = GeometryBasics.Mesh(GeometryBasics.coordinates(cad)/1000, cad.faces)

    proj_errors = Vector{Union{Float64, Missing}}(missing, num_frames)
    for frame = 1:num_frames
        if !(object_id in keys(keypoints_data["radii"][frame]))
            continue
        end
        gt = keypoints_data["gt_poses"][frame][object_id]
        R_gt = project2SO3(gt[1])
        t_gt = gt[2] / 1000. # [m]

        # 2D projection error
        proj_error = 0.
        for coord in GeometryBasics.coordinates(cad_m)
            pixel_est = camK*(Rs[frame]*coord + ts[frame])
            pixel_est ./= pixel_est[3]
            pixel_gt = camK*(R_gt*coord + t_gt)
            pixel_gt ./= pixel_gt[3]
            proj_error += norm(pixel_est - pixel_gt)
        end
        proj_error /= length(GeometryBasics.coordinates(cad_m))

        # save
        proj_errors[frame] = proj_error

        if mod(frame, 100) == 0
            print("$frame ")
        end
    end

    return proj_errors
end