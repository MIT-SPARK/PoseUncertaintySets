## Functions to get pose bounds on datasets
# Lorenzo Shaikewitz, 6/17/2025

"""
    dataset_ransag_bounds(keypoint_data, pose_data, object_id)

Compute angular and translational bounds on dataset using RANSAG.

Reference:
"Object Pose Estimation with Statistical Guarantees: Conformal Keypoint Detection and Geometric Uncertainty Propagation"
by Heng Yang and Marco Pavone

# Returns
- `Δθs`: angular error bounds (Dict of Float64)
- `Δts`: translation error bounds (Dict of Float64)
- `statuses`: runtime of each stage (Dict of vectors)
- `times`: runtime of each stage (Dict of Float64)
"""
function dataset_ransag_bounds(keypoint_data, pose_data, object_id; order=2)
    # setup
    camK = keypoint_data["K"]
    num_frames = length(keys(pose_data[object_id][1]))

    Δθs = Dict{Int, Float64}()
    Δts = Dict{Int, Float64}()
    statuses_t = Dict{Int, MOI.TerminationStatusCode}()
    statuses_a = Dict{Int, MOI.TerminationStatusCode}()
    times = Dict{Int, Float64}()
    gaps_t = Dict{Int, Float64}()
    gaps_a = Dict{Int, Float64}()

    println("Starting $num_frames frames...")
    for frame in sort(collect(keys(pose_data[object_id][1])))

        # frame-specific data
        r = keypoint_data["r"][frame][object_id]
        y = keypoint_data["y"][frame][object_id]
        b = keypoint_data["b"][object_id]

        # eliminate missing measurements
        y = y[1:2,r .>= 0]
        y = [y[1:2,:]; ones(size(y,2))']
        b = b[:, r .>= 0]
        r = r[r .>= 0]

        # load pose data
        R_est = pose_data[object_id][1][frame]
        t_est = pose_data[object_id][2][frame]
        center = [vec(R_est); t_est]

        # PURSE bounds
        out = @timed purse_bounds(center, y, r, b, camK; order=order, silent=true)
        trans_bound, trans_gap, ang_bound, ang_gap, status = out.value
        time = out.time - out.compile_time
        
        # save
        Δθs[frame] = ang_bound
        Δts[frame] = trans_bound
        statuses_t[frame] = status[1]
        statuses_a[frame] = status[2]
        times[frame] = time
        gaps_t[frame] = trans_gap
        gaps_a[frame] = ang_gap

        # if isdefined(Main, :Infiltrator) Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__) end # 🚨 INFILTRATOR 🚨

        if mod(frame, 10) == 0
            print("$frame ")
        end
    end
    statuses = (statuses_t, statuses_a)
    gaps = (gaps_t, gaps_a)

    return Δθs, Δts, statuses, times, gaps
end




"""
    dataset_slem(keypoint_data, pose_data, object_id)

Compute uncertainty ellipsoids on dataset using S-Lemma.

# Returns
- `Hs`: bounding ellipse matrices (Dict of matrices)
- `statuses`: runtime of each stage (Dict of vectors)
- `times`: runtime of each stage (Dict of vectors)
"""
function dataset_slem(keypoint_data, pose_data, object_id; order=1, quat=false)
    # setup
    camK = keypoint_data["K"]
    num_frames = length(keys(pose_data[object_id][1]))

    
    Hs = Dict{Int, Any}()
    statuses = Dict{Int, Any}()
    times = Dict{Int, Any}()
    gaps = Dict{Int, Any}()

    println("Starting $num_frames frames...")
    for frame in sort(collect(keys(pose_data[object_id][1])))

        # frame-specific data
        r = keypoint_data["r"][frame][object_id]
        y = keypoint_data["y"][frame][object_id]
        b = keypoint_data["b"][object_id]

        # eliminate missing measurements
        y = y[1:2,r .>= 0]
        y = [y[1:2,:]; ones(size(y,2))']
        b = b[:, r .>= 0]
        r = r[r .>= 0]

        if length(r) < 3
            continue
        end

        # load pose data
        R_est = pose_data[object_id][1][frame]
        t_est = pose_data[object_id][2][frame]
        if quat
            center = [rotm2quat(R_est); t_est]
        else
            center = [vec(R_est); t_est]
        end

        # S-Lemma
        if quat
            out = @timed bounding_ellipse_quat(center, y, r, b, camK; order=order, silent=true)
        else
            out = @timed bounding_ellipse(center, y, r, b, camK; order=order, silent=true)
        end
        H, gap_s, status_s = out.value
        time_s = out.time - out.compile_time
        
        # save
        Hs[frame] = H
        statuses[frame] = status_s
        times[frame] = time_s
        gaps[frame] = gap_s

        if mod(frame, 10) == 0
            print("$frame ")
        end
    end

    return Hs, statuses, times, gaps
end


"""
    dataset_slem_separated(keypoint_data, pose_data, object_id)

Compute uncertainty ellipsoids on dataset using S-Lemma
WITH SEPARATE ROTATION AND TRANSLATION OBJECTIVES (two runs)

# Returns
- `Hs`: bounding ellipse matrices (Dict of matrices)
- `statuses`: runtime of each stage (Dict of vectors)
- `times`: runtime of each stage (Dict of vectors)
"""
function dataset_slem_separated(keypoint_data, pose_data, object_id; order=1, quat=false)
    # setup
    camK = keypoint_data["K"]
    num_frames = length(keys(pose_data[object_id][1]))

    
    Hs = Dict{Int, Any}()
    statuses_r = Dict{Int, Any}()
    statuses_t = Dict{Int, Any}()
    times = Dict{Int, Any}()

    println("Starting $num_frames frames...")
    for frame in sort(collect(keys(pose_data[object_id][1])))

        # frame-specific data
        r = keypoint_data["r"][frame][object_id]
        y = keypoint_data["y"][frame][object_id]
        b = keypoint_data["b"][object_id]

        # eliminate missing measurements
        y = y[1:2,r .>= 0]
        y = [y[1:2,:]; ones(size(y,2))']
        b = b[:, r .>= 0]
        r = r[r .>= 0]

        if length(r) < 3
            continue
        end

        # load pose data
        R_est = pose_data[object_id][1][frame]
        t_est = pose_data[object_id][2][frame]
        if quat
            center = [rotm2quat(R_est); t_est]
        else
            center = [vec(R_est); t_est]
        end

        # S-Lemma: rotations
        if quat
            out = @timed bounding_ellipse_quat(center, y, r, b, camK; order=order, silent=true)
            H, status_r, status_t = out.value
            time_s = out.time - out.compile_time
        else
            # rotations
            out = @timed bounding_ellipse_separated(center, y, r, b, camK, [1.; 0]; order=order, silent=true)
            H_r, _, status_r = out.value
            time_s = out.time - out.compile_time
            H_r = H_r[1:9,1:9]

            # translations
            out = @timed bounding_ellipse_separated(center, y, r, b, camK, [0; 1.]; order=order, silent=true)
            H_t, _, status_t = out.value
            time_s += out.time - out.compile_time
            H_t = H_t[10:12,10:12]
            H = [H_r zeros(9,3); zeros(3,9) H_t]
        end
        
        
        # save
        Hs[frame] = H
        statuses_r[frame] = status_r
        statuses_t[frame] = status_t
        times[frame] = time_s

        if mod(frame, 10) == 0
            print("$frame ")
        end
    end

    statuses = (statuses_r, statuses_t)

    return Hs, statuses, times
end