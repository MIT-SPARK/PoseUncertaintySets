## Functions to get pose bounds on datasets
# Lorenzo Shaikewitz, 6/17/2025

"""
    dataset_slem_bounds(keypoint_data, pose_data, object_id)

Get ellipse via S-lemma and refine to ang and trans bounds.

# Returns
- `Hs`: bounding ellipse matrices (Dict of matrices)
- `Δθs`: angular error bounds (Dict of vectors)
- `Δts`: translation error bounds (Dict of matrices)
- `statuses`: runtime of each stage (Dict of vectors)
- `times`: runtime of each stage (Dict of vectors)
"""
function dataset_slem_bounds(keypoint_data, pose_data, object_id)
    # setup
    camK = keypoint_data["K"]
    num_frames = length(keys(pose_data[object_id][1]))

    
    Hs = Dict{Int, Any}()
    Δθs = Dict{Int, Any}()
    Δts = Dict{Int, Any}()
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

        # load pose data
        R_est = pose_data[object_id][1][frame]
        t_est = pose_data[object_id][2][frame]
        center = [vec(R_est); t_est]

        # S-Lemma
        out = @timed bounding_ellipse(center, y, r, b, camK; solver=Mosek.Optimizer, silent=true)
        H, status_s = out.value
        time_s = out.time - out.compile_time
        # angular bounds
        out = @timed angular_bounds(center, H; silent=true, order=2)
        Δθ, status_r, gaps_r = out.value
        time_r = out.time - out.compile_time
        # translation bounds
        out = @timed refine_bbox(center, H, y, r, b, camK; mode=3, order=1, silent=true)
        Δt, gaps_t, status_t = out.value
        time_t = out.time - out.compile_time
        
        # save
        Hs[frame] = H
        Δθs[frame] = Δθ
        Δts[frame] = Δt
        statuses[frame] = [status_s; status_r; vec(status_t)]
        times[frame] = [time_s; time_r; time_t]
        gaps[frame] = [gap_s; gaps_r; vec(gaps_t)]

        # if isdefined(Main, :Infiltrator) Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__) end # 🚨 INFILTRATOR 🚨

        if mod(frame, 10) == 0
            print("$frame ")
        end
    end

    return Hs, Δθs, Δts, statuses, times, gaps
end



"""
    dataset_slem_bounds_quat(keypoint_data, pose_data, object_id)

Get ellipse via S-lemma and refine to ang and trans bounds (quaternion version).

# Returns
- `Hs`: bounding ellipse matrices (Dict of matrices)
- `Δθs`: angular error bounds (Dict of vectors)
- `Δts`: translation error bounds (Dict of matrices)
- `statuses`: status of each stage (Dict of vectors)
- `times`: runtime of each stage (Dict of vectors)
- `gaps`: subopt gap of each stage (Dict of vectors)
"""
function dataset_slem_bounds_quat(keypoint_data, pose_data, object_id)
    # setup
    camK = keypoint_data["K"]
    num_frames = length(keys(pose_data[object_id][1]))

    
    Hs = Dict{Int, Any}()
    Δθs = Dict{Int, Any}()
    Δts = Dict{Int, Any}()
    statuses = Dict{Int, Any}()
    times = Dict{Int, Any}()
    gaps = Dict{Int, Any}()

    println("Starting $num_frames frames...")
    for frame in sort(collect(keys(pose_data[object_id][1])))

        # frame-specific data
        prob = get_problem(keypoint_data, object_id, frame)

        # load pose data
        R_est = pose_data[object_id][1][frame]
        t_est = pose_data[object_id][2][frame]
        center = [rotm2quat(R_est); t_est]

        # S-Lemma
        out = @timed bounding_ellipse_quat(center, prob; order=2, silent=true)
        H, gap_s, status_s = out.value
        time_s = out.time - out.compile_time
        # angular bounds
        # TODO: this only approximates ellipse for angular error
        out = @timed angular_bounds_quat(center, H; silent=true, order=1)
        Δθ, status_r, gaps_r = out.value
        time_r = out.time - out.compile_time
        # translation bounds
        # TODO: this only approximates ellipse with bbox
        out = @timed refine_bbox(center, H, prob; R=false, mode=3, order=1, silent=true)
        Δt, gaps_t, status_t = out.value
        time_t = out.time - out.compile_time
        
        # save
        Hs[frame] = H
        Δθs[frame] = Δθ
        Δts[frame] = Δt
        statuses[frame] = [status_s; status_r; vec(status_t)]
        times[frame] = [time_s; time_r; time_t]
        gaps[frame] = [gap_s; gaps_r; vec(gaps_t)]

        # if isdefined(Main, :Infiltrator) Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__) end # 🚨 INFILTRATOR 🚨

        if mod(frame, 10) == 0
            print("$frame ")
        end
    end

    return Hs, Δθs, Δts, statuses, times, gaps
end


"""
    dataset_ransag_bounds(keypoint_data, pose_data, object_id)

Compute uncertainty bounds using the semidefinite relaxation from RANSAG.

"Object Pose Estimation with Statistical Guarantees: Conformal Keypoint Detection and Geometric Uncertainty Propagation"
by Heng Yang and Marco Pavone

# Returns
- `Δθs`: angular error bounds (Dict of Float64)
- `Δts`: translation error bounds (Dict of Float64)
- `statuses`: runtime of each stage (Dict of vectors)
- `times`: runtime of each stage (Dict of Float64)
"""
function dataset_ransag_bounds(keypoint_data, pose_data, object_id)
    # setup
    camK = keypoint_data["K"]
    num_frames = length(keys(pose_data[object_id][1]))

    Δθs = Dict{Int, Float64}()
    Δts = Dict{Int, Float64}()
    statuses = Dict{Int, Any}()
    times = Dict{Int, Float64}()
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

        # load pose data
        R_est = pose_data[object_id][1][frame]
        t_est = pose_data[object_id][2][frame]
        center = [vec(R_est); t_est]

        # PURSE bounds
        out = @timed purse_bounds(center, y, r, b, camK; order=2, silent=true)
        trans_bound, trans_gap, ang_bound, ang_gap, status = out.value
        time = out.time - out.compile_time
        
        # save
        Δθs[frame] = ang_bound
        Δts[frame] = trans_bound
        statuses[frame] = status
        times[frame] = time
        gaps[frame] = [trans_gap; ang_gap]

        # if isdefined(Main, :Infiltrator) Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__) end # 🚨 INFILTRATOR 🚨

        if mod(frame, 10) == 0
            print("$frame ")
        end
    end

    return Δθs, Δts, statuses, times, gaps
end




"""
    dataset_slem(keypoint_data, pose_data, object_id)

Get ellipse via S-lemma.

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