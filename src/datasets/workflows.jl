## Full pipeline for a single frame
# All workflows that work!
#
# Lorenzo Shaikewitz, 7/23/2025

## Workflow 1: l2 norm
function l2_workflow(object_id, frame)
    # load data
    keypoint_data, gt = load_keypoint_data(calibrate_lp, p=2, α=0.1)
    prob = get_problem(keypoint_data, object_id, frame)

    # pose estimate (two options)
    # gaussianpose = PnP: better pose estimate but worse center, slower
    R, t, gap_pose, status_pose = gaussianpose(prob; silent=true, order=2)
    # R, t, gap_pose, status_pose = maxmarginpose(prob; silent=true)
    center = [vec(R); t]

    # bounding sphere
    # order 2 is generally tight, order 1 is not
    rad, status_sphere = bounding_sphere(center, prob; order=2, silent=true)

    # bounding ellipse
    # this is generally NOT tight
    H, status_slem = bounding_ellipse(center, prob; silent=true)

    # translation bounding box
    # order 2 is tight, order 1 is fast.
    bounds, gaps_bbox, status_bbox = refine_bbox(center, H, prob; mode=3, order=1, silent=true)

    # angular bounds: RPY
    # order 3 is much faster, but inequalities violate (not good bound)
    Δθs, gaps_rpy, status_rpy = angular_bounds_rpy(center, H, prob; silent=true, order=4)

    # angular bounds: 1 angle
    # TODO!
    

    # plotting
    P = [zeros(3,9) diagm(ones(3))]
    H_t = inv(P*pinv(H)*P')
    surf = ellipse_to_surf(H_t, center[end-2:end], 100)

    Plots.plot([center[end-2]],[center[end-1]],[center[end]], seriestype=:scatter, label="center")
    Plots.scatter!([0],[0],[0],label="camera")
    p2 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="s-lemma") #, msw=0.)

    plot_bbox!(p2, center, H_t, bounds; label="bbox")

    # Main.@infiltrate

    return p2
end


## Workflow 2: linf norm with R
function linfR_workflow(object_id, frame)
    # load data
    keypoint_data, gt = load_keypoint_data(calibrate_lp, p=Inf, α=0.1)
    prob = get_problem(keypoint_data, object_id, frame)

    # pose estimate (two options)
    # gaussianpose = PnP: better pose estimate but worse center, slower
    R, t, gap_pose, status_pose = gaussianpose(prob; silent=true, order=2)
    # R, t, gap_pose, status_pose = maxmarginpose(prob; silent=true)
    center = [vec(R); t]

    # bounding sphere
    # order 2 is generally tight, order 1 is not
    # order 1 is slow because we add many implied constraints
    rad, status_sphere = bounding_sphere(center, prob; order=2, R=true, silent=true)

    # bounding ellipse
    # this is generally NOT tight (but it solves without SLOW_PROGRESS)
    H, status_slem = bounding_ellipse(center, prob; silent=true)

    # translation bounding box
    # order 2 is tight, order 1 is fast.
    # neither works particularly well.
    bounds, gaps_bbox, status_bbox = refine_bbox(center, H, prob; R=true, mode=1, order=2, silent=true)

    # angular bounds: RPY
    # order 3 is much faster, but inequalities violate (not good bound)
    Δθs, gaps_rpy, status_rpy = angular_bounds_rpy(center, H, prob; silent=true, order=4)

    # angular bounds: 1 angle
    # TODO!
    

    # plotting
    P = [zeros(3,9) diagm(ones(3))]
    H_t = inv(P*pinv(H)*P')
    surf = ellipse_to_surf(H_t, center[end-2:end], 100)
    
    Plots.plot([center[end-2]],[center[end-1]],[center[end]], seriestype=:scatter, label="center")
    Plots.scatter!([0],[0],[0],label="camera")
    p2 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="s-lemma") #, msw=0.)

    plot_bbox!(p2, center, H_t, bounds; label="bbox")

    # Main.@infiltrate

    return p2
end


## Workflow 3: linf norm with q
function linfq_workflow(object_id, frame)
    # load data
    keypoint_data, gt = load_keypoint_data(calibrate_lp, p=Inf, α=0.1)

    if !(object_id in keys(keypoint_data["y"][frame]))
        return -1
    end
    print("$frame ")
    prob = get_problem(keypoint_data, object_id, frame)

    if length(prob.y) < 3
        return -1
    end

    # pose estimate (two options)
    # gaussianpose = PnP: better pose estimate but worse center, slower
    R, t, gap_pose, status_pose = gaussianpose(prob; silent=true, order=2)
    # R, t, gap_pose, status_pose = maxmarginpose(prob; silent=true)
    center = [rotm2quat(R); t]

    # bounding sphere
    # order 2 is generally tight, order 1 is not
    # rad, status_sphere = bounding_sphere(center, prob; order=2, R=false, silent=true)

    # bounding ellipse
    # Tight at second order! Fails for first order
    # TODO: check tightness cert
    H, gap_slem, status_slem = bounding_ellipse_quat(center, prob; order=2, silent=true)
    return gap_slem

    # translation bounding box
    # order 2 is tight and not terribly slow
    bounds, gaps_bbox, status_bbox = refine_bbox(center, H, prob; R=false, mode=1, order=2, silent=true)

    # angular bounds: RPY
    # ridiculously slow and needs order at least 6
    # also all bounds are 90 deg
    # Δθs, gaps_rpy, status_rpy = PoseUncertaintySets.angular_bounds_rpy_quat(center, H, prob; silent=false, order=6)

    # angular bounds: 1 angle
    # ellipse only version works better than all constraints (which gives slow prog)
    Δθ, status_ang = angular_bounds_quat(center, H; silent=true, order=1) # just reason over ellipse
    # Δθ, status_ang = angular_bounds_quat(center, H, prob; silent=false, order=3) # imposes backproj / chirality too
    

    # plotting
    P = [zeros(3,9) diagm(ones(3))]
    H_t = inv(P*pinv(H)*P')
    surf = ellipse_to_surf(H_t, center[end-2:end], 100)
    
    Plots.plot([center[end-2]],[center[end-1]],[center[end]], seriestype=:scatter, label="center")
    Plots.scatter!([0],[0],[0],label="camera")
    p2 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="s-lemma") #, msw=0.)

    plot_bbox!(p2, center, H_t, bounds; label="bbox")

    # Main.@infiltrate

    return p2
end