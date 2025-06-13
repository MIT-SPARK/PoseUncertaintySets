## Functions to load data
# Lorenzo Shaikewitz, 6/13/2025

function load_keypoint_data(cal_fn=calibrate_l2; path_kpts3d="../data/kpts3d.json", 
        parent_cal="../data/bop/lmo/test_bop19/000002", parent_test="../data/bop/lmo/test_all/000002",
        detections_cal_path="../data/detections_lmo_cal.json", detections_test_path="../data/detections_lmo_test.json")
    
    kpt_lib = load_kpt_lib(path_kpts3d)

    camK = load_K(parent_test)
    gt_cal = load_gt(parent_cal)
    gt_test = load_gt(parent_test)

    kpts_cal = load_raw_keypoints(detections_cal_path)
    kpts_test = load_raw_keypoints(detections_test_path)

    # calibrate!
    radii, scores, ns = cal_fn(kpts_cal, gt_cal, camK, kpts_test, kpt_lib)

    data = Dict("K"=>camK, "r"=>radii, "y"=>kpts_test, "b"=>kpt_lib)
    return data, gt_test
end


"""
    calibrate_l2(cal_kpts, cal_gt, test_kpts, kpt_lib)

Calibrate with uncertainty-weighted l2 loss.
"""
function calibrate_l2(kpts_cal, gt_cal, camK, kpts_test, kpt_lib, α=0.1; conf_thresh=0.0)
    scores = Dict(k => Any[] for k in keys(kpt_lib))

    for (img_id, kpts_cal_all) in kpts_cal
        for (obj, kpts_cal_obj) in kpts_cal_all

            # calc gt keypoints
            R = gt_cal[img_id][obj][1]
            t = gt_cal[img_id][obj][2]
            kpts_gt = camK*(R*kpt_lib[obj] .+ t)
            kpts_gt = reduce(hcat,eachcol(kpts_gt[1:2,:]) ./ kpts_gt[3,:])

            # calibrate
            dist = norm.(eachcol(kpts_gt - kpts_cal_obj[1:2,:]))
            score = dist .* kpts_cal_obj[3,:]

            # throw out low confidence detections
            score[kpts_cal_obj[3,:] .<= conf_thresh] .= -1
            
            push!(scores[obj], score)
        end
    end

    # calibrate against ground truth
    quantiles = Dict()
    ns = Dict()
    for (obj, s) in scores
        if length(s) == 0
            continue
        end
        s = reduce(hcat, s)
        s = sort(s,dims=2)
        n = sum.(eachrow(s .>= 0))
        q = []
        for (idx,row) in enumerate(eachrow(s))
            row = row[row .>= 0]
            push!(q, row[ceil(Int,(1-α)*n[idx])])
        end
        quantiles[obj] = q
        ns[obj] = n
    end

    # compute conformal radii
    radii = Dict(k => Dict() for k in keys(kpts_test))
    for (img_id, kpts_all) in kpts_test
        for (obj, kpts_obj) in kpts_all
            # invert calibration (divide by confidence)
            radius = quantiles[obj] ./ kpts_obj[3,:]

            # throw out low confidence
            radius[kpts_obj[3,:] .<= conf_thresh] .= -1

            radii[img_id][obj] = radius
        end
    end

    # VERIFY (set kpts_test = kpts_cal)
    # cov5 = zeros(8)
    # for (img_id, r) in radii
    #     if !(5 in keys(r))
    #     continue
    #     end
        
    #     R = gt_cal[img_id][5][1]
    #     t = gt_cal[img_id][5][2]
    #     kpts_gt = camK*(R*kpt_lib[5] .+ t)
    #     kpts_gt = reduce(hcat,eachcol(kpts_gt[1:2,:]) ./ kpts_gt[3,:])
        
    #     dist = norm.(eachcol(kpts_gt - kpts_test[img_id][5][1:2,:]))
    #     cov5 += dist .< r[5]
    # end
    # display(cov5 ./ ns[5])

    # if isdefined(Main, :Infiltrator) Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__) end # 🚨 INFILTRATOR 🚨

    return radii, scores, ns
end

"""
    load_K(parent)

Load camera calibration matrix from BOP (first one).
"""
function load_K(parent)
    d=JSON.parsefile(parent*"/scene_camera.json")
    for (_,val) in d
        return convert.(Float64,reshape(val["cam_K"],3,3)')
    end
end

"""
    load_gt(parent)

Load ground truth `R`, `t` [m] from BOP format.
"""
function load_gt(parent)
    gt = JSON.parsefile(parent*"/scene_gt.json")
    gt = Dict(parse(Int,k)=>Dict(v2["obj_id"]=>(convert.(Float64,reshape(v2["cam_R_m2c"],3,3)'), v2["cam_t_m2c"] ./ 1000.) for (k2,v2) in pairs(v)) for (k,v) in pairs(gt))
    return gt
end

"""
    load_raw_keypoints(detections_path)

Load raw keypoints and convert to array format
"""
function load_raw_keypoints(detections_path)
    kpts = JSON.parsefile(detections_path)
    kpts = Dict(parse(Int,k)=>Dict(parse(Int,k2)=>reduce(hcat, v2)' for (k2,v2) in pairs(v)) for (k,v) in pairs(kpts))
end

"""
    load_kpt_lib(path, dataset="lmo")

Load 3D keypoint library.
"""
function load_kpt_lib(path, dataset="lmo")
    kpt_lib = JSON.parsefile(path)
    kpt_lib = Dict(parse(Int,k)=>reduce(hcat,v) ./ 1000. for (k,v) in pairs(kpt_lib[dataset])) # [m]
end