## Functions to load data
# Lorenzo Shaikewitz, 6/13/2025

struct Problem
    p               # lp norm
    frame::Int
    object_id::Int
    r               # conformal radii
    y               # pixel keypoints
    b               # 3D keypoints
    camK            # camera calibration matrix
end

function get_problem(keypoint_data, object_id, frame)
    camK = keypoint_data["K"]

    r = keypoint_data["r"][frame][object_id]
    y = keypoint_data["y"][frame][object_id]
    b = keypoint_data["b"][object_id]

    # eliminate missing measurements
    y = y[1:2,r .>= 0]
    y = [y[1:2,:]; ones(size(y,2))']
    b = b[:, r .>= 0]
    r = r[r .>= 0]

    return Problem(keypoint_data["p"], frame, object_id, r, y, b, camK)
end



"""
Load keypoints and calibrate
"""
function load_keypoint_data_lmo(cal_fn=calibrate_lp; p=2, α=0.1, path_kpts3d="../data/kpts3d.json", 
        parent_cal="../data/bop/lmo/test_bop19/000002", parent_test="../data/bop/lmo/test_all/000002",
        detections_cal_path="../data/detections_lmo_cal.json", detections_test_path="../data/detections_lmo_test.json")
    
    kpt_lib = load_kpt_lib(path_kpts3d)

    camK = load_K(parent_test)
    gt_cal = load_gt(parent_cal)
    gt_test = load_gt(parent_test)

    kpts_cal = load_raw_keypoints(detections_cal_path)
    kpts_test = load_raw_keypoints(detections_test_path)

    for key in keys(gt_cal)
        delete!(gt_test, key)
        delete!(kpts_test, key)
    end

    # calibrate!
    radii, scores, ns = cal_fn(kpts_cal, gt_cal, camK, kpts_test, kpt_lib, p, α)

    data = Dict("K"=>camK, "r"=>radii, "y"=>kpts_test, "b"=>kpt_lib, "p"=>p)
    return data, gt_test
end

function load_keypoint_data_ycbv(cal_fn=calibrate_lp; p=2, α=0.1, path_kpts3d="", 
        cal_n=150, parent_test="", detections_test_path="")
    
    kpt_lib = load_kpt_lib(path_kpts3d, "ycbv")

    camK = load_K(parent_test*"/"*readdir(parent_test)[1])

    gt_test = Dict()
    kpts_test = Dict()
    for folder in readdir(parent_test)
        gt_test_f = load_gt(parent_test*"/$folder")
        foldernum = parse(Int, folder)
        gt_test_f = Dict(foldernum*10000 + k => v for (k,v) in pairs(gt_test_f))
        merge!(gt_test, gt_test_f)

        kpts_test_f = load_raw_keypoints(detections_test_path)
        kpts_test_f = Dict(foldernum*10000 + k => v for (k,v) in pairs(kpts_test_f))
        merge!(kpts_test, kpts_test_f)
    end

    # select random frames to become cal
    frames_cal = sample(collect(keys(gt_test)), cal_n; replace=false)
    gt_cal = Dict(frames_cal .=> gt_test.(frames_cal))
    kpts_cal = Dict(frames_cal .=> kpts_test.(frames_cal))

    # TODO: remove from kpts_test the objs that aren't in gt_cal.

    # remove form 
    for key in keys(gt_cal)
        delete!(gt_test, key)
        delete!(kpts_test, key)
    end

    # calibrate!
    radii, scores, ns = cal_fn(kpts_cal, gt_cal, camK, kpts_test, kpt_lib, p, α)

    data = Dict("K"=>camK, "r"=>radii, "y"=>kpts_test, "b"=>kpt_lib, "p"=>p)
    return data, gt_test
end

"""
Load keypoints for one dataset
"""
function load_keypoint_data(cal_fn, dataset; p=2, α=0.1)
    path_kpts3d = "../data/$dataset/kpts3d.json"
    detections_test_path = "../data/$dataset/detections_test.json"

    if dataset == "lmo"
        parent_test = "../data/$dataset/test/000002"
        parent_cal = "../data/$dataset/cal/000002"
        detections_cal_path = "../data/$dataset/detections_cal.json"
        return load_keypoint_data_lmo(cal_fn; p=p, α=α, path_kpts3d=path_kpts3d,
                parent_cal=parent_cal, parent_test=parent_test, 
                detections_cal_path=detections_cal_path, detections_test_path=detections_test_path)
    elseif dataset == "ycbv"
        parent_test = "../data/$dataset/test"
        cal_n = 150
        return load_keypoint_data_ycbv(cal_fn; p=p, α=α, path_kpts3d=path_kpts3d,
            parent_test=parent_test, cal_n=cal_n, detections_test_path=detections_test_path)
    end
end


"""
    calibrate_lp(cal_kpts, cal_gt, test_kpts, kpt_lib, p=2, α=0.1)

Calibrate with uncertainty-weighted lp loss at confidence `α`.
"""
function calibrate_lp(kpts_cal, gt_cal, camK, kpts_test, kpt_lib, p=2, α=0.1; conf_thresh=0.0)
    scores = Dict(k => Any[] for k in keys(kpt_lib))

    for (img_id, kpts_cal_all) in kpts_cal
        for (obj, kpts_cal_obj) in kpts_cal_all
            if !(obj in keys(gt_cal[img_id]))
                continue
            end
            # calc gt keypoints
            R = gt_cal[img_id][obj][1]
            t = gt_cal[img_id][obj][2]
            kpts_gt = camK*(R*kpt_lib[obj] .+ t)
            kpts_gt = reduce(hcat,eachcol(kpts_gt[1:2,:]) ./ kpts_gt[3,:])

            # compute scores
            dist = norm.(eachcol(kpts_gt - kpts_cal_obj[1:2,:]), p)
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
        
    #     dist = norm.(eachcol(kpts_gt - kpts_test[img_id][5][1:2,:]), p)
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