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
    end
    kpts_test = load_raw_keypoints(detections_test_path)

    # TODO: better way to calibrate (maybe try synthetic data?)
    # at minimum we should use a fixed calibration set.
    frames_cal = sample(collect(keys(gt_test)), cal_n; replace=false)
    gt_cal = Dict(frames_cal .=> gt_test.(frames_cal))
    kpts_cal = Dict(frames_cal .=> kpts_test.(frames_cal))

    # remove cal from test
    for key in keys(gt_cal)
        delete!(gt_test, key)
        delete!(kpts_test, key)
    end

    # calibrate!
    radii, scores, ns = cal_fn(kpts_cal, gt_cal, camK, kpts_test, kpt_lib, p, α)

    data = Dict("K"=>camK, "r"=>radii, "y"=>kpts_test, "b"=>kpt_lib, "p"=>p)
    return data, gt_test
end

function load_keypoint_data_cast(cal_fn, frames_cal, detections_path; p=2, α=0.1)
    
    dets=JSON.parsefile(detections_path)
    camK = convert.(Float64,reduce(hcat,dets[1]["K_mat"])')
    kpt_lib = Dict(1 => reduce(hcat,dets[1]["interp_cad_keypoints"][1:7]) / 1000.)

    kpts_test = Dict()
    gt_test = Dict()
    for (i,d) in enumerate(dets)
        kpts_test[i] = Dict(1=>convert.(Float64,reduce(hcat,d["est_pixel_keypoints"])))
        T = convert.(Float64,reduce(hcat,d["gt_teaser_pose"]))'
        gt_test[i] = Dict(1=>(T[1:3,1:3], T[1:3,4]/1000.))
    end


    # extract some frames to calibrate
    # frames_cal = sample(collect(keys(gt_test)), cal_n; replace=false)
    gt_cal = Dict(frames_cal .=> gt_test.(frames_cal))
    kpts_cal = Dict(frames_cal .=> kpts_test.(frames_cal))

    # remove cal from test
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
function load_keypoint_data(cal_fn, dataset="lmo"; p=2, α=0.1)
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
    elseif dataset == "cast"
        # generated by uniform random sample without replacement
        frames_cal = [1249, 35, 1460, 799, 322, 1147, 769, 1163, 481, 420, 1160, 950, 1879, 40, 542, 1870, 1675, 762, 755, 747, 1716, 1246, 1533, 1245, 1293, 1575, 1868, 533, 1895, 968, 823, 1130, 1310, 784, 597, 203, 1429, 19, 1885, 145, 1370, 1105, 647, 622, 1150, 565, 1155, 1757, 665, 1047, 740, 1459, 342, 125, 1222, 1053, 1331, 1595, 401, 276, 477, 1678, 1647, 637, 210, 173, 503, 982, 1131, 79, 114, 913, 1100, 1433, 1592, 863, 1353, 1059, 350, 1268, 882, 1706, 301, 482, 632, 1350, 969, 87, 137, 354, 1500, 1192, 279, 1018, 1810, 1092, 1829, 709, 824, 1320, 511, 71, 1881, 1475, 691, 334, 1115, 854, 928, 470, 1141, 1014, 605, 510, 315, 999, 1458, 1773, 768, 1761, 1856, 560, 562, 890, 706, 415, 904, 1504, 5, 1299, 979, 398, 680, 1427, 652, 714, 1015, 1567, 1116, 831, 798, 1546, 1376, 1790, 1713, 1101, 837, 1480, 1894, 370, 243, 1484, 1373, 109, 178, 490, 1103, 1584, 1156, 1332, 1007, 775, 1337, 670, 216, 1386, 1448, 744, 981, 513, 82, 1167, 827, 456, 1189, 673, 1198, 512, 1651, 1354, 1527, 1841, 1136, 587, 939, 374, 1701, 83, 726, 1663, 1840, 986, 385, 448, 1781, 1820, 548, 891, 1166, 57]
        return load_keypoint_data_cast(cal_fn, frames_cal, detections_test_path; p=p, α=α)
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