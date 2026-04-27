## Compute coverage on specified dataset
# 
# Lorenzo Shaikewitz, 8/12/2025

using Serialization
using Statistics
using LinearAlgebra
using DataFrames, TexTables
using Printf
using JuMP
using Distributions

using SimpleRotations
using PoseUncertaintySets

# parameters
α = 0.1
p = Inf

dataset = "lmo"
object_ids = [1,5,6,9,8,11,12] # omit 10 (eggbox)
# dataset = "ycbv"
# object_ids = [1:12;14;15] # omit 13, 16:21
# dataset = "cast"
# object_ids = [1]

compute_ellipse1 = true
compute_ellipse2 = !true
compute_hessian = !true

posepath = "../data/$dataset/pose_pnp2_$(round(Int,α*100))_$(string(p)).dat"
ellipsepath1 = "../data/$dataset/ellipse_slem_rotm_o1_$(round(Int,α*100))_$(string(p)).dat"
ellipsepath2 = "../data/$dataset/ellipse_slem_quat_o2_$(round(Int,α*100))_$(string(p)).dat"

# load data
keypoint_data, gt = load_keypoint_data(calibrate_lp, dataset; p=p, α=α)
if compute_ellipse1 || compute_ellipse2
    poses = deserialize(posepath)["solns"]
end
if compute_ellipse1
    ellipses1 = deserialize(ellipsepath1)["ellipses"]
end
if compute_ellipse2
    ellipses2 = deserialize(ellipsepath2)["ellipses"]
end

coverage = DataFrame()
for object_id in object_ids
    keypoints_covered = Dict()
    N = Dict()
    poses_covered = Dict()
    ellipses_cov1 = Dict()
    ellipses_cov2 = Dict()
    hessian_cov = Dict()
    for frame in sort(collect(keys(gt)))
        # remove missing keypoints
        if !(object_id in keys(keypoint_data["y"][frame]))
            continue
        end
        prob = get_problem(keypoint_data, object_id, frame)
        if size(prob.b,2) < 3
            continue
        end

        # extract ground truth
        R_gt = project2SO3(gt[frame][object_id][1])
        t_gt = gt[frame][object_id][2]

        ## keypoint coverage
        y_gt = prob.camK*(R_gt*prob.b .+ t_gt)
        y_gt = reduce(hcat, eachcol(y_gt) ./ y_gt[3,:])
        keypoints_covered[frame] = sum(norm.(eachcol(y_gt - prob.y),p) .<= prob.r)
        N[frame] = size(prob.y,2)

        ## pose uncertainty set coverage
        q_front, q_backproj = PoseUncertaintySets.uncertaintyset_linf_R(prob.y, prob.r, prob.b, prob.camK)
        x = [vec(R_gt); t_gt; 1]
        poses_covered[frame] = true
        for q in [q_front; q_backproj]
            poses_covered[frame] = poses_covered[frame] && (x'*[q.H  q.c;  q.c'  q.d]*x <= 0)
        end

        ## ellipsoid coverage
        if compute_ellipse1
            if !(frame in keys(poses[object_id][1]))
                continue
            end
            center = [vec(poses[object_id][1][frame]); poses[object_id][2][frame]]
            H = ellipses1[object_id][frame]
            x = [vec(R_gt); t_gt]
            ellipses_cov1[frame] = (x - center)'*H*(x - center) <= 1
        else
            ellipses_cov1[frame] = NaN
        end

        if compute_ellipse2
            if !(frame in keys(poses[object_id][1]))
                continue
            end
            center = [rotm2quat(poses[object_id][1][frame]); poses[object_id][2][frame]]
            H = ellipses2[object_id][frame]
            x = [rotm2quat(R_gt); t_gt]
            ellipses_cov2[frame] = (x - center)'*H*(x - center) <= 1
        else
            ellipses_cov2[frame] = NaN
        end

        if compute_hessian
            if !(frame in keys(poses[object_id][1]))
                continue
            end
            R₀ = poses[object_id][1][frame]
            center = [zeros(3); poses[object_id][2][frame]]
            H = gaussianjac(prob, [poses[object_id][1][frame], poses[object_id][2][frame]])
            x = [so3_log(R_gt * R₀'); t_gt]
            hessian_cov[frame] = (x - center)'*H*(x - center) <= quantile(Chisq(size(H,1)), 1-α)
        end
    end

    frames = collect(keys(keypoints_covered))
    (d::Dict)(k) = d[k] # make dictionary callable
    cov = DataFrame(dataset=dataset, frame=frames, id=object_id, 
                        keypoint=keypoints_covered.(frames), N=N.(frames), 
                        p=poses_covered.(frames), 
                        ellipsoid1=ellipses_cov1.(frames), ellipsoid2=ellipses_cov2.(frames),
                        hessian=hessian_cov.(frames))
    global coverage = [coverage; cov]
end

cov_keypoint = sum(coverage[:,"keypoint"]) / sum(coverage[:,"N"])
cov_p = mean(coverage[:,"p"])
cov_ellipsoid1 = mean(coverage[:,"ellipsoid1"])
cov_ellipsoid2 = mean(coverage[:,"ellipsoid2"])
cov_hessian = mean(coverage[:,"hessian"])

println("Coverage report for $dataset (α=$α)")
@printf "Keypoint coverage: %.2f%%\n" cov_keypoint*100
@printf "Pose set coverage: %.2f%%\n" cov_p*100
@printf "Ellipse1 coverage: %.2f%%\n" cov_ellipsoid1*100
@printf "Ellipse2 coverage: %.2f%%\n" cov_ellipsoid2*100
@printf "Hessian coverage: %.2f%%\n"  cov_hessian*100