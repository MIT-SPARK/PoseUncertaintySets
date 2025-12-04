## Make video of detections
# Lorenzo Shaikewitz, 8/14/2025

import Plots, JSON
using Serialization
using DataFrames
using PoseUncertaintySets
using LinearAlgebra
using SimpleRotations
using Printf
import FileIO, GeometryBasics, Images
using ColorSchemes, Colors

# settings
# dataset = "ycbv"
# object_id = 4
dataset = "cast"
object_id = 1
α = 0.1
plot_gt = false
plot_keypoints = !true
plot_frame = !true

## load data and calibrate
# paths
parent = "../data/cast_video"
image_parent = "../data/$dataset/test"
cadpath = "../data/$dataset/models_eval/"
posepath = "../data/$dataset/pose_pnp2_$(round(Int,α*100))_2.dat"
ellipsepath = "../data/$dataset/ellipse_slem_quat_o2_$(round(Int,α*100))_Inf.dat"

# keypoints
keypoint_data, gt = load_keypoint_data(calibrate_lp, dataset; p=Inf, α=α)
if dataset == "cast"
    data_cast = JSON.parsefile("../data/$dataset/detections_test.json")
end

# load poses, ellipses
poses = deserialize(posepath)["solns"]
H_slem = deserialize(ellipsepath)["ellipses"]
# load cad
cad = FileIO.load(cadpath*(@sprintf "obj_%06d.ply" object_id))
cad_m = GeometryBasics.Mesh(GeometryBasics.coordinates(cad)/1000, cad.faces)

## Plot
for frame in sort(collect(keys(keypoint_data["y"])))
    # skip bad frames
    if !(object_id in keys(keypoint_data["y"][frame]))
        continue
    end
    if !(frame in keys(poses[object_id][1]))
        continue
    end
    if !(frame in keys(H_slem[object_id]))
        continue
    end

    # load and plot the image
    prob = get_problem(keypoint_data, object_id, frame)
    if dataset != "cast"
        p, img = plot_image(image_parent, frame)
    else
        p, img = plot_image(image_parent, frame; data=data_cast)
    end

    # optional: plot keypoints
    if plot_keypoints
        plot_keypoints!(p, prob.y, prob.r, p=Inf)
        if plot_gt
            plot_3d_keypoints!(p, gt[frame][object_id], prob.b, prob.camK)
        end
    end

    # get pose and ellipsoid
    R_est = poses[object_id][1][frame]
    t_est = poses[object_id][2][frame]
    H = H_slem[object_id][frame]

    # can't plot cad model projection for CAST
    if dataset != "cast"
        # get mask for pose
        seg = get_mask(img, R_est, t_est, cad_m, prob.camK)

        # project uncertainty ellipsoid to axang space
        P = [zeros(6)  I(6)]
        Ph = [Ω2(rotm2quat(R_est))  zeros(4,3);  zeros(3,4)   I(3)]
        H_adj = inv(P*pinv(Ph'*H*Ph)*P')
        # grid on boundary of joint ellipse
        surf = ellipse_to_surf_nd(H_adj, [zeros(3); t_est], 1e4)
        surf_t = eachcol(surf[end-2:end,:])
        # convert axang back to rotations (max in case of bad numerics)
        surf_R = [quat2rotm([sqrt(max(0,1 - ωsinθ2'*ωsinθ2)); ωsinθ2])*R_est for ωsinθ2 in eachcol(surf[1:3,:])]

        # get mask for each sample from ellipsoid
        sample_mask = zeros(Bool, size(img))
        for (R, t) in zip(surf_R, surf_t)
            mask = get_lazy_mask(img, R, t, cad_m, prob.camK)
            sample_mask .|= mask
        end
        img_mask = Images.RGBA.(copy(img)).*0
        img_mask[sample_mask] .= Images.RGBA(0,1,1, 0.5)
        Plots.plot!(img_mask)

        # plot outline of pose estimate
        plot_outline!(p, img, seg)
    else
        # plot pose as frame
        plot_frame!(p, (R_est, t_est), prob.camK; gt=false)

        # just plot translation uncertainty ellipsoid
        (Ht, Hθ), (boundst, boundsθ) = project_ellipse(H, R_est)
        surf = ellipse_to_surf(Ht, t_est, 500)

        # project 3D -> 2D
        surf_px = [prob.camK] .* eachcol(surf)
        depths = reduce(hcat, surf_px)[3,:]
        surf_px = reduce(hcat, [s[1:2] / s[3] for s in surf_px])
        surf_px = round.(Int,surf_px)
        # clamp
        surf_px[1,:] = clamp!(surf_px[1,:], 1, size(img)[2])
        surf_px[2,:] = clamp!(surf_px[2,:], 1, size(img)[1])

        # draw on mask
        dmin, dmax = extrema(depths)
        norm_depths = (depths .- dmin) ./ (dmax - dmin)
        scheme = reverse(ColorSchemes.Greens_4)

        sample_mask = Images.RGBA.(copy(img)).*0
        sample_depths = ones(size(img))
        for (i,col) in enumerate(eachcol(surf_px))
            # only override for closer depths
            if sample_depths[col[2], col[1]] > norm_depths[i]
                sample_mask[col[2], col[1]] = alphacolor(get(scheme, norm_depths[i]), 0.8)
                sample_depths[col[2], col[1]] = norm_depths[i]
            end
        end
        Plots.plot!(sample_mask)
    end

    # save each frame to a folder
    Plots.plot!(dpi=300)

    Plots.savefig(@sprintf "%s/%06d.png" parent frame)


    if frame % 10 == 0
        print("$frame ")
    end
end

# make video with:
# ffmpeg -framerate 10 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4