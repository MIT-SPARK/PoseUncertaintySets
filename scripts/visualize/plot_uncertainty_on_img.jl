## Plot pose uncertainty set on the image
# used in main keypoints/uncertainty figure
# 
# Lorenzo Shaikewitz, 7/31/2025

import Plots
import FileIO, GeometryBasics, Images
using Printf
using LinearAlgebra
using SimpleRotations
using PoseUncertaintySets

image_parent = "../data/lmo/test"
cadpath = "../data/lmo/models_eval/"
# load CAD
cad = FileIO.load(cadpath*(@sprintf "obj_%06d.ply" object_id))
cad_m = GeometryBasics.Mesh(GeometryBasics.coordinates(cad)/1000, cad.faces)

# settings used for figure
object_id = 9
frame = 352
α = 0.1
p = Inf

# load data and calibrate
keypoint_data, gt = load_keypoint_data(calibrate_lp, p=p, α=α)
prob = get_problem(keypoint_data, object_id, frame); camK = prob.camK

# get pose estimate and ellipse
R_est, t_est, gap_pose, status_pose = gaussianpose(prob; silent=true, order=2)
center = [rotm2quat(R_est); t_est]
H, gap_slem, status_slem = bounding_ellipse_quat(center, prob; silent=true, order=2)

# plot pose estimate
p_pose, img = plot_image(image_parent, frame)
(p_pose, seg) = plot_mask!(p_pose, img, cadpath, object_id, (R_est, t_est), camK; lazy=false)
plot_outline!(p_pose, img, seg)

# plot uncertainty (sampled from pose uncertainty set)
p = Plots.plot()
S_R, S_t = sample_set(prob; method="ransag", T=1000)
sample_mask = zeros(Bool, size(img))
for (R, t) in zip(S_R, S_t)
    global sample_mask
    mask = get_lazy_mask(img, R, t, cad_m, camK)
    sample_mask .|= mask
end
p_sampled, img = plot_image(image_parent, frame)
img_mask = Images.RGBA.(copy(img)).*0
img_mask[sample_mask] .= Images.RGBA(0,1,1, 0.5)
p_sampled = Plots.plot!(img_mask)
plot_outline!(p_sampled, img, seg)
Plots.title!("RANSAG Sampled (frame $frame)")

# plot uncertainty (sampled from marginalized ellipses)
# project to axang space
P = [zeros(6)  diagm(ones(6))]
Ph = [Ω2(rotm2quat(R_est))  zeros(4,3);  zeros(3,4)   diagm(ones(3))]
H_adj = inv(P*pinv(Ph'*H*Ph)*P')
# grid on boundary of joint ellipse
surf = ellipse_to_surf_nd(H_adj, [zeros(3); t_est], 1e4)
surf_t = eachcol(surf[end-2:end,:])
# assume θ within 90° of center ⟹ cosθ > 0
surf_R = [quat2rotm([sqrt(1 - ωsinθ2'*ωsinθ2); ωsinθ2])*R_est for ωsinθ2 in eachcol(surf[1:3,:])]

for (R, t) in zip(surf_R, surf_t)
    global sample_mask
    mask = get_lazy_mask(img, R, t, cad_m, camK)
    sample_mask .|= mask
end
p_surf, img = plot_image(image_parent, frame)
img_mask = Images.RGBA.(copy(img)).*0
img_mask[sample_mask] .= Images.RGBA(0,1,1, 0.5)
p_surf = Plots.plot!(img_mask)
plot_outline!(p_surf, img, seg)
Plots.title!("Ellipse Surface (frame $frame)")