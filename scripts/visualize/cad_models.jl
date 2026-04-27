## show CAD models annotated with keypoints
#
# Lorenzo Shaikewitz, 10/9/2025
# TODO: install GLMakie

using GLMakie
import FileIO, GeometryBasics, Images
using Printf
using LinearAlgebra
using SimpleRotations
using PoseUncertaintySets

dataset = "lmo"
object_id = 9
frame = 352
# dataset = "ycbv"
# object_id = 14
# dataset = "cast"
# object_id = 1
α = 0.1
p = Inf

image_parent = "../data/$dataset/test"
cadpath = "../data/$dataset/models_eval/"

# load CAD
cad = FileIO.load(cadpath*(@sprintf "obj_%06d.ply" object_id))
cad_m = GeometryBasics.Mesh(GeometryBasics.coordinates(cad)/1000, cad.faces)

# load data and calibrate
keypoint_data, gt = load_keypoint_data(calibrate_lp, dataset, p=p, α=α)
prob = get_problem(keypoint_data, object_id, frame); camK = prob.camK

f, ax, pl = mesh(cad_m, color=:white, axis=(; show_axis=false))
scatter!(eachrow(keypoint_data["b"][object_id])...,markersize=40)

# get pose estimate and ellipse
R_est, t_est, gap_pose, status_pose = gaussianpose(prob; silent=true, order=2)
center = [rotm2quat(R_est); t_est]
H, gap_slem, status_slem = bounding_ellipse_quat(center, prob; silent=true, order=2)

# plot pose estimate
coords_pose = [Float32.(R_est)] .* GeometryBasics.coordinates(cad_m) .+ [Float32.(t_est)]
coords_pose = convert(Vector{Point{3, Float32}}, coords_pose)
cad_pose = GeometryBasics.Mesh(coords_pose, cad.faces)

f2, ax, pl = mesh(cad_pose, color=:white, axis=(; show_axis=false))

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
for i = 1:20:length(surf_t)
    coordsi = [Float32.(surf_R[i])] .* GeometryBasics.coordinates(cad_m) .+ [Float32.(surf_t[i])]
    coordsi = convert(Vector{Point{3, Float32}}, coordsi)
    cadi = GeometryBasics.Mesh(coordsi, cad.faces)
    mesh!(cadi, color=color = RGBAf(0.8, 0.6, 0.1, 0.1), transparency = true)
end
scatter!([0],[0],[0],markersize=20)
f2
