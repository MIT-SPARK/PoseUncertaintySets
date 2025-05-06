# makes the base images for the main figure
# run at least central pose first

import Images, Plots
import GeometryBasics
using FileIO, MeshIO

frame = 352
plot_gt = false

## LOAD DATA
img_name = @sprintf "%06d.png" parse(Int,img_names[frame])
img = Images.load("/home/lorenzo/Downloads/lmo_test_all/test/000002/rgb/$img_name")

r = all_data["radii"][frame][object_id]
y = all_data["pixel_measurements"][frame][object_id]
b = all_data["canonical_kpts"][frame][object_id]
gt = all_data["gt_poses"][frame][object_id]
R_gt = gt[1]
t_gt = gt[2] / 1000

## PLOT KEYPOINTS
# l2
Plots.plot(img, axis=false, title="Frame $frame")
img_kpts = Images.RGBA.(copy(img)).*0
pixel_radius = 2
for (idx, kpt) = enumerate(eachcol(y))
    u = Int(round(kpt[2]))
    v = Int(round(kpt[1]))
    Plots.scatter!([v], [u], ms=Int(round(r[idx])), label=false, c=idx)
    Plots.scatter!([v], [u], ms=2, label=false, c=idx)
end
plot_kpts_l2 = Plots.plot!(img_kpts, grid=false, axis=false)

# linf
Plots.plot(img, axis=false, title="Frame $frame")
img_kpts = Images.RGBA.(copy(img)).*0
pixel_radius = 2
for (idx, kpt) = enumerate(eachcol(y))
    u = Int(round(kpt[2]))
    v = Int(round(kpt[1]))
    Plots.scatter!([v], [u], ms=Int(round(r[idx])), label=false, c=idx, markershape=:rect)
    # Plots.scatter!([v], [u], ms=Int(round(r[idx])), label=false, c=idx)
    Plots.scatter!([v], [u], ms=2, label=false, c=idx)
end
plot_kpts_linf = Plots.plot!(img_kpts, grid=false, axis=false)

# project each gt keypoint and plot
Plots.plot(img, axis=false, title="Frame $frame")
img_kpts = Images.RGBA.(copy(img)).*0
pixel_radius = 2
for (idx, kpt) = enumerate(eachcol(y))
    u = Int(round(kpt[2]))
    v = Int(round(kpt[1]))
    # Plots.scatter!([v], [u], ms=Int(round(r[idx])), label=false, c=idx)
    # Plots.scatter!([v], [u], ms=2, label=false, c=idx)
end
b_proj = camK*(R_gt*b .+ t_gt)
b_proj = reduce(hcat, eachcol(b_proj) ./ b_proj[3,:])
Plots.plot!(b_proj[1,:], b_proj[2,:])

plot_kpts = Plots.plot!(img_kpts, grid=false, axis=false)

## PLOT POSE ESTIMATE
if plot_gt
    gt = all_data["gt_poses"][frame][object_id]
    R = gt[1]
    t = gt[2] / 1000
else
    R = R_ests[frame]
    t = t_ests[frame]
end
# Option 1: project each coordinate of duck
img2 = Images.RGBA.(copy(img)).*0
bbox = zeros(Int,2,2) # [min u max u; min v max v]
bbox[:,1] = [1000;1000]
for coord in GeometryBasics.coordinates(cad_m)
    pixel = camK*(R*coord + t)
    pixel ./= pixel[3]
    coords = [Int(round(pixel[1])), Int(round(pixel[2]))]
    coords[1] = clamp(coords[1], 1, size(img)[2])
    coords[2] = clamp(coords[2], 1, size(img)[1])
    img2[coords[2], coords[1]] = Images.RGBA(0,1,1, 0.5)

    bbox[1,1] = min(bbox[1,1], coords[1])
    bbox[1,2] = max(bbox[1,2], coords[1])
    bbox[2,1] = min(bbox[2,1], coords[2])
    bbox[2,2] = max(bbox[2,2], coords[2])
end
Plots.plot(img, axis=false, title="Frame $frame")
plot_pose = Plots.plot!(img2, grid=false, axis=false)

## ray casting ish
function check_triangle(triangle, ray_origin, ray_direction)
    # face normal
    e1 = triangle[2] - triangle[1]
    e2 = triangle[3] - triangle[1]

    # weights in Barycentric coordinates
    q = ray_direction × e2
    a = e1'*q
    
    s = ray_origin - triangle[1]
    r = s × e1
    weights = [0; s'*q/a; ray_direction'*r/a]
    weights[1] = 1. - (weights[2]+weights[3])

    dist = e2'*r/a

    ϵ = 1e-7
    if (a <= ϵ) || sum(weights .< -ϵ) > 0 || (dist <= 0)
        return false
    else
        return true
    end
end

# this may take a while
coords = copy(GeometryBasics.coordinates(cad_m))
for (i,c) in enumerate(coords)
    coords[i] = R*c + t
end
faces = GeometryBasics.faces(cad_m)
img3 = Images.RGBA.(copy(img)).*0
for i in bbox[1,1]:bbox[1,2]
    for j in bbox[2,1]:bbox[2,2]
        # compute ray direction
        ray_direction = normalize(inv(camK)*[i;j;1])

        # check intersection with face
        for face in faces
            triangle = coords[face]
            ray_origin = zeros(3)
            if check_triangle(triangle, ray_origin, ray_direction)
                img3[j, i] = Images.RGBA(0,1,1, 0.5)
                break
            end
        end
    end
end
Plots.plot(img, axis=false, title="Frame $frame")
plot_pose_mask = Plots.plot!(img3, grid=false, axis=false)

## PLOT ALL POSE ESTIMATES
