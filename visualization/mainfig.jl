# makes the base images for the main figure
# run at least central pose first

import Images, Plots
import GeometryBasics
using FileIO, MeshIO

include("viz_utils.jl")


frame = 352
plot_mask = false

begin
    object_id = 11

    all_status = status_dict[object_id]["all_status"]
    all_feas = status_dict[object_id]["all_feas"]
    all_gaps = status_dict[object_id]["all_gaps"]
    all_times = status_dict[object_id]["all_times"]

    R_ests = status_dict[object_id]["R_ests"]
    t_ests = status_dict[object_id]["t_ests"]
end


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
for (idx, kpt) = enumerate(eachcol(y))
    u = Int(round(kpt[2]))
    v = Int(round(kpt[1]))
    Plots.scatter!([v], [u], ms=Int(round(r[idx])), label=false, c=idx, ma=0.8)
    Plots.scatter!([v], [u], ms=1, label=false, c=idx)
end
plot_kpts_l2 = Plots.plot!(grid=false, axis=false)

begin
    # project gt keypoints
    R_est = R_ests[frame]
    t_est = t_ests[frame]
    img_kpts = Images.RGBA.(copy(img)).*0
    for (idx, kpt) = enumerate(eachcol(b))
        # use gt transform
        kpt = camK*(R_gt*b[:,idx] + t_gt)
        kpt ./= kpt[3]
        u = Int(round(kpt[2]))
        v = Int(round(kpt[1]))
        Plots.scatter!([v], [u], ms=1, label=false, c=idx, msc=:white)

        # use est transform
        kpt = camK*(R_est*b[:,idx] + t_est)
        kpt ./= kpt[3]
        u = Int(round(kpt[2]))
        v = Int(round(kpt[1]))
        Plots.scatter!([v], [u], ms=1, label=false, c=idx, msc=:blue)
    end
end


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
plot_gt = false
if plot_mask
    if plot_gt
        gt = all_data["gt_poses"][frame][object_id]
        R = gt[1]
        t = gt[2] / 1000
    else
        R = R_ests[frame]
        t = t_ests[frame]
    end

    p1=Plots.plot(img, axis=false, grid=false, title="Frame $frame")
    mask = get_mask(img, R, t, cad_m, camK)
    img2 = Images.RGBA.(copy(img)).*0
    img2[mask] .= Images.RGBA(0,1,1, 0.5)
    Plots.plot!(img2)
end

plot_kpts_l2