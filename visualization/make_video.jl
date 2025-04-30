# Run centralpose first
# uses all_data, R_ests, t_ests

import Images, Plots
import GeometryBasics
using FileIO, MeshIO

plot_gt = false
name = "est.mp4"
fps = 15

anim = @Plots.animate for frame = 1:num_frames
    img_name = @sprintf "%06d.png" parse(Int,img_names[frame])
    img = Images.load("/home/lorenzo/Downloads/lmo_test_all/test/000002/rgb/$img_name")
    Plots.plot(img, axis=false, title="Frame $frame")
    if !(object_id in keys(all_data["radii"][frame]))
        continue
    end

    r = all_data["radii"][frame][object_id]
    y = all_data["pixel_measurements"][frame][object_id]
    b = all_data["canonical_kpts"][frame][object_id]

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
    for coord in GeometryBasics.coordinates(duck_m)
        pixel = camK*(R*coord + t)
        pixel ./= pixel[3]
        coords = [Int(round(pixel[1])), Int(round(pixel[2]))]
        coords[1] = clamp(coords[1], 1, size(img)[2])
        coords[2] = clamp(coords[2], 1, size(img)[1])
        img2[coords[2], coords[1]] = Images.RGBA(0,1,1, 0.5)
    end
    Plots.plot!(img2, grid=false, axis=false)
    print("$frame ")
end
Plots.gif(anim, name, fps = fps)