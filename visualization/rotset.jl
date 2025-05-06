## Visualize rotation uncertainty set
# Lorenzo Shaikewitz, 5/6/2025

import Images, Plots
import GeometryBasics
using FileIO, MeshIO

include("viz_utils.jl")

frame = 352

## LOAD DATA
img_name = @sprintf "%06d.png" parse(Int,img_names[frame])
img = Images.load("/home/lorenzo/Downloads/lmo_test_all/test/000002/rgb/$img_name")

r = all_data["radii"][frame][object_id]
y = all_data["pixel_measurements"][frame][object_id]
b = all_data["canonical_kpts"][frame][object_id]

# estimated central pose
Rc = R_ests[frame]
tc = t_ests[frame]

R_bound = all_R_bounds[:,frame]

# central pose
mask_center = get_mask(img, Rc, tc, cad_m, camK)
mask = copy(mask_center)
p1=Plots.plot(img, axis=false, grid=false, title="Frame $frame")
# Plots.plot!(mask)

for j = 1:3
    println(j)
    global mask
    ej = zeros(3)
    ej[j] = 1
    Rj = axang2rotm(ej, R_bound[j]*π/180)
    maskj = get_mask(img, Rj, tc, cad_m, camK)
    mask = mask .|| maskj

    Rj = axang2rotm(ej, R_bound[j]*π/180)
    maskj = get_mask(img, Rj, tc, cad_m, camK)
    mask = mask .|| maskj
end
img2 = Images.RGBA.(copy(img)).*0
img2[mask] .= Images.RGBA(0,1,1, 0.5)
Plots.plot!(img2)

bound = boundary_map(mask_center)
img3 = Images.RGBA.(copy(img)).*0
img3[bound] .= Images.RGBA(0,0,0, 1.)
Plots.plot!(img3)

p1