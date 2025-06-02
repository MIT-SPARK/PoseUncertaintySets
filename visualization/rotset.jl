## Visualize rotation uncertainty set
# Lorenzo Shaikewitz, 5/6/2025

import Images, Plots
import GeometryBasics
using FileIO, MeshIO
using Printf

include("viz_utils.jl")
include("../src/utils.jl")

using Serialization
using JuMP

# cadnames = Dict(1=>"ape", 5=>"can", 6=>"cat", 8=>"driller", 9=>"duck", 10=>"eggbox", 11=>"glue", 12=>"holepuncher")
object_id = 10
frame = 352 #k[408] # 352
# frame = 1099
# 843, 203, 408, 551, 583, 584, 585
# 972, 

# Load data
all_data = deserialize("data/lmo/linf_01_real.dat")
img_names = all_data["img"]
centers = deserialize("linf_01_percent01")
R_ests = centers["status"][object_id]["R_ests"]
t_ests = centers["status"][object_id]["t_ests"]
all_feas = centers["status"][object_id]["all_feas"]

marginal_dict = deserialize("angbounds40_fromlinf")
all_R_bounds = marginal_dict[object_id]["all_R_bounds"]

# println(argmin(all_R_bounds[1,all_feas.==1]))
display(all_R_bounds[:,frame]')

cadpath = "./data/lmo/models_eval/" # _eval
cad = FileIO.load(cadpath*(@sprintf "obj_%06d.ply" object_id))
cad_m = GeometryBasics.Mesh(GeometryBasics.coordinates(cad)/1000, cad.faces)


## LOAD DATA
img_name = @sprintf "%06d.png" parse(Int,img_names[frame])
img = Images.load("/home/lorenzo/Downloads/lmo_test_all/test/000002/rgb/$img_name")

r = all_data["radii"][frame][object_id]
y = all_data["pixel_measurements"][frame][object_id]
b = all_data["canonical_kpts"][frame][object_id]
camK = all_data["camK"]

# estimated central pose
Rc = R_ests[frame]
tc = t_ests[frame]

# ground truth (for printing)
tgt = all_data["gt_poses"][frame][object_id][2]/1000.
println("Error: $(norm(tgt - tc))")

# central pose
mask_center = get_mask(img, Rc, tc, cad_m, camK)
mask = copy(mask_center)
p1=Plots.plot(img, axis=false, grid=false, title="Frame $frame")
img_center = Images.RGBA.(copy(img)).*0
img_center[mask] .= Images.RGBA(0,1,1, 0.5)
# Plots.plot!(img_center)

# ellipse
R_bound = all_R_bounds[:,frame]
for j = 1:3
    println(j)
    global mask
    ej = zeros(3)
    ej[j] = 1
    Rj = axang2rotm(ej, R_bound[j]*π/180)
    maskj = get_mask(img, Rj*Rc, tc, cad_m, camK)
    mask = mask .|| maskj

    Rj = axang2rotm(ej, R_bound[j]*π/180)
    maskj = get_mask(img, Rj*Rc, tc, cad_m, camK)
    mask = mask .|| maskj
end
img2 = Images.RGBA.(copy(img)).*0
img2[mask] .= Images.RGBA(0,1,1, 0.5)
Plots.plot!(img2)

bound = boundary_map(BitMatrix(mask_center))
img3 = Images.RGBA.(copy(img)).*0
img3[bound] .= Images.RGBA(0,0,0, 1.)
Plots.plot!(img3)

p1