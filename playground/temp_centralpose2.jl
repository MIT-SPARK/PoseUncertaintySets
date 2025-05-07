## Compute the central pose
# Lorenzo Shaikewitz, 5/7/2025

using Serialization

# TODO: move to module
include("../src/uncertaintyset.jl")
include("../src/centralpose.jl")

# Parameters
object_id = 9

analytic = false
lowerb = nothing #-10 # decrease to get more feasible frames at cost of runtime
upperb = 10
silent = true

# Load data
cadpath = "./data/lmo/models_eval/"
datapath = "./data/lmo/l2_04_real.dat"
all_data = deserialize(datapath)
camK = all_data["camK"]
img_names = all_data["img"]
num_frames = length(img_names)


frame = 1

r = all_data["radii"][frame][object_id] .+ 1e-3 # make sure 0 radii doesn't happen
y = all_data["pixel_measurements"][frame][object_id]
b = all_data["canonical_kpts"][frame][object_id]

p, d, v, f, g = centralpose_l2(y, r, b, camK; upperb=0.7*r, lowerb=-0.9*r)
# p, d, v, f, g = centralpose_percent_l2(y, r, b, camK; lowerb=0.3, upperb=2)
# these two give almost the same result;
# percent gives faster results but not tight, regular is tight but 2x slower
# put regular in paper, percent version in appendix

gt = all_data["gt_poses"][frame][object_id]
R_gt = project2SO3(gt[1])
t_gt = gt[2] / 1000. # [m]
R_est = p[1]
t_est = p[2]
err_R = roterror(R_gt, R_est)
err_t = norm(t_est - t_gt)*1000
@printf "R error: %.2f deg\n" err_R
@printf "t error: %.2f mm\n" err_t