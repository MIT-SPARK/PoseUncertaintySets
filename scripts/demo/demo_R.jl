## A simple demo of pose & uncertainty estimation
# using the S-Lemma method, rotation matrix form
# 
# Lorenzo Shaikewitz, 9/24/2025

using LinearAlgebra
using Printf
using TexTables
import Plots

using PoseUncertaintySets
using SimpleRotations

## Parameters
# dataset / object / image frame
dataset = "lmo" # ["lmo", "ycbv", "cast"]
object_id = 9
frame = 11
# relaxation order / confidence / keypoint uncertainty norm
estorder = 1
α = 0.1
p = 2 # options: [Inf, 2]

## load data
keypoint_data, gt = load_keypoint_data(calibrate_lp, "lmo"; p=p, α=α)
prob = get_problem(keypoint_data, object_id, frame)
# pose estimation
R, t, gap_pose, status_pose = gaussianpose(prob; silent=true, order=2)


## S-Lemma approach
println("Starting S-Lemma (order $estorder)...")
# joint uncertainty ellipse
out = @timed bounding_ellipse([vec(R); t], prob; order=estorder, silent=false)
H, gap, status = out.value
time_slem = out.time - out.compile_time
# marginalize
(Ht, Hθ), (boundst, boundsθ) = project_ellipse(H, R)

## RANSAG approach
println("Starting RANSAG (order $estorder)...")
out = @timed purse_bounds([vec(R); t], prob; order=estorder, silent=false)
boundt_ransag, gapt_ransag, boundθ_ransag, gapθ_ransag, status_ransag = out.value
time_ransag = out.time - out.compile_time


## Summarize / plot
# ground truth
gt = gt[frame][object_id]

# print stats
df_slem   = TableCol("S-Lemma", "time"=>time_slem, 
    "vol_t" => 4/3*π*prod(boundst),
    "vol_θ" => 4/3*π*prod(boundsθ),
    "gtcov" => ([vec(R); t]'*H*[vec(R); t] ≤ 1),
    "status"=>string(status), "gap"=>gap, 
    "status2"=>"-", "gap2"=>"-")

df_ransag = TableCol("RANSAG" , "time"=>time_ransag, 
    "vol_t" => 4/3*π*boundt_ransag^3,
    "vol_θ" => 4/3*π*boundθ_ransag^3,
    "gtcov" => (norm(t - gt[2]) ≤ boundt_ransag) && (roterror(R, project2SO3(gt[1])) ≤ boundθ_ransag),
    "status"=>string(status_ransag[1]), "gap"=>gapt_ransag, 
    "status2" => string(status_ransag[2]), "gap2"=>gapθ_ransag)

df = [df_slem df_ransag]
println("")
display(df)

# draw s-lemma ellipse results
## TODO: also include data--min working example!
## probably put data folder in repo.