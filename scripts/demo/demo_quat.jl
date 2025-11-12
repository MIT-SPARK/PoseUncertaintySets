## A simple demo of pose & uncertainty estimation
# using the S-Lemma method, quaternion matrix form
# this version works best for SECOND ORDER
# 
# Lorenzo Shaikewitz, 9/24/2025

using LinearAlgebra
using Printf
using TexTables
import Plots

using PoseUncertaintySets
using SimpleRotations

## Parameters
plot = true
# dataset / object / image frame
dataset = "lmo" # ["lmo", "ycbv", "cast"]
object_id = 1
frame = 111
# relaxation order / confidence / keypoint uncertainty norm
sdporder = 2 # cannot do order 1
α = 0.1

## load data
keypoint_data, gt = load_keypoint_data(calibrate_lp, "lmo"; p=Inf, α=α)
prob = get_problem(keypoint_data, object_id, frame)
# pose estimation
R, t, gap_pose, status_pose = gaussianpose(prob; silent=true, order=2)


## S-Lemma approach
println("Starting S-Lemma (order $sdporder)...")
# joint uncertainty ellipse
out = @timed bounding_ellipse_quat([rotm2quat(R); t], prob; order=sdporder, silent=false)
H, g, status = out.value
time_slem = out.time - out.compile_time
# marginalize
(Ht, Hθ), (boundst, boundsθ) = project_ellipse(H, R)

## RANSAG approach
println("Starting RANSAG (order $sdporder)...")
out = @timed purse_bounds([vec(R); t], prob; p=2, order=sdporder, silent=false)
boundt_ransag, gapt_ransag, boundθ_ransag, gapθ_ransag, status_ransag = out.value
time_ransag = out.time - out.compile_time


## Summarize / plot
# ground truth
gt = gt[frame][object_id]

# print stats
df_slem   = TableCol("S-Lemma", "time"=>time_slem, 
    "vol_t" => 4/3*π*prod(boundst),
    "vol_θ" => 4/3*π*prod(boundsθ),
    "gtcov" => ([rotm2quat(gt[1]) - rotm2quat(R); gt[2] - t]'*H*[rotm2quat(gt[1]) - rotm2quat(R); gt[2] - t] ≤ 1),
    "status_r"=>string(g), "status_t"=>string(status))

df_ransag = TableCol("RANSAG" , "time"=>time_ransag, 
    "vol_t" => 4/3*π*boundt_ransag^3,
    "vol_θ" => 4/3*π*boundθ_ransag^3,
    "gtcov" => (norm(t - gt[2]) ≤ boundt_ransag) && (roterror(R, project2SO3(gt[1])) ≤ boundθ_ransag),
    "status_r"=>string(status_ransag[2]), "status_t" => string(status_ransag[1]))

df = [df_slem df_ransag]
println("")
display(df)

# draw ellipse results
if plot
    Plots.plotlyjs()
    # rotation ellipse
    plotR = Plots.scatter([0],[0],[0], label="center", title="Rotation", ratio=1, colorbar=false)
    Plots.surface!(ellipse_to_surf2(Hθ, zeros(3)), alpha=0.6, label="order $sdporder", c=2)
    Plots.surface!(ellipse_to_surf2(1/(sin(boundθ_ransag*π/180 / 2)^2)*diagm(ones(3)), zeros(3)), alpha=0.6, label="RANSAG", c=3)

    # translation ellipse
    plott = Plots.scatter([t[1]],[t[2]],[t[3]], label="center", title="Translation", ratio=1, colorbar=false)
    Plots.surface!(ellipse_to_surf2(Ht, t, 100), alpha=0.6, label="order $sdporder", c=2)
    Plots.surface!(ellipse_to_surf2(1/(boundt_ransag^2)*diagm(ones(3)), t), alpha=0.6, label="RANSAG", c=3)

    # add samples
    S_R, S_t = sample_set(prob; method="ransag", T=1000)
    axangs = rotm2axang.([Rs*R' for Rs in S_R])
    ωsinθ2 = reduce(hcat,[ω*sin(θ/2) for (ω, θ) in axangs])
    Plots.scatter3d!(plotR, eachrow(ωsinθ2)..., label="samples")
    S_t = reduce(hcat, S_t)
    Plots.scatter!(plott, eachrow(S_t)..., label="samples")
    Plots.scatter!(plott, [0],[0],[0],label="camera")

    # draw on image (TODO)

    Plots.plot(plotR, plott)
end