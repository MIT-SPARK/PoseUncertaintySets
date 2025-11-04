## Test computing separate ellipses.
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
object_id = 9
frame = 11
# relaxation order / confidence / keypoint uncertainty norm
sdporder = 1
α = 0.1
p = 2 # options: [Inf, 2]

## load data
keypoint_data, gt = load_keypoint_data(calibrate_lp, "lmo"; p=p, α=α)
prob = get_problem(keypoint_data, object_id, frame)
# pose estimation
R, t, gap_pose, status_pose = gaussianpose(prob; silent=true, order=2)


## S-Lemma (together)
println("Starting S-Lemma (together)...")
# joint uncertainty ellipse
out = @timed bounding_ellipse([vec(R); t], prob; order=sdporder, silent=false)
H, gap, status = out.value
time_together = out.time - out.compile_time
# marginalize
(Ht, Hθ), (boundst, boundsθ) = project_ellipse(H, R)

## Rotation-Only
println("Starting S-Lemma (rotation only)...")
out = @timed bounding_ellipse_separated([vec(R); t], prob, [1,0]; order=sdporder, silent=false)
H_r, gap_r, status_r = out.value
time_r = out.time - out.compile_time
# marginalize
(_, Hθ_r), (_, boundsθ_r) = project_ellipse(H_r, R)

## Translation-Only
println("Starting S-Lemma (translation only)...")
out = @timed bounding_ellipse_separated([vec(R); t], prob, [0,1]; order=sdporder, silent=false)
H_t, gap_t, status_t = out.value
time_t = out.time - out.compile_time
# marginalize
(Ht_t, _), (boundst_t, _) = project_ellipse(H_t, R)


## Summarize / plot
# ground truth
gt = gt[frame][object_id]

# print stats
df_slem   = TableCol("Joint", "time"=>time_together, 
    "vol_t" => 4/3*π*prod(boundst),
    "vol_θ" => 4/3*π*prod(boundsθ),
    "gtcov" => ([vec(gt[1] - R); gt[2] - t]'*H*[vec(gt[1] - R); gt[2] - t] ≤ 1),
    "status"=>string(status), "gap"=>gap)

df_r   = TableCol("R only", "time"=>time_r, 
    "vol_t" => "-",
    "vol_θ" => 4/3*π*prod(boundsθ_r),
    "gtcov" => ([vec(gt[1] - R); gt[2] - t]'*H_r*[vec(gt[1] - R); gt[2] - t] ≤ 1),
    "status"=>string(status_r), "gap"=>gap_r)

df_t   = TableCol("t only", "time"=>time_t, 
    "vol_t" => 4/3*π*prod(boundst_t),
    "vol_θ" => "-",
    "gtcov" => ([vec(gt[1] - R); gt[2] - t]'*H_t*[vec(gt[1] - R); gt[2] - t] ≤ 1),
    "status"=>string(status_t), "gap"=>gap_t)


df = [df_slem df_r df_t]
println("")
display(df)

# draw ellipse results
if plot
    Plots.plotlyjs()
    # rotation ellipse
    plotR = Plots.scatter([0],[0],[0], label="center", title="Rotation", ratio=1, colorbar=false)
    Plots.surface!(ellipse_to_surf2(Hθ, zeros(3)), alpha=0.6, label="joint", c=2)
    Plots.surface!(ellipse_to_surf2(Hθ_r, zeros(3)), alpha=0.6, label="R only", c=3)
    # translation ellipse
    plott = Plots.scatter([t[1]],[t[2]],[t[3]], label="center", title="Translation", ratio=1, colorbar=false)
    Plots.surface!(ellipse_to_surf2(Ht, t, 100), alpha=0.6, label="joint", c=2)
    Plots.surface!(ellipse_to_surf2(Ht_t, t, 100), alpha=0.6, label="t only", c=3)

    # add samples
    S_R, S_t = sample_set(prob; method="ransag", T=1000)
    axangs = rotm2axang.([Rs*R' for Rs in S_R])
    ωsinθ = reduce(hcat,[ω*sin(θ) for (ω, θ) in axangs])
    Plots.scatter3d!(plotR, eachrow(ωsinθ)..., label="samples")
    S_t = reduce(hcat, S_t)
    Plots.scatter!(plott, eachrow(S_t)..., label="samples")
    Plots.scatter!(plott, [0],[0],[0],label="camera")

    Plots.plot(plotR, plott)
end