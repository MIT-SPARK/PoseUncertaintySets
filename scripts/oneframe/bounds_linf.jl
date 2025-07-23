## Uncertainty bounds on a single frame
# Lorenzo Shaikewitz, 6/11/2025

using LinearAlgebra
using Printf
using Clarabel, MosekTools
import Plots

using PoseUncertaintySets
using SimpleRotations

object_id = 9
frame = 11

# Load data
keypoint_data2, gt = load_keypoint_data(calibrate_lp, p=2  , α=0.1)
keypoint_datai, gt = load_keypoint_data(calibrate_lp, p=Inf, α=0.1)

prob2 = get_problem(keypoint_data2, object_id, frame)
probi = get_problem(keypoint_datai, object_id, frame)

# pose estimate
# R2, t2, gap2, status2 = gaussianpose(prob2; silent=true, order=2)
# Ri, ti, gapi, statusi = gaussianpose(probi; silent=true, order=2)
R2, t2, gap2, status2 = maxmarginpose(prob2; silent=true)
Ri, ti, gapi, statusi = maxmarginpose(probi; silent=true)

# bounding SPHERE
# center2 = [vec(R2); t2]
# rad2, statusb2 = bounding_sphere(center2, prob2; order=2, silent=false)
centeri = [rotm2quat(Ri); ti]
radi, statusbi = bounding_sphere(centeri, probi; order=2, silent=false) # uses quat form

# bounding ELLIPSE
center2 = [vec(R2); t2]
H2, statusb2 = bounding_ellipse(center2, prob2; silent=false)
centeri = [vec(Ri); ti]
Hi, statusbi = bounding_ellipse(centeri, probi; silent=false) # uses rot with redundant backproj constraints
centeri = [rotm2quat(Ri); ti]
Hqi, statusbqi = bounding_ellipse_quat(centeri, probi; order=2, silent=false)

# translation bounds
bounds2, gaps2, statusx2 = refine_bbox(center2, H2, prob2; mode=3, order=1, silent=true) # mode 3 only fastest for 1st order
# centeri = [rotm2quat(Ri); ti]
# Hi = diagm(ones(7))/radi^2
# boundsi3, gapsi, statusxi = refine_bbox(centeri, Hi, probi; mode=3, order=2, silent=true) # 4 s, worse than 1/2
# boundsi2, gapsi, statusxi = refine_bbox(centeri, Hi, probi; mode=2, order=2, silent=true) # 5.5 s, same as 1
boundsi1, gapsi, statusxi = refine_bbox(centeri, Hqi, probi; R=false, mode=1, order=2, silent=true) # 0.5 s, same as 2

# PLOT
function plot_ellipse(center, H_t; p=2)
    surf = ellipse_to_surf(H_t, center[end-2:end], 100)
    Plots.plot([center[end-2]],[center[end-1]],[center[end]], seriestype=:scatter, label="center")
    Plots.scatter!([0],[0],[0],label="camera")
    p2 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], label="s-lemma") #, msw=0.)
end

# P = [zeros(3,9) diagm(ones(3))]
P = [zeros(3,4) diagm(ones(3))]
H_t = inv(P*pinv(Hqi)*P')
p2 = plot_ellipse(centeri, H_t; p=Inf)

# plot_bbox!(p2, centeri, H_t, boundsi3; label="bbox3")
# plot_bbox!(p2, centeri, H_t, boundsi2; label="bbox2")
plot_bbox!(p2, centeri, H_t, boundsi1; label="bbox")


# linf vs. l2: Inf is generally tighter / faster.
# Should change objective for sphere in translation only for fair comparison
# Next: second order S-Lemma for Inf norm (same as R version of Inf norm)

# gaussian vs maxmargin: maxmargin is a better center (tighter uncertainty sets)
# Next: try defining ellipse by foci

# Conclusion: Inf norm works and is generally tighter and faster than 2 norm!
# Next: second order version of S-Lemma for Inf norm (experience shows it will have only small penalty)
# - this is the same as the R version of Inf norm (we also need some redundant constraints)