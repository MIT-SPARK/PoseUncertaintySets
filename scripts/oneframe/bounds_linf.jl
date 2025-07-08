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
R2, t2, gap2, status2 = gaussianpose(prob2; silent=true, order=2)
Ri, ti, gapi, statusi = gaussianpose(probi; silent=true, order=2)
# R2, t2, gap2, status2 = maxmarginpose(prob2; silent=true)
# Ri, ti, gapi, statusi = maxmarginpose(probi; silent=true)

# bounds estimate
center2 = [vec(R2); t2]
rad2, statusb2 = bounding_sphere(center2, prob2; order=2, silent=false)
centeri = [rotm2quat(Ri); ti]
radi, statusbi = bounding_sphere(centeri, probi; order=2, silent=false)


center2 = [vec(R2); t2]
H2, statusb2 = bounding_ellipse(center2, prob2; silent=false)

centeri = [vec(Ri); ti]
Hi, statusbi = bounding_ellipse(centeri, probi; silent=false)



# linf vs. l2: Inf is generally tighter / faster.
# Should change objective for sphere in translation only for fair comparison
# Next: second order S-Lemma for Inf norm (same as R version of Inf norm)

# gaussian vs maxmargin: maxmargin is a better center (tighter uncertainty sets)
# Next: try defining ellipse by foci

# Conclusion: Inf norm works and is generally tighter and faster than 2 norm!
# Next: second order version of S-Lemma for Inf norm (experience shows it will have only small penalty)
# - this is the same as the R version of Inf norm (we also need some redundant constraints)