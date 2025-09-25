module PoseUncertaintySets

using LinearAlgebra, Statistics
using StatsBase
using TSSOS, DynamicPolynomials
using Printf
using JuMP
using Clarabel, MosekTools
using SDPLR, Ipopt
using Dualization
using GeometryBasics
using FileIO, MeshIO

import JSON
import Images, Plots

# import Plots

using SimpleRotations
using P3P

# make dictionary callable
(d::Dict)(k) = d[k] 

# Files
include("load_data.jl")
export get_problem, calibrate_lp, load_keypoint_data

include("pose_estimation.jl")
export gaussianpose, ransagpose, maxmarginpose

include("uncertainty_set.jl")
export sample_set
include("uncertainty_bounds.jl")
export bounding_ellipse, bounding_sphere, bounding_ellipse_quat, bounding_ellipse_separated
include("explicit_bounds.jl")
export purse_bounds, refine_bbox, project_ellipse
export grcc_bounds

include("datasets/poses.jl")
export dataset_pose_est, calc_pose_errors, calc_projection_errors
include("datasets/bounds.jl")
export dataset_ransag_bounds, dataset_slem

include("visualization/plot_3d.jl")
export ellipse_to_surf, plot_bbox!, plot_cdf!, ellipse_to_surf_nd, ellipse_to_surf2
include("visualization/image_tools.jl")
export plot_image, plot_keypoints!, plot_3d_keypoints!, plot_mask!, plot_outline!
export get_lazy_mask, get_mask, plot_frame!

end # module PoseUncertaintySets
