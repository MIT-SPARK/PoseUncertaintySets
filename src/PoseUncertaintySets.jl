module PoseUncertaintySets

using LinearAlgebra, Statistics
using StatsBase
using TSSOS, DynamicPolynomials
using Printf
using JuMP
using Clarabel, MosekTools
using SDPLR, Ipopt
using GeometryBasics
using FileIO, MeshIO

import JSON
import Images, Plots

# import Plots

using SimpleRotations
using P3P

# Files
include("load_data.jl")
export get_problem, calibrate_lp, load_keypoint_data

include("pose_estimation.jl")
export gaussianpose, ransagpose, gaussianpose_sdplr, maxmarginpose, conformalpose, conformalpose_local

include("uncertainty_set.jl")
include("uncertainty_bounds.jl")
export bounding_ellipse, bounding_sphere, bounding_ellipse_quat
include("explicit_bounds.jl")
export purse_bounds, angular_bounds_rpy, angular_bounds_quat, refine_bbox

include("datasets/poses.jl")
export dataset_pose_est, calc_pose_errors, calc_projection_errors
include("datasets/bounds.jl")
export dataset_slem_bounds, dataset_slem_bounds_quat, dataset_ransag_bounds
include("datasets/workflows.jl")
export l2_workflow, linfR_workflow, linfq_workflow

include("visualization/plot_3d.jl")
export ellipse_to_surf, plot_bbox!, plot_cdf!
include("visualization/image_tools.jl")
export plot_image, plot_keypoints!, plot_3d_keypoints!, plot_mask!, plot_outline!

end # module PoseUncertaintySets
