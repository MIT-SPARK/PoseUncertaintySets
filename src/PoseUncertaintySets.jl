module PoseUncertaintySets

using LinearAlgebra, Statistics
using StatsBase
using TSSOS, DynamicPolynomials
using Printf
using JuMP
using Clarabel, MosekTools
using SDPLR
using GeometryBasics
using FileIO, MeshIO

import JSON
import Images, Plots

# import Plots

using SimpleRotations
using P3P

# Files
include("load_data.jl")
export calibrate_l2, load_keypoint_data

include("pose_estimation.jl")
export gaussianpose, ransagpose, gaussianpose_sdplr

include("uncertainty_set.jl")
include("uncertainty_bounds.jl")
export bounding_ellipse, angular_bounds, purse_bounds
export bounding_sphere, check_feasibility, refine_bbox

include("datasets/poses.jl")
export dataset_pose_est, calc_pose_errors, calc_projection_errors
include("datasets/bounds.jl")
export dataset_bounds

include("visualization/plot_3d.jl")
export ellipse_to_surf, plot_bbox!
include("visualization/image_tools.jl")
export plot_image, plot_keypoints!, plot_3d_keypoints!, plot_mask!, plot_outline!

end # module PoseUncertaintySets
