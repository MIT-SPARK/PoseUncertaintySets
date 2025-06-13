module PoseUncertaintySets

using LinearAlgebra, Statistics
using StatsBase
using TSSOS, DynamicPolynomials
using Printf
using JuMP
using Clarabel, MosekTools
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
export gaussianpose, ransagpose

include("uncertainty_set.jl")
include("uncertainty_bounds.jl")
export bounding_ellipse, angular_bounds

include("datasets/poses.jl")
export dataset_pose_est, calc_pose_errors, calc_projection_errors

include("visualization/plot_ellipse.jl")
export ellipse_to_surf
include("visualization/image_tools.jl")
export plot_image, plot_keypoints!, plot_3d_keypoints!, plot_mask!, plot_outline!

end # module PoseUncertaintySets
