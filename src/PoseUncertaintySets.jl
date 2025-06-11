module PoseUncertaintySets

using LinearAlgebra, Statistics
using StatsBase
using TSSOS, DynamicPolynomials
using Printf
using JuMP
using Clarabel, MosekTools
using GeometryBasics
using FileIO, MeshIO

# import Plots

using SimpleRotations
using P3P

# Files
include("pose_estimation.jl")
export gaussianpose, ransagpose

include("uncertainty_set.jl")
include("uncertainty_bounds.jl")
export bounding_ellipse, angular_bounds

include("datasets/poses.jl")
export dataset_pose_est, calc_pose_errors, calc_projection_errors

include("visualization/plot_ellipse.jl")
export ellipse_to_surf

end # module PoseUncertaintySets
