## Functions to compute the central pose solution
# Lorenzo Shaikewitz, 5/7/2025

using Printf
using LinearAlgebra
using TSSOS, DynamicPolynomials

# TODO: move to submodule
include("utils.jl")
include("refine.jl")

"""

"""
function central_pose_l2()