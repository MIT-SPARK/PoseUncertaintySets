## show CAD models annotated with keypoints
#
# Lorenzo Shaikewitz, 10/9/2025
# TODO: install GLMakie

using GLMakie
import FileIO, GeometryBasics, Images
using Printf
using LinearAlgebra
using SimpleRotations
using PoseUncertaintySets

dataset = "lmo"
object_id = 9
# dataset = "ycbv"
# object_id = 14
# dataset = "cast"
# object_id = 1
α = 0.1
p = Inf

image_parent = "../data/$dataset/test"
cadpath = "../data/$dataset/models_eval/"

# load CAD
cad = FileIO.load(cadpath*(@sprintf "obj_%06d.ply" object_id))
cad_m = GeometryBasics.Mesh(GeometryBasics.coordinates(cad)/1000, cad.faces)

# load data and calibrate
keypoint_data, gt = load_keypoint_data(calibrate_lp, dataset, p=p, α=α)

f, ax, pl = mesh(cad_m, color=:yellow, axis=(; show_axis=false))
scatter!(eachrow(keypoint_data["b"][object_id])...,markersize=40)

