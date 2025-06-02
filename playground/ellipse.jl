## Get bounding ellipse via S-Lemma
# Lorenzo Shaikewitz 4/29/2025

using Serialization

# TODO: move to module
include("../src/uncertaintyset.jl")
include("../src/maxmargin.jl")
include("../src/slemma.jl")

# Parameters
object_id = 9

analytic = false
lowerb = -1
upperb = 10
silent = true

# Load data
cadpath = "./data/lmo/models_eval/"
datapath = "./data/lmo/l2_01.dat"
all_data = deserialize(datapath)
camK = all_data["camK"]
img_names = all_data["img"]
num_frames = length(img_names)

frame = 1

r = all_data["radii"][frame][object_id]
y = all_data["pixel_measurements"][frame][object_id]
b = all_data["canonical_kpts"][frame][object_id]

# build uncertainty set
q_front, q_backproj = uncertaintyset_l2(y, r, b, camK)
q_eqs = SO3_constraints()

# find center
println("----------MAX MARGIN----------")
sdp_pose, status, data, model = center_l2(q_front, q_backproj, q_eqs; analytic=analytic, lowerb=lowerb, upperb=upperb*r, silent=silent)

# local refinement + gap
tight = data[1]
vars_proj = data[3]
if tight
    gap = 0.
    est_pose = sdp_pose
else
    est_pose, loc_status, vars_proj, gap = local_refine(q_front, q_backproj, q_eqs, data; analytic=analytic, lowerb=lowerb, upperb=upperb*r, silent=silent)
end

# check feasibility
feas = check_feas(q_front, q_backproj, q_eqs, vars_proj; tol=1e-3, silent=silent)

println("Solved with status: $status")

if !feas
    println("Failed to find feasible solution!")
end

println("----------S LEMMA----------")
center = [vec(est_pose[1]); est_pose[2]]
H, status = bounding_ellipse(center, q_front, q_backproj, q_eqs; solver=Mosek.Optimizer, silent=true)

println("Solved with status: $status")


## Project ellipse to translations
P = [zeros(3,9) diagm(ones(3))]
H_t = inv(P*inv(H)*P')

# project to SO(3)?
P = [diagm(ones(9)) zeros(9,3)]
H_r = inv(P*inv(H)*P')


using Plots

function generate_ellipse(A, c, num_points=100)
    # Eigenvalue decomposition of A
    λ, V = eigen(Symmetric(A))
        
    # Generate points on unit sphere
    u = LinRange(0, 2π, num_points)
    v = LinRange(0, π, num_points)
    x = zeros(num_points, num_points)
    y = zeros(num_points, num_points)
    z = zeros(num_points, num_points)
    for (i,ui) in enumerate(u)
        # Parametric equations of the unit sphere
        x[:,i] = cos(ui)*sin.(v)
        y[:,i] = sin(ui)*sin.(v)
        z[:,i] = cos.(v)
    end
    sphere = [vec(x) vec(y) vec(z)]' # 3 x num_points^2

    # Transform sphere to ellipsoid
    axes_lengths = 1. ./ sqrt.(λ)
    surf = V * diagm(axes_lengths) * sphere
    surf .+= c
    return surf
end
# plot
Plots.plot([center[10]],[center[11]],[center[12]], seriestype=:scatter, label="center")
surf = generate_ellipse(H_t, center[10:12], 100)
Plots.scatter!([0],[0],[0],label="camera")
p2 = Plots.scatter3d!(surf[1,:], surf[2,:], surf[3,:], msw=0., alpha = 1)
Plots.plot!(label="Ellipse", xlabel="X", ylabel="Y", zlabel="Z", title="Projected")