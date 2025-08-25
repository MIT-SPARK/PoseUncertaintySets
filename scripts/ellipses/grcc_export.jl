## Export center and constraints for GRCC
# I couldn't figure out how to implement their method
# so we'll run it in MATLAB on a separate machine...

using Serialization
using MAT
using DataFrames
using PoseUncertaintySets

# parameters
# dataset = "lmo"
# object_ids = [1,5,6,9,8,11,12]
# dataset = "ycbv"
# object_ids = 1:21
dataset = "cast"
object_ids = [1]
α = 0.1
p = 2 # cannot do Inf
pose = "pnp2" # original paper uses ransag

posepath = "../data/$dataset/pose_$(pose)_$(round(Int,α*100))_$(string(p)).dat"
keypoint_data, gt = load_keypoint_data(calibrate_lp, dataset; p=p, α=α)
pose_data = deserialize(posepath)["solns"]
savepath = "../data/$dataset/grcc/"


println("Starting GRCC Export...")
for object_id in object_ids
    println("\n------------$object_id------------")
    num_frames = length(keys(pose_data[object_id][1]))

    println("Starting $num_frames frames...")
    for frame in sort(collect(keys(pose_data[object_id][1])))

        # frame-specific data
        r = keypoint_data["r"][frame][object_id]
        y = keypoint_data["y"][frame][object_id]
        b = keypoint_data["b"][object_id]

        # eliminate missing measurements
        y = y[1:2,r .>= 0]
        y = [y[1:2,:]; ones(size(y,2))']
        b = b[:, r .>= 0]
        r = r[r .>= 0]

        # load pose data
        R_est = pose_data[object_id][1][frame]
        t_est = pose_data[object_id][2][frame]
        center = [vec(R_est); t_est]

        # constraints
        q_front, q_backproj = PoseUncertaintySets.uncertaintyset_l2(y, r, b, keypoint_data["K"])
        # linCon
        linCon = reduce(hcat,[qf.c for qf in q_front])
        # quadCon
        quadCon = reduce((x, y) -> cat(x, y; dims = 3),[qb.H for qb in q_backproj])

        # save
        file=matopen(savepath*"obj$(object_id)_frame$(frame).mat","w")
        write(file, "center", center)
        write(file, "linCon", linCon)
        write(file, "quadCon", quadCon)
        close(file)
    end

end