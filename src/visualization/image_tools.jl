## Plot keypoints, pose, masks on image
# Lorenzo Shaikewitz, 6/13/2025


function plot_image(parent, frame; type="png", data=nothing)
    if occursin("lmo", parent)
        parent = parent * "/000002/rgb"

        img_name = @sprintf "%06d.%s" frame type
    elseif occursin("ycbv", parent)
        folder = frame - (frame % 10000)
        parent = parent * "/" * (@sprintf "%06d" folder / 10000) * "/rgb"
        frame -= folder

        img_name = @sprintf "%06d.%s" frame type
    elseif occursin("cast", parent)
        !isnothing(data) || error("Data cannot be nothing for CAST.")
        img_name = split(data[frame]["rgb_image_filename"],"/")[end]
        # imgs = sort(filter(x -> startswith(x,"rgb_image"), readdir(parent)), by=x->parse(Int,split(split(x,"_")[end],".")[1]))
        # img_name = imgs[frame]
    else
        println("dataset not implemented")
    end
    img = Images.load(parent*"/"*img_name)

    plt = Plots.plot(img, axis=false, grid=false, title="Frame $frame")
    return plt, img
end

"""
Plot keypoints `y` and conformal radii `r`
"""
function plot_keypoints!(plt, y, r; set=true, p=2)
    for (idx, kpt) = enumerate(eachcol(y))
        u = Int(round(kpt[2]))
        v = Int(round(kpt[1]))
        if set
            if p != 2
                Plots.scatter!(plt,[v], [u], ms=r[idx], label=false, c=idx, ma=0.8, markershape=:rect)
            else
                Plots.scatter!(plt,[v], [u], ms=r[idx], label=false, c=idx, ma=0.8)
            end
        end
        Plots.scatter!(plt,[v], [u], ms=1, label=false, c=idx)
    end
    return plt
end

"""
Add projected 3D keypoints to keypoint plot `plt`
"""
function plot_3d_keypoints!(plt, pose, b, camK; msc=:white)
    R = project2SO3(pose[1])
    t = pose[2]
    for (idx, kpt) = enumerate(eachcol(b))
        # use gt transform
        kpt = camK*(R*b[:,idx] + t)
        kpt ./= kpt[3]
        u = Int(round(kpt[2]))
        v = Int(round(kpt[1]))
        Plots.scatter!(plt,[v], [u], ms=1, label=false, c=idx, msc=msc)
    end
    return plt
end

"""
Plot mask of object on image.
"""
function plot_mask!(plt, img, cadpath, object_id, pose, camK; lazy=true)
    # load CAD
    cad = FileIO.load(cadpath*(@sprintf "obj_%06d.ply" object_id))
    cad_m = GeometryBasics.Mesh(GeometryBasics.coordinates(cad)/1000, cad.faces)

    R = pose[1]
    t = pose[2]
    if lazy
        # just project vertices
        mask = get_lazy_mask(img, R, t, cad_m, camK)
    else
        # dense segmentation mask
        mask = get_mask(img, R, t, cad_m, camK)
    end
    img_mask = Images.RGBA.(copy(img)).*0
    img_mask[mask] .= Images.RGBA(0,1,1, 0.5)
    Plots.plot!(img_mask, grid=false, axis=false)
    return plt, mask
end

"""
Plot outline of object on image.
"""
function plot_outline!(plt, img, cadpath, object_id, pose, camK)
    # get segmentation mask
    cad = FileIO.load(cadpath*(@sprintf "obj_%06d.ply" object_id))
    cad_m = GeometryBasics.Mesh(GeometryBasics.coordinates(cad)/1000, cad.faces)

    seg = get_mask(img, pose[1], pose[2], cad_m, camK)
    plot_outline!(plt, img, seg)
    return plt, seg
end

function plot_outline!(plt, img, seg)
    bound = boundary_map(BitMatrix(seg))
    img_new = Images.RGBA.(copy(img)).*0
    img_new[bound] .= Images.RGBA(0,0,0, 1.)
    Plots.plot!(plt, img_new)
    return plt
end

"""
Get lazy mask by projecting vertices of object
"""
function get_lazy_mask(img, R, t, cad_m, camK)
    mask = zeros(Bool, size(img))
    for coord in GeometryBasics.coordinates(cad_m)
        pixel = camK*(R*coord + t)
        pixel ./= pixel[3]
        coords = [Int(round(pixel[1])), Int(round(pixel[2]))]
        coords[1] = clamp(coords[1], 1, size(img)[2])
        coords[2] = clamp(coords[2], 1, size(img)[1])
        mask[coords[2], coords[1]] = true
    end
    return mask
end


"""
Get segmentation mask by tracing object pose.

Rasterization version, relatively fast.
"""
function get_mask(img, R, t, cad_m, camK)
    H, W = size(img)         # image height, width
    mask = falses(H, W)      # Bool mask
    zbuf = fill(Inf, H, W)   # Z-buffer (depth)

    verts = [R * v + t for v in GeometryBasics.coordinates(cad_m)]
    faces = GeometryBasics.faces(cad_m)

    # Project vertices to image plane
    proj = Vector{Vector{Float64}}(undef, length(verts))
    for (i, v) in enumerate(verts)
        p = camK * v
        proj[i] = (p ./ p[3])[:]   # plain Vector{Float64} of length 3
    end

    # Rasterize each triangle
    for face in faces
        idxs = Vector(face)       # indices of triangle vertices
        tri = proj[idxs]         # projected 2D+homog vertices
        tri3d = verts[idxs]      # 3D vertices for depth

        # Extract x,y pixel coords and z depth
        xs = [tri[k][1] for k = 1:3]
        ys = [tri[k][2] for k = 1:3]
        zs = [tri3d[k][3] for k = 1:3]

        # Bounding box (clamped to image size)
        xmin = clamp(floor(Int, minimum(xs)), 1, W)
        xmax = clamp(ceil(Int,  maximum(xs)), 1, W)
        ymin = clamp(floor(Int, minimum(ys)), 1, H)
        ymax = clamp(ceil(Int,  maximum(ys)), 1, H)

        # Edge function
        edge(x0,y0, x1,y1, x,y) = (x - x0)*(y1 - y0) - (y - y0)*(x1 - x0)

        # Area of the triangle
        area = edge(xs[1], ys[1], xs[2], ys[2], xs[3], ys[3])

        if area == 0.0
            continue  # skip degenerate triangles
        end

        # Loop pixels in bounding box
        for j in ymin:ymax    # y = row index
            for i in xmin:xmax  # x = col index
                w0 = edge(xs[2], ys[2], xs[3], ys[3], i, j) / area
                w1 = edge(xs[3], ys[3], xs[1], ys[1], i, j) / area
                w2 = edge(xs[1], ys[1], xs[2], ys[2], i, j) / area

                if w0 >= 0 && w1 >= 0 && w2 >= 0
                    z = w0*zs[1] + w1*zs[2] + w2*zs[3]
                    if z < zbuf[j,i]
                        zbuf[j,i] = z
                        mask[j,i] = true
                    end
                end
            end
        end
    end

    return mask
end


# stolen from https://github.com/lucianolorenti/ImageSegmentationEvaluation.jl/blob/master/src/utils.jl#L61
function boundary_map(seg::BitMatrix)
    local ee = zeros(size(seg));
    local s = zeros(size(seg));
    local se = zeros(size(seg));

    ee[:,1:end-1] = seg[ :,2:end];
    s[1:end-1,:] = seg[2:end,:];
    se[1:end-1,1:end-1] = seg[2:end,2:end];

    local b = (seg.!=ee) .| (seg.!=s) .| (seg.!=se);
    b[end,:] = seg[end,:] .!= ee[end,:];
    b[:,end] = seg[:,end] .!= s[:,end];
    b[end,end] = 0;
    return b
end

"""
Plot a reference frame on an image
"""
function plot_frame!(plt, pose, K; gt=false)
    R = pose[1]
    t = pose[2]

    origin = t
    axes   = [t + 0.1*R[:,i] for i in 1:3]  # x, y, z axes in world

    # project to image
    project(p) = (K * (p))[1:2] ./ (K * (p))[3]
    o2d = project(origin)
    a2d = [project(p) for p in axes]

    colors = [:red, :green, :blue]
    for (a, c) in zip(a2d, colors)
        Plots.plot!(plt, [o2d[1], a[1]], [o2d[2], a[2]], color=c, lw=gt ? 2 : 1, legend=false)
    end
    return plt
end