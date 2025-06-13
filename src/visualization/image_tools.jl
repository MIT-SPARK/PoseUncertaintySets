## Plot keypoints, pose, masks on image
# Lorenzo Shaikewitz, 6/13/2025


function plot_image(parent, frame)
    img_name = @sprintf "%06d.png" frame
    img = Images.load(parent*"/"*img_name)

    plt = Plots.plot(img, axis=false, grid=false, title="Frame $frame")
    return plt, img
end

"""
Plot keypoints `y` and conformal radii `r`
"""
function plot_keypoints!(plt, y, r)
    for (idx, kpt) = enumerate(eachcol(y))
        u = Int(round(kpt[2]))
        v = Int(round(kpt[1]))
        Plots.scatter!(plt,[v], [u], ms=r[idx], label=false, c=idx, ma=0.8)
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
        img2 = Images.RGBA.(copy(img)).*0
        for coord in GeometryBasics.coordinates(cad_m)
            pixel = camK*(R*coord + t)
            pixel ./= pixel[3]
            coords = [Int(round(pixel[1])), Int(round(pixel[2]))]
            coords[1] = clamp(coords[1], 1, size(img)[2])
            coords[2] = clamp(coords[2], 1, size(img)[1])
            img2[coords[2], coords[1]] = Images.RGBA(0,1,1, 0.5)
        end
        Plots.plot!(img2, grid=false, axis=false)
        return plt
    else
        # dense segmentation mask
        mask = get_mask(img, R, t, cad_m, camK)
        img_mask = Images.RGBA.(copy(img)).*0
        img_mask[mask] .= Images.RGBA(0,1,1, 0.5)
        Plots.plot!(plt, img_mask)
        return plt, mask
    end
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
Get segmentation mask by tracing object pose
"""
function get_mask(img, R, t, cad_m, camK)
    # get bbox using vertices
    bbox = zeros(Int,2,2) # [min u max u; min v max v]
    bbox[:,1] = [1000;1000]
    mask = zeros(Bool, size(img))
    for coord in GeometryBasics.coordinates(cad_m)
        pixel = camK*(R*coord + t)
        pixel ./= pixel[3]
        coords = [Int(round(pixel[1])), Int(round(pixel[2]))]
        coords[1] = clamp(coords[1], 1, size(img)[2])
        coords[2] = clamp(coords[2], 1, size(img)[1])

        bbox[1,1] = min(bbox[1,1], coords[1])
        bbox[1,2] = max(bbox[1,2], coords[1])
        bbox[2,1] = min(bbox[2,1], coords[2])
        bbox[2,2] = max(bbox[2,2], coords[2])

        mask[coords[2], coords[1]] = true
    end

    # ray casting helper function: check intersection with triangle
    function check_triangle(face, coords, ray_origin, ray_direction)
        triangle = coords[face]

        # face normal
        e1 = triangle[2] - triangle[1]
        e2 = triangle[3] - triangle[1]

        # weights in Barycentric coordinates
        q = ray_direction × e2
        a = e1'*q
        
        s = ray_origin - triangle[1]
        r = s × e1
        weights = [0; s'*q/a; ray_direction'*r/a]
        weights[1] = 1. - (weights[2]+weights[3])
    
        dist = e2'*r/a
    
        ϵ = 1e-7
        if (a <= ϵ) || sum(weights .< -ϵ) > 0 || (dist <= 0)
            return false
        else
            return true
        end
    end

    # transform coords by pose
    coords = copy(GeometryBasics.coordinates(cad_m))
    for (i,c) in enumerate(coords)
        coords[i] = R*c + t
    end
    faces = GeometryBasics.faces(cad_m)

    # build mask
    for i in bbox[1,1]:bbox[1,2]
        for j in bbox[2,1]:bbox[2,2]
            if mask[j,i]
                continue
            end

            # compute ray direction
            ray_direction = normalize(inv(camK)*[i;j;1])
            ray_origin = zeros(3)

            # check intersection with face
            for face in faces
                if check_triangle(face, coords, ray_origin, ray_direction)
                    mask[j, i] = true
                    break
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