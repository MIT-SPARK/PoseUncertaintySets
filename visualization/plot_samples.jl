## Draw samples from pose uncertainty set as mask
# Lorenzo Shaikewitz, 5/6/2025

img_name = @sprintf "%06d.png" parse(Int,img_names[frame])
img = Images.load("/home/lorenzo/Downloads/lmo_test_all/test/000002/rgb/$img_name")


# central pose
Rc = R_ests[frame]
tc = t_ests[frame]
mask_center = get_mask(img, Rc, tc, cad_m, camK)
mask = copy(mask_center)
p1=Plots.plot(img, axis=false, grid=false, title="Frame $frame")

# samples
for (i,pose) in enumerate(poses)
    println(i)
    R = pose[1]
    t = pose[2]
    global mask
    maski = get_mask(img, R, t, cad_m, camK)
    mask = mask .|| maski
end
img2 = Images.RGBA.(copy(img)).*0
img2[mask] .= Images.RGBA(0,1,1, 0.5)
Plots.plot!(img2)

bound = boundary_map(mask_center)
img3 = Images.RGBA.(copy(img)).*0
img3[bound] .= Images.RGBA(0,0,0, 1.)
Plots.plot!(img3)