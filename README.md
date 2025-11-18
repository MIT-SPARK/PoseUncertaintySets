# Uncertainty Quantification for Visual Object Pose Estimation
<h3 align="center"><a href="TODO"> Paper</a> | <a href="TODO">Video</a> | <a href="https://drive.google.com/file/d/1-Hpl3zb1hX3p-uaOD5r-ZhoLCqYKJRII/view?usp=sharing">Data</a></h3>


  90% confidence | 60% confidence
:-------------------------:|:-------------------------:
![](assets/lmo_qual.png)|![](assets/lmo_qual2.png)

This is the project repo for S-Lemma Uncertainty Estimation (SLUE). SLUE is a fast and statistically rigorous way to estimate pose uncertainty. Given a pose estimate and conformal uncertainty bounds on object keypoints, we compute an ellipsoidal bound on pose uncertainty for a given confidence. The shaded bounds above show the 90% and 60% confidence bounds for each object, projected onto the image plane. Pose uncertainty can be expressed as a single ellipsoid which is joint in rotation and translation, or independent ellipsoids for each quantity. These ellipsoidal bounds are easy to incorporate into downstream optimization for further reasoning.

The main *theoretical* contribution of the paper is a sum-of-squares approach (inspired by the S-lemma) to efficiently bound semialgebraic sets with ellipsoids (see the [2D demos](scripts/demo/demo_2d.jl)). The main *practical* contribution is the application to pose uncertainty, which requires first running conformal prediction on object keypoints. We find that synthetic data, if appropriately generated, can satisfy the exchangability requirements of conformal prediction.


## Quick Start
First, make sure you have [Julia installed](https://julialang.org/install/). This repository was tested with v1.11.6. Then, clone the repository and follow the directions below. We assume you are in the repo folder.
1. Clone this repository
```shell
git clone https://github.com/lopenguin/PoseUncertaintySets.git
cd PoseUncertaintySets
```
2. Open the Julia REPL
```shell
julia --project
```
3. Install dependencies
```julia-repl
] # enters pkg> mode
add https://github.com/lopenguin/SimpleRotations.jl https://github.com/lopenguin/TSSOSMinimal.git https://github.com/lopenguin/P3P.jl.git
```
*You may also need to get a [MOSEK license](https://www.mosek.com/products/academic-licenses/). These are available for free to academic users.*

4. Ellipsoidal bounds with 2D data
```julia-repl
# press backspace to return to the main REPL.
include("scripts/demo/demo_2d.jl")
```
It may take a while to run the first time, but try running it again (should be much faster!). This will produce a plot like the one below:

<p align="center">
  <img src="assets/2d_demo.png" />
</p>

## Reproduce results
We assume you are in the home directory of this repository and you've been through the quick start step. 

First, download the data folder. We first move to the directory containing `PoseUncertaintySets`, and then pull data off of Google drive.
```shell
cd ..
wget -O data.zip "https://drive.usercontent.google.com/download?id=1-Hpl3zb1hX3p-uaOD5r-ZhoLCqYKJRII&export=download&confirm=yes"
unzip data.zip -d data
rm data.zip
```
You can also download [from Google drive](https://drive.google.com/file/d/1-Hpl3zb1hX3p-uaOD5r-ZhoLCqYKJRII/view?usp=sharing) and put it in a "data" folder.

This should give a directory structure which looks like:
```
├── data
├── PoseUncertaintySets
```

You can now run the method with CAST. Try:
```shell
julia --project
include("scripts/demo/demo_R.jl")
include("scripts/demo/demo_quat.jl")
```

You can also see the results for LM-O, YCB-V, and CAST. Just use any of the `summarize.jl` scripts or see the more details section. To actually run the method on LM-O or YCB-V, you need to download their data.

### Run with LM-O
For LM-O, download `all test images`, `BOP test images`, and `object models` from [BOP](https://bop.felk.cvut.cz/datasets/#LM-O). Unzip. Place all test images as `data/lmo/test`. Place BOP test images as `data/lmo/cal`. Place object models as `data/lmo/models_eval`.

Change the dataset in `demo_R.jl` to `lmo` to test.

### Run with YCB-V
For YCB-V, download `BOP test images` and `object models` from [BOP](https://bop.felk.cvut.cz/datasets/#YCB-V). Unzip. Place all test images as `data/ycbv/test`. Place object models as `data/ycbv/models_eval`. We provide synthetic calibration images.

Change the dataset in `demo_R.jl` to `ycbv` to test.

### More details


<details closed>

<summary><b>Keypoint Detection</b></summary>

For BOP keypoints, clone the [bop-keypoints repo](https://github.com/lopenguin/bop-keypoints) and follow the instructions in the README to setup and run keypoint detection on your dataset of choice. You'll need to run it on lmo and ycbv for the full test split. For lmo only, you need to run on the test and cal splits.

We do not release the CAST keypoint detector.

</details>


<details closed>

<summary><b>Pose Estimation</b></summary>

```shell
# run certifiable PnP
julia --project scripts/poses/pnp_pose.jl 
# run RANSAG (sample-based approach)
julia --project scripts/poses/ransag_pose.jl 
# produce the table in appendix
julia --project scripts/poses/summarize.jl 
```
The following options are available for pose estimation:
- `dataset ∈ {"lmo", "ycbv", "cast"}`: dataset to use
- `p ∈ {2,Inf}`: p-norm uncertainty set
- `α ∈ (0, 1)`: conformal confidence

(next time I will incorporate argparse)

</details>


<details closed>

<summary><b>Ellipsoids, Uncertainty Bounds, and Runtime</b></summary>

```shell
# S-Lemma (first order / rotation matrix)
julia --project scripts/ellipses/slem_rotm.jl 
# S-Lemma (second order / quaternion)
julia --project scripts/ellipses/slem_quat.jl 
# RANSAG (first order, can do higher order)
julia --project scripts/ellipses/ransag_bounds.jl 
# produce the table
julia --project scripts/ellipses/summarize.jl 
```
The following options are available for pose estimation:
- `dataset ∈ {"lmo", "ycbv", "cast"}`: dataset to use
- `p ∈ {2,Inf}`: p-norm uncertainty set (only `Inf` for quat)
- `α ∈ (0, 1)`: conformal confidence
- `order ∈ {1,2,...}`: relaxation order (only `2+` for quat)
- `pose ∈ {"ransag", "pnp1", "pnp2", "maxmargin"}`: source of pose estimate / center

</details>



<details closed>

<summary><b>Conformal Coverage</b></summary>

```shell
# estimate the coverage for a specific confidence / dataset
julia --project scripts/coverage.jl
```
The following options are available:
- `dataset ∈ {"lmo", "ycbv", "cast"}`: dataset to use
- `p ∈ {2,Inf}`: p-norm uncertainty set (only `Inf` for quat)
- `α ∈ (0, 1)`: conformal confidence

Note that the pose estimate choice must match the pose estimate used to generate the bounding ellipse. All experiments in the paper use `pnp2`. This script will throw errors if the slemma / pose data files are not present.

</details>



<details closed>

<summary><b>Visualizations</b></summary>

Visualize keypoints on an image:
```shell
# plot on a single image
julia --project scripts/visualize/plot_keypoints.jl
# or generate a video
julia --project scripts/visualize/video.jl
```

Visualize ellipsoids, with interactivity:
```shell
# rotation
julia --project scripts/visualize/plot_r_ellipses.jl
# translation
julia --project scripts/visualize/plot_t_ellipses.jl
```

Visualize the second order ellipsoids projected onto image plane:
```shell
julia --project scripts/plot_uncertainty_on_img.jl
```

</details>


<details closed>

<summary><b>Compare with GRCC</b></summary>

To compare with GRCC we use their [official MATLAB implementation](https://github.com/Negotch/GRCC-code).

You can export data for GRCC using `scripts/ellipses/grcc_export.jl`. The results from running GRCC are in the `dataset_grcc.mat` files.

</details>


## References
- H. Yang and M. Pavone, "Object pose estimation with statistical
guarantees: Conformal keypoint detection and geometric uncertainty
propagation", 2023. Available: https://arxiv.org/abs/2303.12246.
- Y. Tang, J.-B. Lasserre, and H. Yang, “Uncertainty quantification
of set-membership estimation in control and perception: Revisiting
the minimum enclosing ellipsoid”, 2024. Available: https://proceedings.mlr.press/v242/tang24a.html.
- E. Brachmann, A. Krull, F. Michel, S. Gumhold, J. Shotton, and
C. Rother, “Learning 6d object pose estimation using 3d object coordinates,” 2014. Available: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/PoseEstimationECCV2014.pdf
- Y. Xiang, T. Schmidt, V. Narayanan, and D. Fox, “PoseCNN: A
convolutional neural network for 6D object pose estimation in cluttered
scenes,” 2018. Available: https://arxiv.org/abs/1711.00199.
- L. Shaikewitz, S. Ubellacker, and L. Carlone, “A certifiable algorithm
for simultaneous shape estimation and object tracking,” 2024. Available: https://arxiv.org/abs/2406.16837.
- K. Schmeckpeper et al., “Semantic keypoint-based pose estimation from
single rgb frames,” 2022. Available: https://arxiv.org/abs/2204.05864.

## BibTeX
```
@misc{Shaikewitz25arxiv-PoseUncertaintySets,
      TODO
}
```