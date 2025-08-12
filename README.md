# Pose Uncertainty Sets
Julia code for Pose Uncertainty Sets

**TODO: nice figure / animation here**

## Quick Start
TODO:
- Data folder
- Julia environment

### Setting up the data folder
To run experiments you will need a data folder. We show the default file structure below.
```
├── data
│   ├── lmo
│   │   ├── models_eval
│   │   ├── cal
│   │   ├── test
│   │   ├── kpts3d.json
│   │   ├── detections_cal.json
│   │   ├── detections_test.json
│   ├── ycbv
│   ├── cast
├── PoseUncertaintySets
```
TODO: where to find all this / quick download script?


## Organization
```
├── scripts
│   ├── oneframe
├── src
│   ├── datasets
│   ├── visualization
```
- `scripts` has ready-written scripts to reproduce experiments. They may be run directly via `include("scripts/NAME.jl")` in the Julia REPL.
- `src` has functions. At the top level are the core functions for processing a single image.
- `src/datasets` contains functions for processing datasets. These rely on the core functions in `src`.

## Reproduce results
We assume you are in the home directory of this repository and you've been through the quick start step.

<details closed>

<summary><b>Keypoint Detection</b></summary>



```shell

```

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
- `dataset ∈ {"lmo", "ycbv", "drone"}`: dataset to use
- `p ∈ {2,Inf}`: p-norm uncertainty set
- `α ∈ (0, 1)`: conformal confidence

</details>


<details closed>

<summary><b>Ellipsoids and Uncertainty Bounds</b></summary>



```shell

```

</details>



<details closed>

<summary><b>Conformal Coverage</b></summary>



```shell

```

</details>


<details closed>

<summary><b>Runtime</b></summary>



```shell

```

</details>


<summary><b>Visualizations</b></summary>



```shell

```

</details>

## References
- RANSAG
- GRCC?
- LM-O
- YCB-V
- CAST

## BibTeX