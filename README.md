# Pose Uncertainty Sets
Julia code for Pose Uncertainty Sets

**TODO: nice figure / animation here**

## Quick Start
TODO
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
<details closed>

<summary><b>Keypoint Detection</b></summary>



```shell

```

</details>


<details closed>

<summary><b>Pose Estimation</b></summary>



```shell

```

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