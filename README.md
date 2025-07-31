# Pose Uncertainty Sets
Julia code for Pose Uncertainty Sets

## Quick Start
- data folder


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
1. Keypoint detection
2. `exp_estpose`
3. `exp_bounds` (TODO)
4. `exp_coverage` (TODO)


## Setting up the data folder
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