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