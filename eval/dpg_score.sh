#!/bin/bash
# Score DPG-Bench images (from eval/dpg_generate.py) with the official scorer, external/ELLA/dpg_bench (mPLUG-large),
# as dist_eval.sh runs it, on one GPU, in env_dpg.
# usage: bash eval/dpg_score.sh <image dir> <results.txt>
set -euo pipefail
IMAGES=$(realpath "$1"); RESULTS=$(realpath -m "$2")
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd $REPO/external/ELLA
$REPO/env_dpg/bin/accelerate launch --num_machines 1 --num_processes 1 --mixed_precision fp16 \
  ./dpg_bench/compute_dpg_bench.py --image-root-path $IMAGES --res-path $RESULTS --resolution 512 --pic-num 4 \
  --vqa-model mplug
