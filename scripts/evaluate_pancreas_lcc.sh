#!/usr/bin/env bash
# Evaluate a checkpoint with sliding-window inference (64^3 windows, stride 32)
# and report Dice / HD95 / ASD before and after Largest-Connected-Component filtering.
#
# Run from the code/ directory:
#   bash ../scripts/evaluate_pancreas_lcc.sh <checkpoint.pth>                        # held-out test.list
#   MODEL=unet_3D bash ../scripts/evaluate_pancreas_lcc.sh <checkpoint.pth>
#   TEST_LIST=test1.list bash ../scripts/evaluate_pancreas_lcc.sh <checkpoint.pth>   # thesis protocol
#
# Extra arguments are passed through, e.g. --csv per_case.csv or --save_dir predictions/
set -euo pipefail

MODEL_PATH="${1:?Usage: evaluate_pancreas_lcc.sh <checkpoint.pth> [extra args]}"
shift
MODEL="${MODEL:-swin_umamba}"        # swin_umamba | unet_3D
TEST_LIST="${TEST_LIST:-test.list}"

python evaluate_pancreas_lcc.py \
    --model_path "$MODEL_PATH" \
    --model_type "$MODEL" \
    --data_root ../data/Pancreas \
    --test_list "$TEST_LIST" \
    --patch_size 64 64 64 \
    --stride_xy 32 \
    --stride_z 32 \
    --scaler 2 \
    --chunk_size 8 \
    "$@"
