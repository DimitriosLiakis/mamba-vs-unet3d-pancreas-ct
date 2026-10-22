#!/usr/bin/env bash
# Run the 3D U-Net experiments of the v2 protocol on a local GPU, one after the other:
# every setting (sup12, dycon, full) for every seed. Finished runs are skipped, and an
# interrupted run continues from its latest_model.pth.
#
# Run from the code/ directory:
#   bash ../scripts/run_local_unet.sh
#   SEEDS="1337" SETTINGS="dycon" bash ../scripts/run_local_unet.sh
#   SETTINGS="dycon" MAX_ITERS=40000 TAG=40k bash ../scripts/run_local_unet.sh   # runs go to PancreasCT_v2_dycon40k
set -euo pipefail

SEEDS="${SEEDS:-1337 1338 1339}"
SETTINGS="${SETTINGS:-sup12 dycon full}"
PROTOCOL="${PROTOCOL:-v2}"
MAX_ITERS="${MAX_ITERS:-20000}"
TAG="${TAG:-}"                       # appended to the experiment folder name

for seed in $SEEDS; do
    for setting in $SETTINGS; do
        exp="PancreasCT_${PROTOCOL}_${setting}${TAG}"
        run_dir=$(find "../models/${exp}" -maxdepth 1 -type d \
                  -name "UNET_3D_*_max_iterations${MAX_ITERS}_seed${seed}" 2>/dev/null | head -1 || true)
        if [ -n "$run_dir" ] && grep -q "Training finished" "$run_dir/log.txt" 2>/dev/null; then
            echo "Skipping finished run: $run_dir"
            continue
        fi
        load_path=""
        if [ -n "$run_dir" ] && [ -f "$run_dir/latest_model.pth" ]; then
            load_path="$run_dir/latest_model.pth"
        fi
        echo "=== 3D U-Net, ${setting}, seed ${seed}${load_path:+ (resuming)}"
        MODEL=unet_3D PROTOCOL="$PROTOCOL" SETTING="$setting" SEED="$seed" MAX_ITERS="$MAX_ITERS" EXP="$exp" \
            LOAD_PATH="$load_path" bash ../scripts/train_pancreas.sh
    done
done
