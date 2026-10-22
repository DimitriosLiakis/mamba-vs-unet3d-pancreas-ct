#!/usr/bin/env bash
# Run the Swin-UMamba experiments of the v2 protocol on a local GPU with 8 GB, one setting after
# the other, with each chunk of slices recomputed in the backward pass (USE_CHECKPOINT=1).
# A call trains for at most HOURS (no limit if unset) and can also be stopped at any time. The
# next call continues from the latest checkpoint, and finished runs are skipped.
#
# Run from the code/ directory, in the pancreas-swin environment:
#   bash ../scripts/run_local_swin.sh            # until every run is done
#   HOURS=8 bash ../scripts/run_local_swin.sh    # stop after 8 hours, continue with the same command
set -euo pipefail

SEED="${SEED:-1337}"
SETTINGS="${SETTINGS:-dycon sup12 full}"
PROTOCOL="${PROTOCOL:-v2}"
MAX_ITERS="${MAX_ITERS:-20000}"
HOURS="${HOURS:-}"

VMAMBA=./data/pretrained/vmamba/vmamba_tiny_e292.pth
if [ ! -f "$VMAMBA" ]; then
    mkdir -p "$(dirname "$VMAMBA")"
    wget -q -O "$VMAMBA" "https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth"
fi

deadline=""
if [ -n "$HOURS" ]; then
    deadline=$(( $(date +%s) + $(awk -v h="$HOURS" 'BEGIN { print int(h * 3600) }') ))
fi

for setting in $SETTINGS; do
    run_dir=$(find "../models/PancreasCT_${PROTOCOL}_${setting}" -maxdepth 1 -type d \
              -name "SWIN_UMAMBA_*_max_iterations${MAX_ITERS}_seed${SEED}" 2>/dev/null | head -1 || true)
    if [ -n "$run_dir" ] && grep -q "Training finished" "$run_dir/log.txt" 2>/dev/null; then
        echo "Skipping finished run: $run_dir"
        continue
    fi
    load_path=""
    if [ -n "$run_dir" ] && [ -f "$run_dir/latest_model.pth" ]; then
        load_path="$run_dir/latest_model.pth"
    fi
    echo "=== Swin-UMamba, ${setting}, seed ${SEED}${load_path:+ (resuming)}"
    cmd=(env MODEL=swin_umamba PROTOCOL="$PROTOCOL" SETTING="$setting" SEED="$SEED" MAX_ITERS="$MAX_ITERS"
         USE_CHECKPOINT=1 LOAD_PATH="$load_path" bash ../scripts/train_pancreas.sh)

    if [ -z "$deadline" ]; then
        "${cmd[@]}"
        continue
    fi
    left=$(( deadline - $(date +%s) ))
    if [ "$left" -lt 600 ]; then
        echo "Time is up. Run the same command again to continue."
        exit 0
    fi
    status=0
    timeout "${left}s" "${cmd[@]}" || status=$?
    if [ "$status" -eq 124 ]; then
        echo "Time is up. Run the same command again to continue."
        exit 0
    elif [ "$status" -ne 0 ]; then
        exit "$status"
    fi
done
