#!/usr/bin/env bash
# Evaluate the best-on-validation checkpoint of every finished run on the held-out test.list
# and write per-case metrics to results/<protocol>/<model>_<setting>_seed<seed>.csv.
# Runs that already have a CSV are skipped (Kaggle runs bring theirs from kaggle_train.sh).
#
# Run from the code/ directory:
#   bash ../scripts/evaluate_all.sh
#   MODELS=unet_3D bash ../scripts/evaluate_all.sh
set -euo pipefail

PROTOCOL="${PROTOCOL:-v2}"
MAX_ITERS="${MAX_ITERS:-20000}"
MODELS="${MODELS:-unet_3D swin_umamba}"
OUT="../results/${PROTOCOL}"
mkdir -p "$OUT"

for run_dir in ../models/PancreasCT_"${PROTOCOL}"_*/*_max_iterations"${MAX_ITERS}"_seed*; do
    [ -d "$run_dir" ] || continue
    run=$(basename "$run_dir")
    case "$run" in
        UNET_3D_*)     model=unet_3D ;;
        SWIN_UMAMBA_*) model=swin_umamba ;;
        *) continue ;;
    esac
    case " $MODELS " in *" $model "*) ;; *) continue ;; esac

    setting=$(basename "$(dirname "$run_dir")")
    setting=${setting#PancreasCT_"${PROTOCOL}"_}
    seed=${run##*_seed}
    csv="$OUT/${model}_${setting}_seed${seed}.csv"

    if [ -f "$csv" ]; then
        echo "Already evaluated: $csv"
        continue
    fi
    if ! grep -q "Training finished" "$run_dir/log.txt" 2>/dev/null; then
        echo "Skipping unfinished run: $run_dir"
        continue
    fi
    echo "=== ${model}, ${setting}, seed ${seed}"
    MODEL="$model" bash ../scripts/evaluate_pancreas_lcc.sh "$run_dir/${model}_best_model.pth" --csv "$csv"
done
