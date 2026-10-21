#!/usr/bin/env bash
# One Kaggle session of a v2 run (GPU T4, internet on, the pancreas-data dataset attached).
#
# Notebook cells:
#   !git clone -b v2 https://github.com/DimitriosLiakis/mamba-vs-unet3d-pancreas-ct /kaggle/working/repo
#   !SETTING=dycon bash /kaggle/working/repo/scripts/kaggle_train.sh
# then "Save Version". To continue the run in a new session, add that version's output as input and set
#   RESUME_FROM=/kaggle/input/<that version>/repo/models
#
# Training stops SESSION_HOURS after the script starts, setup included, so the session ends before
# Kaggle's 12-hour limit and the version and its checkpoints are saved. Once the run reaches its last
# iteration, its best checkpoint is evaluated on test.list and the per-case metrics go to
# repo/results/v2/.
#
# Variables, with defaults:
#   MODEL=swin_umamba   SETTING=dycon (dycon | sup12 | full)   SEED=1337   PROTOCOL=v2   MAX_ITERS=20000
#   DATA_DIR=/kaggle/input/datasets/jimliakis/pancreas-data    folder with the PANCREAS_*.h5 volumes
#   LOCAL_DATA=/tmp/pancreas_data   local copy of the volumes that training reads
#   WHEELS_DIR=      saved causal-conv1d / mamba-ssm wheels. Without it they are built (slow) into
#                    /kaggle/working/wheels. Add that folder as a dataset to skip the build next time.
#   RESUME_FROM=     models/ folder of an earlier session of this run
#   SESSION_HOURS=11.25   time budget of this session from the start of the script, setup included
#   EVALUATE=1       evaluate on test.list once training has finished
#   DRY_RUN=0        1 prints the commands without running them
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-swin_umamba}"
SETTING="${SETTING:-dycon}"
SEED="${SEED:-1337}"
PROTOCOL="${PROTOCOL:-v2}"
MAX_ITERS="${MAX_ITERS:-20000}"
DATA_DIR="${DATA_DIR:-/kaggle/input/datasets/jimliakis/pancreas-data}"
LOCAL_DATA="${LOCAL_DATA:-/tmp/pancreas_data}"
WHEELS_DIR="${WHEELS_DIR:-}"
RESUME_FROM="${RESUME_FROM:-}"
SESSION_HOURS="${SESSION_HOURS:-11.25}"
EVALUATE="${EVALUATE:-1}"
DRY_RUN="${DRY_RUN:-0}"

run() {
    echo "+ $*"
    if [ "$DRY_RUN" != "1" ]; then "$@"; fi
}

find_run_dir() {
    find "../models/PancreasCT_${PROTOCOL}_${SETTING}" -maxdepth 1 -type d \
        -name "${MODEL^^}_*_max_iterations${MAX_ITERS}_seed${SEED}" 2>/dev/null | head -1 || true
}

# 1. Python packages
run pip install -q -r "$REPO/requirements.txt"
if [ "$MODEL" = "swin_umamba" ]; then
    if [ -n "$WHEELS_DIR" ] && compgen -G "$WHEELS_DIR/*.whl" > /dev/null; then
        run pip install -q --no-deps "$WHEELS_DIR"/*.whl
    else
        run pip install -q ninja packaging wheel
        run pip wheel --no-deps --no-build-isolation -w /kaggle/working/wheels causal-conv1d==1.6.1 mamba-ssm==2.3.1
        run pip install -q --no-deps /kaggle/working/wheels/*.whl
    fi
fi
run python -c "import torch, numpy; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '| numpy', numpy.__version__)"

# 2. Machine, data and VMamba weights
run nproc
run free -g
run nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv,noheader

# Copy the volumes to local disk once. In a committed session, training read them from
# /kaggle/input and stalled for up to two minutes at a time.
copy_start=$SECONDS
run mkdir -p "$LOCAL_DATA" "$REPO/data/Pancreas/Pancreas_data"
run find "$DATA_DIR" -maxdepth 2 -name '*.h5' -exec cp {} "$LOCAL_DATA/" \;
echo "Copied the volumes to $LOCAL_DATA in $((SECONDS - copy_start)) s."
run find "$LOCAL_DATA" -maxdepth 1 -name '*.h5' -exec ln -sf {} "$REPO/data/Pancreas/Pancreas_data/" \;
if [ "$DRY_RUN" != "1" ]; then
    n=$(find "$REPO/data/Pancreas/Pancreas_data" -name '*.h5' | wc -l)
    if [ "$n" -lt 82 ]; then
        echo "Found $n of 82 volumes under $DATA_DIR (set DATA_DIR)." >&2
        exit 1
    fi
fi

VMAMBA="$REPO/code/data/pretrained/vmamba/vmamba_tiny_e292.pth"
if [ "$MODEL" = "swin_umamba" ] && [ ! -f "$VMAMBA" ]; then
    run mkdir -p "$(dirname "$VMAMBA")"
    run wget -q -O "$VMAMBA" "https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth"
fi

# 3. Checkpoints of earlier sessions
cd "$REPO/code"
if [ -n "$RESUME_FROM" ]; then
    run mkdir -p ../models
    run cp -rn "$RESUME_FROM"/. ../models/
fi

run_dir="$(find_run_dir)"
finished=0
load_path=""
if [ -n "$run_dir" ] && grep -q "Training finished" "$run_dir/log.txt" 2>/dev/null; then
    echo "Training already finished: $run_dir"
    finished=1
elif [ -n "$run_dir" ] && [ -f "$run_dir/latest_model.pth" ]; then
    load_path="$run_dir/latest_model.pth"
    echo "Resuming from $load_path"
fi

# 4. Train until the run ends or the session time is up
if [ "$finished" != "1" ]; then
    # Bash's SECONDS counts from the start of this script, so the setup time is subtracted.
    train_seconds=$(awk -v h="$SESSION_HOURS" -v s="$SECONDS" 'BEGIN { print int(h * 3600 - s) }')
    if [ "$train_seconds" -lt 600 ]; then
        echo "Less than 10 minutes of the session budget left after setup. Start a new session." >&2
        exit 1
    fi
    echo "Training for at most $((train_seconds / 60)) minutes in this session."
    set +e
    run timeout "${train_seconds}s" env MODEL="$MODEL" SETTING="$SETTING" SEED="$SEED" PROTOCOL="$PROTOCOL" \
        MAX_ITERS="$MAX_ITERS" KEEP_TOP_K=2 LOAD_PATH="$load_path" bash ../scripts/train_pancreas.sh
    status=$?
    set -e
    if [ "$status" -eq 124 ]; then
        echo "Session time is up. Save this version, then continue with RESUME_FROM=<its output>/repo/models."
        exit 0
    elif [ "$status" -ne 0 ]; then
        exit "$status"
    fi
fi

# 5. Evaluate the best checkpoint on the held-out test.list
if [ "$EVALUATE" = "1" ]; then
    run_dir="$(find_run_dir)"
    csv="../results/${PROTOCOL}/${MODEL}_${SETTING}_seed${SEED}.csv"
    if [ -f "$csv" ]; then
        echo "Already evaluated: $csv"
    else
        run mkdir -p "../results/${PROTOCOL}"
        run env MODEL="$MODEL" bash ../scripts/evaluate_pancreas_lcc.sh "$run_dir/${MODEL}_best_model.pth" --csv "$csv"
    fi
fi
