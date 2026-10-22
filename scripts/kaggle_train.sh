#!/usr/bin/env bash
# One Kaggle session of v2 runs (GPU T4 x2, internet on, the pancreas-data dataset attached).
# Each setting in SETTINGS trains on its own GPU, so "GPU T4 x2" runs two settings at once.
#
# Notebook cells:
#   !git clone -b v2 https://github.com/DimitriosLiakis/mamba-vs-unet3d-pancreas-ct /kaggle/working/repo
#   !SETTINGS="dycon sup12" bash /kaggle/working/repo/scripts/kaggle_train.sh
# then "Save Version". To continue in a new session, add that version's output as input (Add Input >
# Your Work) and run the same cells. The checkpoints and the wheels are found under /kaggle/input.
#
# Training stops SESSION_HOURS after the script starts, setup included, so the session ends before
# Kaggle's 12-hour limit and the version and its checkpoints are saved. Once a run reaches its last
# iteration, its best checkpoint is evaluated on test.list and the per-case metrics go to
# repo/results/v2/. Each run writes its output to repo/models/session_<setting>.log, and this log
# shows the latest validation of every run each 10 minutes.
#
# Variables, with defaults:
#   MODEL=swin_umamba   SETTINGS=dycon (any of dycon, sup12, full, at most one per GPU)
#   SEED=1337   PROTOCOL=v2   MAX_ITERS=20000   (SETTING=... is accepted for a single setting)
#   DATA_DIR=/kaggle/input/datasets/jimliakis/pancreas-data    folder with the PANCREAS_*.h5 volumes
#   LOCAL_DATA=/tmp/pancreas_data   local copy of the volumes that training reads
#   WHEELS_DIR=      saved causal-conv1d / mamba-ssm wheels. If unset, they are looked for under
#                    /kaggle/input, and built (slow) into /kaggle/working/wheels if none are found.
#   RESUME_FROM=     models/ folder of earlier sessions. If unset, it is looked for under /kaggle/input,
#                    so attach only the output of the latest version.
#   SESSION_HOURS=11.25   time budget of this session from the start of the script, setup included
#   GPU_IDS=         GPUs for the settings, in order (default: 0, 1, ...)
#   EVALUATE=1       evaluate on test.list once training has finished
#   DRY_RUN=0        1 prints the commands without running them
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-swin_umamba}"
SETTINGS="${SETTINGS:-${SETTING:-dycon}}"
SEED="${SEED:-1337}"
PROTOCOL="${PROTOCOL:-v2}"
MAX_ITERS="${MAX_ITERS:-20000}"
DATA_DIR="${DATA_DIR:-/kaggle/input/datasets/jimliakis/pancreas-data}"
LOCAL_DATA="${LOCAL_DATA:-/tmp/pancreas_data}"
WHEELS_DIR="${WHEELS_DIR:-}"
RESUME_FROM="${RESUME_FROM:-}"
SESSION_HOURS="${SESSION_HOURS:-11.25}"
GPU_IDS="${GPU_IDS:-}"
EVALUATE="${EVALUATE:-1}"
DRY_RUN="${DRY_RUN:-0}"

run() {
    echo "+ $*"
    if [ "$DRY_RUN" != "1" ]; then "$@"; fi
}

find_run_dir() {  # $1: setting
    find "../models/PancreasCT_${PROTOCOL}_$1" -maxdepth 1 -type d \
        -name "${MODEL^^}_*_max_iterations${MAX_ITERS}_seed${SEED}" 2>/dev/null | head -1 || true
}

if [ -z "$GPU_IDS" ]; then
    n_gpus=$(nvidia-smi -L 2>/dev/null | wc -l || true)
    GPU_IDS=$(seq -s ' ' 0 $(( ${n_gpus:-1} > 0 ? ${n_gpus:-1} - 1 : 0 )))
fi
read -r -a settings <<< "$SETTINGS"
read -r -a gpus <<< "$GPU_IDS"
if [ "${#settings[@]}" -gt "${#gpus[@]}" ]; then
    echo "SETTINGS has ${#settings[@]} runs but only ${#gpus[@]} GPU(s) are available (GPU_IDS: $GPU_IDS)." >&2
    exit 1
fi

# 0. Outputs of earlier sessions attached as input
if [ -z "$WHEELS_DIR" ]; then
    whl=$(find /kaggle/input -maxdepth 7 -name 'mamba_ssm-*.whl' 2>/dev/null | head -1 || true)
    if [ -n "$whl" ]; then WHEELS_DIR="$(dirname "$whl")"; echo "Wheels found in $WHEELS_DIR"; fi
fi
resume_dirs=()
if [ -n "$RESUME_FROM" ]; then
    resume_dirs=("$RESUME_FROM")
else
    for setting in "${settings[@]}"; do
        latest=$(find /kaggle/input -maxdepth 10 -name latest_model.pth \
                 -path "*/PancreasCT_${PROTOCOL}_${setting}/${MODEL^^}_*_max_iterations${MAX_ITERS}_seed${SEED}/*" \
                 2>/dev/null | head -1 || true)
        if [ -n "$latest" ]; then
            resume_dirs+=("$(dirname "$(dirname "$(dirname "$latest")")")")
            echo "Checkpoints of ${setting} found in ${resume_dirs[-1]}"
        fi
    done
fi

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
LINKS="$REPO/data/Pancreas/Pancreas_data"
# The links go into a new folder. Where that folder is a symlink or already holds volumes, as in a
# local copy of the repository, linking would replace the real files, so stop instead.
if [ -L "$LINKS" ] || [ -n "$(find "$LINKS" -maxdepth 1 -type f -name '*.h5' -print -quit 2>/dev/null)" ]; then
    echo "$LINKS already holds the data. kaggle_train.sh is meant for a fresh Kaggle machine." >&2
    exit 1
fi
copy_start=$SECONDS
run mkdir -p "$LOCAL_DATA" "$LINKS"
run find "$DATA_DIR" -maxdepth 2 -name '*.h5' -exec cp {} "$LOCAL_DATA/" \;
echo "Copied the volumes to $LOCAL_DATA in $((SECONDS - copy_start)) s."
run find "$LOCAL_DATA" -maxdepth 1 -name '*.h5' -exec ln -sf {} "$LINKS/" \;
if [ "$DRY_RUN" != "1" ]; then
    n=$(find "$LINKS" -name '*.h5' | wc -l)
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
run mkdir -p ../models
for dir in "${resume_dirs[@]}"; do
    run cp -rn "$dir"/. ../models/
done

# Bash's SECONDS counts from the start of this script, so the setup time is subtracted.
train_seconds=$(awk -v h="$SESSION_HOURS" -v s="$SECONDS" 'BEGIN { print int(h * 3600 - s) }')

# 4. Train each setting on its own GPU until it ends or the session time is up, then evaluate it
train_one() {  # $1: setting, $2: GPU
    local setting=$1 gpu=$2 run_dir load_path="" status=0 log="../models/session_$1.log" csv
    if [ "$DRY_RUN" = "1" ]; then log=/dev/stdout; fi
    run_dir="$(find_run_dir "$setting")"
    if [ -n "$run_dir" ] && grep -q "Training finished" "$run_dir/log.txt" 2>/dev/null; then
        echo "[$setting] training already finished: $run_dir"
    else
        if [ "$train_seconds" -lt 600 ]; then
            echo "[$setting] less than 10 minutes of the session budget left after setup." >&2
            return 1
        fi
        if [ -n "$run_dir" ] && [ -f "$run_dir/latest_model.pth" ]; then
            load_path="$run_dir/latest_model.pth"
            echo "[$setting] resuming from $load_path"
        fi
        echo "[$setting] training on GPU $gpu for at most $((train_seconds / 60)) minutes, output in $log"
        run timeout "${train_seconds}s" env MODEL="$MODEL" SETTING="$setting" SEED="$SEED" PROTOCOL="$PROTOCOL" \
            MAX_ITERS="$MAX_ITERS" KEEP_TOP_K=2 GPU_ID="$gpu" LOAD_PATH="$load_path" \
            bash ../scripts/train_pancreas.sh > "$log" 2>&1 || status=$?
        if [ "$status" -eq 124 ]; then
            echo "[$setting] session time is up. Save this version and run it again with its output as input."
            return 0
        elif [ "$status" -ne 0 ]; then
            echo "[$setting] training failed with exit code $status, see $log" >&2
            return "$status"
        fi
    fi
    if [ "$EVALUATE" = "1" ]; then
        run_dir="$(find_run_dir "$setting")"
        csv="../results/${PROTOCOL}/${MODEL}_${setting}_seed${SEED}.csv"
        if [ -f "$csv" ]; then
            echo "[$setting] already evaluated: $csv"
        else
            run mkdir -p "../results/${PROTOCOL}"
            run env CUDA_VISIBLE_DEVICES="$gpu" MODEL="$MODEL" \
                bash ../scripts/evaluate_pancreas_lcc.sh "$run_dir/${MODEL}_best_model.pth" --csv "$csv" >> "$log" 2>&1
            echo "[$setting] test.list metrics written to $csv"
        fi
    fi
}

pids=()
for i in "${!settings[@]}"; do
    train_one "${settings[$i]}" "${gpus[$i]}" &
    pids+=($!)
done

any_alive() {
    local p
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done
    return 1
}

# Every 10 minutes, the latest validation of each run.
minutes=0
while [ "$DRY_RUN" != "1" ] && any_alive; do
    sleep 60
    minutes=$((minutes + 1))
    if [ $((minutes % 10)) -eq 0 ]; then
        for setting in "${settings[@]}"; do
            line=$(grep -hE 'Iteration [0-9]+ :' "$(find_run_dir "$setting")/log.txt" 2>/dev/null | tail -1 || true)
            echo "[$setting] ${line:-starting}"
        done
    fi
done

status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
exit "$status"
