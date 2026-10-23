#!/usr/bin/env bash
# Train on PancreasCT with the thesis settings (one 16 GB GPU): 64^3 patches,
# gradient accumulation over 4 steps, AdamW at 1e-4, 20k iterations.
#
# PROTOCOL picks the data protocol:
#   v2      (default) train_v2.list (0001-0054), best checkpoint on val.list (0055-0062),
#           FeCL masks from the teacher, no labels of unlabeled scans in the data loader
#   thesis  the flags of the thesis runs: train.list (0001-0062), best checkpoint on test1.list
#           (0001-0020), FeCL with the ground-truth masks of the unlabeled scans
#
# SETTING picks the labels:
#   dycon   DyCON: 12 labeled scans + the rest of the list unlabeled (default)
#   sup12   the 12 labeled scans only
#   full    every scan of the list labeled (54 in v2, 62 in the thesis protocol)
#
# Run from the code/ directory:
#   bash ../scripts/train_pancreas.sh                                      # Swin-UMamba + DyCON, v2
#   MODEL=unet_3D SETTING=sup12 SEED=1338 bash ../scripts/train_pancreas.sh
#   PROTOCOL=thesis SETTING=full MAX_ITERS=10000 bash ../scripts/train_pancreas.sh
#   USE_PRETRAIN=0 bash ../scripts/train_pancreas.sh                       # Swin-UMamba from random weights
#
# Resume an interrupted run with LOAD_PATH=../models/<exp>/<run>/latest_model.pth
set -euo pipefail

MODEL="${MODEL:-swin_umamba}"        # swin_umamba | unet_3D
PROTOCOL="${PROTOCOL:-v2}"           # v2 | thesis
SETTING="${SETTING:-dycon}"          # dycon | sup12 | full
SEED="${SEED:-1337}"
MAX_ITERS="${MAX_ITERS:-20000}"
USE_PRETRAIN="${USE_PRETRAIN:-1}"    # Swin-UMamba only: VMamba-tiny ImageNet encoder weights (0 = random init)
KEEP_TOP_K="${KEEP_TOP_K:-5}"        # best iter_*.pth checkpoints kept on disk
USE_CHECKPOINT="${USE_CHECKPOINT:-0}"  # Swin-UMamba: 1 recomputes activations to fit GPUs with less memory
GPU_ID="${GPU_ID:-0}"
LOAD_PATH="${LOAD_PATH:-}"

case "$PROTOCOL" in
    v2)     TRAIN_LIST=train_v2.list; VAL_LIST=val.list;   FECL_MASK=teacher; HIDE_LABELS=1; NUM_TRAIN=54 ;;
    thesis) TRAIN_LIST=train.list;    VAL_LIST=test1.list; FECL_MASK=gt;      HIDE_LABELS=0; NUM_TRAIN=62 ;;
    *) echo "Unknown PROTOCOL: $PROTOCOL (v2 or thesis)" >&2; exit 1 ;;
esac

case "$SETTING" in
    dycon) LABELNUM="${LABELNUM:-12}";         LABELED_BS=1 ;;
    sup12) LABELNUM="${LABELNUM:-12}";         LABELED_BS=2 ;;
    full)  LABELNUM="${LABELNUM:-$NUM_TRAIN}"; LABELED_BS=2 ;;
    *) echo "Unknown SETTING: $SETTING (dycon, sup12 or full)" >&2; exit 1 ;;
esac

if [ "$LABELNUM" -gt "$NUM_TRAIN" ] || { [ "$SETTING" = "dycon" ] && [ "$LABELNUM" -ge "$NUM_TRAIN" ]; }; then
    echo "LABELNUM=$LABELNUM does not fit SETTING=$SETTING with $TRAIN_LIST ($NUM_TRAIN scans)." >&2
    exit 1
fi

EXP="${EXP:-PancreasCT_${PROTOCOL}_${SETTING}}"

VMAMBA_CKPT=./data/pretrained/vmamba/vmamba_tiny_e292.pth
if [ "$MODEL" = "swin_umamba" ] && [ "$USE_PRETRAIN" = "1" ] && [ ! -f "$VMAMBA_CKPT" ]; then
    echo "VMamba weights not found at code/${VMAMBA_CKPT#./} (see the README, Setup)." >&2
    exit 1
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True

exec python train_DyCON_Pancreas.py \
    --root_dir ../data/Pancreas \
    --train_list "$TRAIN_LIST" \
    --val_list "$VAL_LIST" \
    --exp "$EXP" \
    --model "$MODEL" \
    --seed "$SEED" \
    --gpu_id "$GPU_ID" \
    --patch_size 64 64 64 \
    --max_iterations "$MAX_ITERS" \
    --batch_size 2 \
    --labeled_bs "$LABELED_BS" \
    --labelnum "$LABELNUM" \
    --accumulation_steps 4 \
    --optimizer adamw \
    --base_lr 1e-4 \
    --chunk_size 8 \
    --use_checkpoint "$USE_CHECKPOINT" \
    --feature_scaler 2 \
    --temp 0.6 \
    --u_weight 0.5 \
    --consistency 0.1 \
    --use_teacher_loss 1 \
    --use_focal 1 \
    --fecl_mask "$FECL_MASK" \
    --hide_unlabeled_labels "$HIDE_LABELS" \
    --keep_top_k "$KEEP_TOP_K" \
    --use_pretrain "$USE_PRETRAIN" \
    --pretrained_ckpt "$VMAMBA_CKPT" \
    ${LOAD_PATH:+--load_path "$LOAD_PATH"}
