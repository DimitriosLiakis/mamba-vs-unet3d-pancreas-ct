#!/usr/bin/env bash
# Train on PancreasCT with the thesis settings (one 16 GB GPU): 64^3 patches,
# 1 labeled + 1 unlabeled scan per step, gradient accumulation over 4 steps,
# AdamW at 1e-4, 20k iterations, 12 labeled scans.
#
# Run from the code/ directory:
#   bash ../scripts/train_pancreas.sh                                            # Swin-UMamba + DyCON, 12 labels
#   MODEL=unet_3D bash ../scripts/train_pancreas.sh                              # 3D U-Net + DyCON, 12 labels
#   LABELNUM=62 LABELED_BS=2 MAX_ITERS=10000 bash ../scripts/train_pancreas.sh   # fully supervised baseline
#   USE_PRETRAIN=1 bash ../scripts/train_pancreas.sh                             # Swin-UMamba with VMamba weights
#
# Resume an interrupted run with LOAD_PATH=../models/<exp>/<run>/latest_model.pth
set -euo pipefail

MODEL="${MODEL:-swin_umamba}"        # swin_umamba | unet_3D
LABELNUM="${LABELNUM:-12}"           # labeled scans: the first N entries of train.list
LABELED_BS="${LABELED_BS:-1}"        # 2 (= batch size) trains on labeled scans only
MAX_ITERS="${MAX_ITERS:-20000}"
USE_PRETRAIN="${USE_PRETRAIN:-0}"    # Swin-UMamba only: 1 = VMamba-tiny ImageNet encoder weights (the thesis runs used 0)
LOAD_PATH="${LOAD_PATH:-}"

if [ "$LABELED_BS" -ge 2 ]; then
    EXP="${EXP:-PancreasCT_FullSupervised}"
else
    EXP="${EXP:-PancreasCT_DyCON_SemiSupervised}"
fi

VMAMBA_CKPT=./data/pretrained/vmamba/vmamba_tiny_e292.pth
if [ "$MODEL" = "swin_umamba" ] && [ "$USE_PRETRAIN" = "1" ] && [ ! -f "$VMAMBA_CKPT" ]; then
    echo "VMamba weights not found at code/${VMAMBA_CKPT#./} (see the README, Setup)." >&2
    exit 1
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True

python train_DyCON_Pancreas.py \
    --root_dir ../data/Pancreas \
    --exp "$EXP" \
    --model "$MODEL" \
    --patch_size 64 64 64 \
    --max_iterations "$MAX_ITERS" \
    --batch_size 2 \
    --labeled_bs "$LABELED_BS" \
    --labelnum "$LABELNUM" \
    --accumulation_steps 4 \
    --optimizer adamw \
    --base_lr 1e-4 \
    --chunk_size 8 \
    --feature_scaler 2 \
    --temp 0.6 \
    --u_weight 0.5 \
    --consistency 0.1 \
    --use_teacher_loss 1 \
    --use_focal 1 \
    --use_pretrain "$USE_PRETRAIN" \
    --pretrained_ckpt "$VMAMBA_CKPT" \
    ${LOAD_PATH:+--load_path "$LOAD_PATH"}
