#!/bin/bash

# Train motion vector refiner using Mamba architecture
# This script trains a standalone MV refiner that can later be used during main model training

# Set up environment
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

# Training configuration
OUTPUT_DIR="mv_refiner_logs/mamba_$(date +%Y%m%d_%H%M%S)"
EPOCHS=10
BATCH_SIZE=32
LR=1e-3
HIDDEN_DIM=128
SEED=42

# Mamba-specific hyperparameters
MAMBA_D_STATE=16
MAMBA_D_CONV=4
MAMBA_EXPAND=2

# Dataset configuration — pickle files of (codec_mv, optical_flow) pairs
# extracted from your training videos. Each entry should contain the codec
# motion vectors and the corresponding ground-truth optical flow at the
# same sampling rate (16 FPS for the canonical ReMoRa setup).
PKL_PATHS=(
    "./data/motion_vectors_part1.pkl"
    # add more parts as needed
)

# Optional: Enable WandB logging
WANDB_ENABLED="--wandb"
WANDB_PROJECT="remora_mv_refiner"
WANDB_ENTITY=""  # Set your wandb entity/team if needed

# Run training
python llava/train/train_mv_refiner.py \
    --pkl-paths "${PKL_PATHS[@]}" \
    --model-type mamba \
    --hidden-dim $HIDDEN_DIM \
    --mamba-d-state $MAMBA_D_STATE \
    --mamba-d-conv $MAMBA_D_CONV \
    --mamba-expand $MAMBA_EXPAND \
    --output-dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --weight-decay 1e-4 \
    --num-workers 8 \
    --device cuda \
    --seed $SEED \
    --val-split 0.1 \
    --predict-residual \
    $WANDB_ENABLED \
    --wandb-project $WANDB_PROJECT \
    ${WANDB_ENTITY:+--wandb-entity $WANDB_ENTITY}

echo "Training complete. Best model saved to: $OUTPUT_DIR/best_model.pt"
echo "To use this checkpoint, update MV_REFINER_CHECKPOINT in your main training script"
