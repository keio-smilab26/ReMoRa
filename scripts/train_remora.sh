#!/bin/bash

# Set up the data folder (point these at your local LLaVA-Video-178K mirror).
IMAGE_FOLDER="./DATAS/LLaVA-Video-178K"
VIDEO_FOLDER="./DATAS/LLaVA-Video-178K"
DATA_YAML="scripts/exp.yaml"

############### Prepare Envs #################
# python3 -m pip install flash-attn --no-build-isolation
# alias python=python3

# CPU Optimization for 192-core system
export OMP_NUM_THREADS=24  # 192 cores / 8 GPUs
export MKL_NUM_THREADS=24
export TORCH_NUM_THREADS=24
export CV_FFMPEG_THREAD_COUNT=8
export OPENCV_FFMPEG_THREAD_COUNT=8

# Reset CUDA_VISIBLE_DEVICES to use numeric indices instead of UUIDs
# This is needed when running on systems that use GPU UUIDs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

############### Show Envs ####################

# nvidia-smi

################ Arnold Jobs ################

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
PROMPT_VERSION="qwen_1_5"
PREV_STAGE_CHECKPOINT="lmms-lab/LLaVA-Video-7B-Qwen2"
RUN_NAME="ReMoRa_Qwen2_7B"
OUTPUT_DIR="work_dirs/${RUN_NAME}"

# Motion Vector Refiner Settings - MAMBA Configuration
MV_REFINER_CHECKPOINT="mv_refiner_logs/best_model.pt"  # set to output of scripts/train_mv_refiner.sh
USE_MV_REFINER="True"
MV_REFINER_TYPE="mamba"
MV_REFINER_HIDDEN_DIM=128
MV_REFINER_D_STATE=16
MV_REFINER_D_CONV=4
MV_REFINER_EXPAND=2
MV_REFINER_PREDICT_RESIDUAL="True"
MV_AUX_LOSS_WEIGHT=0.0  # Weight for auxiliary loss (0.0 to disable)

export WANDB_PROJECT=ReMoRa

deepspeed --master_port 30000 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 32 \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model,compressor,mv_refiner" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --compressor_type bimba_two_stage \
    --gop_fps 16 \
    --gop_n_layers 1 \
    --gop_d_state 16 \
    --gop_d_conv 4 \
    --gop_expand 2 \
    --mv_mode raw \
    --global_n_layers 1 \
    --global_d_state 16 \
    --global_d_conv 4 \
    --global_expand 2 \
    --use_mv_refiner ${USE_MV_REFINER} \
    --mv_refiner_checkpoint ${MV_REFINER_CHECKPOINT} \
    --mv_refiner_type ${MV_REFINER_TYPE} \
    --mv_refiner_hidden_dim ${MV_REFINER_HIDDEN_DIM} \
    --mv_refiner_d_state ${MV_REFINER_D_STATE} \
    --mv_refiner_d_conv ${MV_REFINER_D_CONV} \
    --mv_refiner_expand ${MV_REFINER_EXPAND} \
    --mv_refiner_predict_residual ${MV_REFINER_PREDICT_RESIDUAL} \
    --mv_aux_loss_weight ${MV_AUX_LOSS_WEIGHT} \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --torch_compile False \
    --dataloader_drop_last True \
    --frames_upbound 256 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --use_gop_loading True \
    --mm_spatial_pool_stride 2 \
    --report_to wandb
exit 0;
