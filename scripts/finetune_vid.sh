#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-3-13b"
################## VICUNA ##################

# ################# LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
# ################# LLaMA-2 ##################

deepspeed lita/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /path/to/datasets \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/lita-$MODEL_VERSION-finetune \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --tasks "dvc||event_loc||imgqa||vidqa||temporal_reasoning" \
    --dvc_data "activitynet||youcook2" \
    --event_loc_data "activitynet||youcook2" \
    --imgqa_data "llava" \
    --vidqa_data "nextqa" \
    --temporal_reasoning_data "activitynet" \
    --task_sample_rate 1 1 1 1 1\
    --dvc_sample_rate 5 1 \
    --event_loc_sample_rate 5 1 \
    --imgqa_sample_rate 1 \
    --vidqa_sample_rate 1 \
    --temporal_reasoning_sample_rate 1 \
    --samples_per_epoch 50000 \
    --video_arch temporal_spatial_pool

