#!/usr/bin/env bash

export CUDA_LAUNCH_BLOCKING=1

model_name="GPT4MTS"

python3 main.py \
    --model "$model_name" \
    --model_id "$model_name" \
    --checkpoints "./checkpoints/" \
    --res_dir "results_tts" \
    --root_path "dataset/" \
    --data_dir "demo_data" \
    --train_epochs 10 \
    --batch_size 16 \
    --channel_independent True \
    --d_model 768 \
    --summary True \
    --revin True \
    --itr 1
