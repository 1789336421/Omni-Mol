#!/bin/bash
##############################################
#                  WARNING                   #
# This is a preview version, we will release #
# optimized and verified version soon!       #
##############################################

PROMPT_VERSION="llama3"
MODEL_VERSION="llama"

GRAPH_TOWER="moleculestm"
INIT_CHECKPOINT_GNN="<Path to MoleculeSTM>"

CHECKPOINT_FOLDER_PREFIX="_checkpoints"
MODEL_PATH="<Path to Llama3.2-1B>"
PROJECTOR="naive_linear"
REMARK="omnimol-train"

export WANDB_ENTITY="Omni-Mol"
export WANDB_PROJECT="XXXXX"
export WANDB_API_KEY="xxxx"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========== Start Training =========="

# DeepSpeed doesn't work on some specific machines (AWS especially)
# and the training may become unstable, if it happens, then remove
# --deepspeed scripts/zero_configs/zero2.json \
deepspeed --master_port 29505 main.py \
    --deepspeed scripts/zero_configs/zero2.json \
    --use_alpha True \
    --model_name_or_path $MODEL_PATH \
    --base_model $MODEL_PATH \
    --language_backbone_name $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path "<Path to Omni-Mol train folder>" \
    --graph_tower $GRAPH_TOWER \
    --mm_projector_type $PROJECTOR \
    --graph_init_checkpoint $INIT_CHECKPOINT_GNN \
    --pretrain_mm_mlp_adapter "<Path to mm_projector.bin>" \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/$MODEL_VERSION-$REMARK \
    --num_train_epochs 15 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --stop_epoch 15 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 8e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.0075 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --logging_dir tf-logs/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --moe_class deepseek \
    --moe_mode second_quarter\
    --ep_size 1 \
    --num_experts 5 \
    --use_residual True \
    --router_aux_loss_coef 0.01 \
    --is_training True \
    --top_k_experts 2 \
    --ddp_find_unused_parameters False \
