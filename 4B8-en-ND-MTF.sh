#!/bin/bash

EXPERIMENT_NAME=4B8-en-ND-MTF
REPO_PATH=experiments/$EXPERIMENT_NAME
CHECKPOINT_PATH=$REPO_PATH/checkpoints
TENSORBOARD_PATH=$REPO_PATH/tensorboard
CODECARBON_PATH=$REPO_PATH/codecarbon
LOGS_PATH=$REPO_PATH/logs

DATA_PATH=data/mc4-id_text_document

# XXX: edit me
GPUS_PER_NODE=8
NNODES=1
PP_SIZE=2 # NLAYERS must be a multiple of PP_SIZE here
TP_SIZE=1 # always fixed to the size of a single node
DP_SIZE=$((NNODES*GPUS_PER_NODE/(PP_SIZE*TP_SIZE))) # will get derived automatically by trainer

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_ITER=10_000
INPUT_LEN=1024
TARGET_LEN=256

NLAYERS=24
NHIDDEN=4096
NHEADS=64
FFN_HIDDEN_SIZE=10240
MAX_POSITION_EMBEDDING=1280

SAVE_INTERVAL=1500

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr 2e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

EXIT_OPTS=" \
    --exit-duration-in-mins 1190 \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --max-position-embeddings $MAX_POSITION_EMBEDDING \
    --encoder-seq-length $INPUT_LEN \
    --decoder-seq-length $TARGET_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITER \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path t5-base \
    --loss-scale 12 \
    --clip-grad 1.0 \
    --fp16 \
    --checkpoint-activations \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

OUTPUT_ARGS=" \
    --log-interval 200 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $TRAIN_ITER \
    --eval-iters 1 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=1

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

# export LAUNCHER="python -u -m torch.distributed.launch \
#     --nproc_per_node $GPUS_PER_NODE \
#     "
#     # --nnodes $NNODES \
#     # --master_addr $MASTER_ADDR \
#     # --master_port $MASTER_PORT \

export CMD=" \
    `pwd`/train_ND_MTF_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "


# # clear old checkpoint as it'd mismatch while we sort things out
#     rm -rf $SAVE_CHECKPOINT_PATH


echo $CMD

# We create the folder where the logs and codecarbon will be stored.
mkdir -p $REPO_PATH
mkdir -p $LOGS_PATH
# to debug - add echo (it exits and prints what it would have launched)

python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    $CMD

# srun '$LAUNCHER --node_rank $SLURM_PROCID $CMD' 2>&1 | tee -a $LOGS_PATH/main_log.txt