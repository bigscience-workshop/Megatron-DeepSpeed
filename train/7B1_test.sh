#!/bin/bash
#set -x

export OMP_NUM_THREADS=16
DATA_PATH=/mnt/Megatron-DeepSpeed/data/zhihu_100000_text_document
RUSH_PATH=/mnt/Megatron-DeepSpeed/7B1_test
mkdir -p $RUSH_PATH
CHECKPOINT_PATH=$RUSH_PATH/checkpoint
TENSORBOARD_PATH=$RUSH_PATH/tensorboard
LOGS_PATH=$RUSH_PATH/logs
TOKENIZER_NAME_OR_PATH=$(pwd)/bloom_tokenizer

MASTER_ADDR=$1
MASTER_PORT=$2
NNODES=$3
NODE_RANK=$4

GPUS_PER_NODE=8

TP_SIZE=1
PP_SIZE=1

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024

NLAYERS=30
NHIDDEN=4096
NHEADS=32
SEQ_LEN=2048

SAVE_INTERVAL=200000

TRAIN_SAMPLES=220_000
LR_DECAY_SAMPLES=200_000
LR_WARMUP_SAMPLES=183_105


OPTIMIZER_ARGS=" \
    --lr 2e-5 \
    --min-lr 2e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "
# for 20h 1190, for 100h 5990
#    --exit-duration-in-mins 1190 \
#--pad-vocab-size-to 250880 \
#--rampup-batch-size 128 64 9_765_625 \
EXIT_OPTS=" \
    --exit-duration-in-mins 599000 \
    "

GPT_ARGS=" \
    --finetune \
    --override-lr-scheduler \
    --no-load-optim \
    --no-load-rng \
    --reset-position-ids \
    --reset-attention-mask \

    --pp-partition-method type:transformer|embedding \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \

    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --init-method-std 0.0048 \
    --embed-layernorm \
    --fp16 \
    --seed 42 \
    --position-embedding-type alibi \
    --checkpoint-activations \
    --abort-on-unmet-fused-kernel-constraints \
    --pad-vocab-size-to 250880 \

    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "
#--glu-activation swiglu \
#--position-embedding-type alibi \
# TODO: decide on efficient eval-interval + eval-iters

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 40000 \
    --eval-iters 1 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=1 # important: bf16 must use z0! it implements its own zero stage 1 equivalent
mkdir -p ds_config
config_json="./ds_config/ds_config.$SLURM_JOB_ID.json"

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
  "wall_clock_breakdown": false,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.00002,
      "betas": [
        0.9,
        0.95
      ],
      "eps": 1e-8,
      "weight_decay": 1e-1
    }
  }
}
EOT


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

export CMD=" \
    `pwd`/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --split 998,1,1 \
    --data-impl mmap \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "
export NODE_RANK
mkdir -p $(dirname $0)/logs
logfile=$(dirname $0)/logs/$NNODES-${GPUS_PER_NODE}-${HOSTNAME}.log

echo "SCRIPT_CMD:$CMD"
echo "MASTER_ADDR:$MASTER_ADDR MASTER_PORT:$MASTER_PORT NNODES:$NNODES NODE_RANK:$NODE_RANK"
echo "LOGFILE:$logfile"

logfile=$(dirname $0)/logs/$NNODES-${GPUS_PER_NODE}-${HOSTNAME}.log
bash -c '$LAUNCHER --node_rank ${NODE_RANK} $CMD' > >(tee $logfile) 2>&1


