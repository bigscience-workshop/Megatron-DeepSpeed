#!/bin/bash
export OMP_NUM_THREADS=16

DATA_PATH=/mnt/Megatron-DeepSpeed/data/zhihu_100000_text_document
RUSH_PATH=/mnt/Megatron-DeepSpeed/176B_ft_qb_60w_en_80w
mkdir -p $RUSH_PATH
LOAD_PATH=/mnt/Megatron-DeepSpeed/176B_ft_qb_60w_en_80w/checkpoint
CHECKPOINT_PATH=$RUSH_PATH/checkpoint
TENSORBOARD_PATH=$RUSH_PATH/tensorboard
LOGS_PATH=$RUSH_PATH/logs
TOKENIZER_NAME_OR_PATH=/mnt/Megatron-DeepSpeed/bloom_tokenizer

MASTER_ADDR=$1
MASTER_PORT=$2
NNODES=$3
NODE_RANK=$4

GPUS_PER_NODE=8

TP_SIZE=8
PP_SIZE=8

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024

NLAYERS=70
NHIDDEN=14336
NHEADS=112
SEQ_LEN=2048

SAVE_INTERVAL=50000000

TRAIN_SAMPLES=220_000_000  # 450B tokens
LR_DECAY_SAMPLES=200_000_000  # Decay for the first 410B tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=183_105  # 375M tokens


OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 6e-6 \
    --min-lr 6e-7 \
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
    --bf16 \
    --seed 42 \
    --position-embedding-type alibi \
    --checkpoint-activations \

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
    --eval-interval 10000 \
    --eval-iters 1 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=0 # important: bf16 must use z0! it implements its own zero stage 1 equivalent
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
  "bf16": {
    "enabled": true
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
    --load $LOAD_PATH \
    --data-path $DATA_PATH \
    --split 998,1,1 \
    --data-impl mmap \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "

export NODE_RANK
mkdir -p $(dirname $0)/logs

echo "CMD:$CMD"

echo "master_addr:$MASTER_ADDR master_port:$MASTER_PORT nnodes:$NNODES node_rank:$NODE_RANK GPUS_PER_NODE:$GPUS_PER_NODE"
echo "logfile:$(dirname $0)/logs/${HOSTNAME}.log"

bash -c '$LAUNCHER --node_rank ${NODE_RANK} $CMD' 2>&1 | tee $(dirname $0)/logs/$NNODES-${GPUS_PER_NODE}-${HOSTNAME}.log

