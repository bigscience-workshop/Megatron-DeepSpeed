set -ex
# bash args_deepspeed_gpt.sh 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 true true 80 1280

# volcengine.com
export NCCL_IB_PCI_RELAXED_ORDERING=1

NNODES=${1:-1}
GPUS_PER_NODE=${2:-8}
# Change for multinode config
NODE_RANK=${3:-0}
MASTER_ADDR=${4:-"127.0.0.1"}
MASTER_PORT=6000
MP=${5:-1}
PP=${6:-1}
USE_FP16=${7:-true}
ACTIVATION_CHECKPOINT=${8:-false}
MICRO_BATCH_SIZE=${9:-4}
GLOBAL_BATCH_SIZE=${10:-4}
NUM_LAYER=${11:-24}
RUN_COMMIT=${12:-"01b1d32"}
TRAIN_ITERS=${13:-220}
LOG_PERIOD=${14:-100}
CHECKPOINT_PATH=${15:-"checkpoints/gpt2"}
VOCAB_FILE=${16:-"gpt2-vocab.json"}
MERGE_FILE=${17:-"gpt2-merges.txt"}
DATA_PATH=${18:-"my-gpt2_text_document"}


SRC_DIR=$(realpath $(dirname $0)/)
TRAN_MODEL="Megatron-Deepspeed_gpt2"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=${SRC_DIR}/test_logs/$RUN_COMMIT/${NNODES}n${GPUS_PER_NODE}g
if [[ ! -z "$LOG_FOLDER" ]]; then
    mkdir -p $LOG_FOLDER
fi

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi

LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_nl${NUM_LAYER}_nah16_hs1024_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_mp${MP}_pp${PP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${NNODES}n${GPUS_PER_NODE}g_${RUN_TIME}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

GPT_ARGS=" \
    --num-layers $NUM_LAYER \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 0.00015 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --lr-warmup-fraction .01 \
    "

OUTPUT_ARGS=" \
    --log-interval $LOG_PERIOD \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    "
# --checkpoint-activations \

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    "
# --load $CHECKPOINT_PATH \


ZERO_STAGE=1
config_json="./ds_config.json"

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

export LOGLEVEL=WARNING
# LAUNCHER="deepspeed -num_gpus $GPUS_PER_NODE"

CMD="python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ./pretrain_gpt.py \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    $DATA_ARGS \
    $DEEPSPEED_ARGS \
    --tensor-model-parallel-size $MP \
    --pipeline-model-parallel-size $PP \
    --DDP-impl local "

if $USE_FP16; then
    CMD+=" \
      --fp16 "
fi

if $ACTIVATION_CHECKPOINT; then
    CMD+=" \
      --checkpoint-activations "
fi

echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}.log
