#!/bin/bash


DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
#mkdir -p $DIR/logs
#mkdir -p /tmp/logs


#DATASET_1="<PATH TO THE FIRST DATASET>"
#DATASET_2="<PATH TO THE SECOND DATASET>"
#DATASET_3="<PATH TO THE THIRD DATASET>"
#DATASET="0.2 ${DATASET_1} 0.3 ${DATASET_2} 0.5 ${DATASET_3}"

BASE_DATA_PATH=/data/Megatron-LM/data
DATASET=${BASE_DATA_PATH}/indexed_datasets/megatron
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt


script_path=$(realpath $0)
script_dir=$(dirname $script_path)
#CONFIG_JSON="$script_dir/ds_config.json"
CONFIG_JSON="/tmp/ds_config.json"

USE_DEEPSPEED=1
ZERO_STAGE=0


# Debug
#TP=4
#PP=4
#LAYERS=8
#HIDDEN=512
#SEQ=1024
#GLOBAL_BATCH=128
#WORKER_STR="-i worker-0"

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --no-deepspeed)
    USE_DEEPSPEED=0;
    shift
    ;;
    -z|--zero-stage)
    ZERO_STAGE=$2;
    shift    
    shift
    ;;
    *)
    echo "Unknown argument(s): $key"
    exit 1
    shift
    ;;
esac
done

TP=4
PP=1
DP=2
WORLD_SIZE=$((TP*PP*DP))

HIDDEN=1024
LAYERS=24
NHEADS=32
SEQ=1024

#LAYERS=2
#HIDDEN=8
#NHEADS=2
#SEQ=8

GLOBAL_BATCH=64
WORKER_STR=""
EXIT_ITERS=10
TRAIN_SAMPLES=1000000

MICRO_BATCH=32
LR=1.0e-1
MIN_LR=1.0e-1
DTYPE="bf16"
RUN_VERSION=1
EXP_DIR=${HOME}/experiments/results/bf16
RUN_TAG="tp${TP}_pp${PP}_dp${DP}_hd${HIDDEN}_nl${LAYERS}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}_LR_${LR}_${MIN_LR}_${DTYPE}_train_${EXIT_ITERS}_v${RUN_VERSION}"
LOG_DIR="${EXP_DIR}/tensorboard/${RUN_TAG}"
mkdir -p $LOG_DIR
export BIT16_DUMP_FILE="${EXP_DIR}/${RUN_TAG}.txt"
CHECKPOINT_DIR="./checkpoints/${DTYPE}_z${ZERO_STAGE}_tp${TP}_pp${PP}_dp${DP}_nl${LAYERS}_exit_${EXIT_ITERS}_v${RUN_VERSION}"
options=" \
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size $PP \
        --num-layers $LAYERS \
        --hidden-size $HIDDEN \
        --num-attention-heads ${NHEADS} \
        --seq-length $SEQ \
        --max-position-embeddings $SEQ \
	--micro-batch-size $MICRO_BATCH \
	--global-batch-size $GLOBAL_BATCH \
        --optimizer adam \
        --adam-eps 1e-8 \
        --lr-warmup-samples 5 \
        --min-lr 1e-6 \
        --lr-decay-style cosine \
        --lr-decay-samples 12 \
        --override-lr-scheduler \
        --clip-grad 1.0 \
        --weight-decay 1e-1 \
        --embed-layernorm \
        --partition-activations \
        --lr $LR \
	--min-lr $MIN_LR \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 40 \
        --eval-interval 10 \
	--data-path ${DATASET} \
	--vocab-file ${VOCAB_PATH} \
	--merge-file ${MERGE_PATH} \
	--save-interval 10000 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
        --${DTYPE} \
	--checkpoint-activations \
        --train-samples ${TRAIN_SAMPLES} \
	--exit-interval ${EXIT_ITERS} \
        --seed 42 \
        --load ${CHECKPOINT_DIR} \
        --save ${CHECKPOINT_DIR} \
	--tensorboard-dir $LOG_DIR
        "
#        --split 10,0,0 \
#         --rampup-batch-size 2 2 1_000 \


if [[ ${USE_DEEPSPEED} -eq 1 ]]; then
	echo "Using DeepSpeed"
	options="${options} \
		--deepspeed \
		--deepspeed_config=${CONFIG_JSON} \
		--zero-stage=${ZERO_STAGE} \
		--deepspeed-activation-checkpointing \
	"
fi


cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "bf16": {
    "enabled": true
  },

  "fp16": {
    "enabled": false,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

WORKER_STR="--num_nodes 1 --num_gpus $WORLD_SIZE"
#WORKER_STR="-i worker-0:0,1,2,3"
#run_cmd="deepspeed -i worker-0:0,1,2,3 ${DIR}/pretrain_gpt.py $@ ${options}"
#run_cmd="deepspeed -i worker-0 ${DIR}/pretrain_gpt.py $@ ${options}"

run_cmd="deepspeed --master_port 29600 $WORKER_STR ${DIR}/pretrain_gpt.py $@ ${options}"


echo ${run_cmd}
eval ${run_cmd}

set +x
