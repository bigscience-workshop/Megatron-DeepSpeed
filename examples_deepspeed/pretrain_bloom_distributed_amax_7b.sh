#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
set -ex

module load anaconda/2021.11
module load cuda/11.7
module load gcc/9.4
source activate py38-bigscience

ulimit -n 102400

export PYTHONUNBUFFERED=1
# export OMP_NUM_THREADS=16
export TRANSFORMERS_VERBOSITY="debug"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond2
export NCCL_IB_HCA=mlx5_2,mlx5_5
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export OMP_NUM_THREADS=1

# ${NODES} ${GPUS} ${node_rank} ${MASTER_ADDR} ${HOSTFILE} ${JOB_ID}
# nodes gpus rank master_addr hosfile job_id
# nodes
NODES=$1

# gpus
NUM_GPUS=$2

# rank
NODE_RANK=$3

# Master addr
MASTER_ADDR=$4
MASTER_PORT=29501

#DHOSTFILE
DHOSTFILE=$5

# JOB_ID
JOB_ID=$6

# logs
OUTPUT_LOG="${JOB_ID}/train_rank${NODE_RANK}.log"
echo "nodes,gpus,mp_size,node_rank,master_addr,master_port,dhostfile" >> $OUTPUT_LOG
echo "$NODES,$NUM_GPUS,$MP_SIZE,$NODE_RANK,$MASTER_ADDR,$MASTER_PORT,$DHOSTFILE" >> $OUTPUT_LOG


DATASET="1 /data/yechen/data/pretrain-v1/zh-wudao-web-50_content_document
1 /data/yechen/data/pretrain-v1/zh-wudao-web-56_content_document
1 /data/yechen/data/pretrain-v1/zh-zhihu-0_content_document
1 /data/yechen/data/pretrain-v1/zh-zhihu-2_content_document"

######################################
# Change the below configurations here
DS_CONFIG=/data/yechen/Megatron-DeepSpeed/configs/ds_config_zero3_7b_tmp.json

TOKENIZER_PATH=/data/yechen/models/bloom-7b1
#CHECKPOINT_PATH=/share/home/hbuser1/yechen/Llama-2-7b
LAYER_WEIGHT_PATH=/data/cong.fu/models/bloom-7b1-megatron-tp-2
OUTPUT_PATH=/data/cong.fu/models/bloom-7b1-megatron-tp-2-dev

TP=2
PP=2
ZERO_STAGE=0


HIDDEN_SIZE=4096 #NHIDDEN=14336 for 70b
NUM_LAYERS=30 #NUM_LAYERS=70 for 70b
NUM_HEADS=32 #NUM_HEADS=112 for 70b
SEQ_LENGTH=2048  #SEQ_LEN=2048 for 70b

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=336 # e.g. llama: 4M tokens
TRAIN_STEPS=10000

LR=2e-4
MIN_LR=2e-5
LR_WARMUP_STEPS=300
#LR_WARMUP_SAMPLES=336000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################



cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "gradient_clipping": $GRAD_CLIP,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"


DISTRIBUTED_ARGS="--nproc_per_node $NUM_GPUS --nnodes $NODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#--from-weights $LAYER_WEIGHT_PATH \
torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --pp-partition-method 'type:transformer|embedding' \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --position-embedding-type alibi \
       --embed-layernorm \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --train-iters $TRAIN_STEPS \
       --save $OUTPUT_PATH \
       --from-weights $LAYER_WEIGHT_PATH \
       --data-path $DATASET \
       --data-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 100 \
       --eval-interval 10000 \
       --eval-iters 10 \
       --bf16 \
       --sync-tp-duplicated-parameters \
       --make-vocab-size-divisible-by 128 \
       --tensorboard-dir $JOB_ID/tb-bloom-7b-tp-$TP-pp-$PP-nodes-$NODES-gbs-$GLOBAL_BATCH_SIZE \
       --tensorboard-queue-size 10 \
       --log-timers-to-tensorboard \
       --log-batch-size-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --no-log-loss-scale-to-tensorboard \
       $ds_args >> $OUTPUT_LOG 2>&1
       
#--from-weights $LAYER_WEIGHT_PATH \
#--load $OUTPUT_PATH \
