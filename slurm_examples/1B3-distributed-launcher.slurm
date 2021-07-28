#!/bin/bash
#SBATCH --job-name=1B3-test.slurm
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=24         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 00:20:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.out            # error file name (same to watch just one file)
#SBATCH --account=six@gpu

set -x -e


ROUND=1
TESTING=1

SAVE_CHECKPOINT_PATH=$SCRATCH/checkpoints/1B3-test/
MEGATRON_DEEPSPEED_REPO=$SCRATCH/Megatron-DeepSpeed

if [[ ${TESTING} == 1 ]]; then
    # testing on 10k
    DATA_PATH=$six_ALL_CCFRSCRATCH/datasets-custom/c4_preprocessing/c4_100k_text_document
else
    # production on full 304M records
    DATA_PATH=$six_ALL_CCFRSCRATCH/datasets-custom/c4_preprocessing/c4_en_train_text_document

fi

source $six_ALL_CCFRWORK/start-prod
cd $MEGATRON_DEEPSPEED_REPO
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000

# adjust depending on the number of the nodes

# XXX: edit me
GPUS_PER_NODE=4
NNODES=16
PP_SIZE=8 # NLAYERS must be a multiple of PP_SIZE here
TP_SIZE=4 # always fixed to the size of a single node
DP_SIZE=$((NNODES*GPUS_PER_NODE/(PP_SIZE*TP_SIZE))) # will get derived automatically by trainer
GAS=256

MICRO_BATCH_SIZE=2

EXIT_EARLY=0
#EXIT_EARLY=1

NLAYERS=24
NHIDDEN=2048
NHEADS=16
FFN_HIDDEN_SIZE=8192
SEQ_LEN=2048

if   [[ ${ROUND} == 1 ]]; then TRAIN_ITER=500    SAVE_INTERVAL=10
elif [[ ${ROUND} == 2 ]]; then TRAIN_ITER=1000   SAVE_INTERVAL=18
elif [[ ${ROUND} == 3 ]]; then TRAIN_ITER=146_484_375 SAVE_INTERVAL=1500
else echo "invalid ROUND: $ROUND"
fi

# --exit-duration-in-mins
# --exit-interval - exit after this many iterations
if [[ ${EXIT_EARLY} == 1 ]]; then EXIT_OPTS="--exit-interval 2"
else EXIT_OPTS=""
fi

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples 126_953_125 \
    --lr-warmup-samples 183_105 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

GLOBAL_BATCH_SIZE=2048
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_ITER \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path t5-small \
    --loss-scale 12 \
    --clip-grad 1.0 \
    --fp16 \
    --checkpoint-activations \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 100 \
    --eval-iters 10 \
    "

ZERO_STAGE=1

config_json="./ds_config.$SLURM_JOBID.json"

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "gradient_accumulation_steps": $GAS,
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
    --save $SAVE_CHECKPOINT_PATH \
    --tensorboard-dir $SAVE_CHECKPOINT_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "


# # clear old checkpoint as it'd mismatch while we sort things out
#     rm -rf $SAVE_CHECKPOINT_PATH


echo $CMD

# to debug - add echo (it exits and prints what it would have launched)
clear; srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD' 2>&1 | tee 1B3-distributed-launcher.$SLURM_JOBID.out
