#!/bin/bash
#SBATCH --job-name=run_evalharness-tr13f-6b3
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --reservation=hug

set -x -e

source $six_ALL_CCFRWORK/start-muennighofflmeval

echo "START TIME: $(date)"

# a unique identifier for the current eval ideally correspnding to the modelname
VARIANT="tr13f-prefix"


CHECKPOINT_PATH=/gpfsscratch/rech/six/commun/checkpoints/tr13f-6B3-ml-t0/checkpoints/prefix/global_step3100
MEGATRON_DEEPSPEED_REPO=$six_ALL_CCFRSCRATCH/commun/experiments/muennighoff/megdsbslmeval/Megatron-DeepSpeed
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasetseval
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export TOKENIZERS_PARALLELISM=false

cd $MEGATRON_DEEPSPEED_REPO

TOKENIZER_NAME_OR_PATH=bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles

PP_SIZE=1
TP_SIZE=1
SEQ_LEN=2048

# different from the training MICRO_BATCH_SIZE - no optim memory, so can do bigger BS
# make as big as it can fit into gpu w/o OOM, but not too close to 100%
EVAL_MICRO_BATCH_SIZE=1

#dummy arguments to make megatron happy.
MEGATRON_REQUIRED_ARGS=" \
    --num-layers -1 \
    --hidden-size -1 \
    --num-attention-heads -1 \
    --seq-length -1  \
    --max-position-embeddings -1 \
"


ZERO_STAGE=0

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": 1,
  "train_batch_size": 1,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": false
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT


CMD="./tasks/eval_harness/evaluate_bsevalharness_prefix.py  \
    --load $CHECKPOINT_PATH \
    --results_path $VARIANT-results.json \
    --tensor-model-parallel-size $TP_SIZE  \
    --pipeline-model-parallel-size $PP_SIZE \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --micro-batch-size $EVAL_MICRO_BATCH_SIZE \
    --no-load-optim \
    --no-load-rng \
    --eval_fp32 \
    --inference \
    --seq-length $SEQ_LEN \
    --task_list copa \
    --prefix \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --intermed_results \
    --adaptive_seq_len \
    --micro_bs_multiplier 8 \
    $MEGATRON_REQUIRED_ARGS \
    "

GPUS_PER_NODE=1
NNODES=$SLURM_NNODES
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CUDA_LAUNCH_BLOCKING=1

echo $LAUNCHER $CMD

export PYTHONPATH=$MEGATRON_DEEPSPEED_REPO

$LAUNCHER $CMD 2>&1 | tee $VARIANT-eval-harness.log
