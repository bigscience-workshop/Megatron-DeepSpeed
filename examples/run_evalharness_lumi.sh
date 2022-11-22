#!/bin/bash
#SBATCH --exclude=nid005159
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -p eap
#SBATCH -t 2-0:00:00
#SBATCH --gpus-per-node=mi250:1
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000119
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0"
    exit
fi

set -euo pipefail

# symlink logs/latest_eval.out and logs/latest_eval.err
ln -f -s $SLURM_JOB_ID.out logs/latest_eval.out
ln -f -s $SLURM_JOB_ID.err logs/latest_eval.err

# Data
CHECKPOINT_PATH=/scratch/project_462000119/muennighoff/nov-2022-optimization/checkpoints/global_step10
VARIANT=global_step10

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/scratch/project_462000119/ds_cache

VOCAB_FILE="gpt2/vocab.json"
MERGE_FILE="gpt2/merges.txt"

PP_SIZE=1
TP_SIZE=1
# different from the training MICRO_BATCH_SIZE - no optim memory, so can do bigger BS
# make as big as it can fit into gpu w/o OOM, but not too close to 100%
EVAL_MICRO_BATCH_SIZE=1
MICRO_BS_MULTIPLIER=1

# Model parameters
SEQ_LEN=2048

# Dummy arguments
MEGATRON_REQUIRED_ARGS=" \
    --num-layers -1 \
    --hidden-size -1 \
    --num-attention-heads -1 \
    --seq-length -1  \
    --max-position-embeddings -1 \
"

ZERO_STAGE=0

mkdir -p ds_configs
DS_CONFIG_PATH="ds_configs/$SLURM_JOB_ID.json"

cat <<EOF > $DS_CONFIG_PATH
{
    "train_micro_batch_size_per_gpu": 1,
    "train_batch_size": 1,
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
EOF

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config $DS_CONFIG_PATH \
    --zero-stage $ZERO_STAGE \
    "

CMD="Megatron-DeepSpeed/tasks/eval_harness/evaluate.py \
    --load $CHECKPOINT_PATH \
    --results_path $VARIANT-results.json \
    --tensor-model-parallel-size $TP_SIZE  \
    --pipeline-model-parallel-size $PP_SIZE \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --micro-batch-size $EVAL_MICRO_BATCH_SIZE \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --inference \
    --seq-length $SEQ_LEN \
    --task_list piqa \
    --intermed_results \
    --adaptive_seq_len \
    --micro_bs_multiplier $MICRO_BS_MULTIPLIER \
    $MEGATRON_REQUIRED_ARGS \
    $DEEPSPEED_ARGS \
    "

echo $CMD

echo "START $SLURM_JOBID: $(date)"

srun --label launch.sh $CMD

echo "END $SLURM_JOBID: $(date)"

