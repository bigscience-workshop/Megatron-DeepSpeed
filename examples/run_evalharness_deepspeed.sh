CHECKPOINT_PATH=/gpfsdsstore/projects/rech/six/commun/checkpoints/tr3m-1B3-emb-norm-pile/global_step296023
MEGATRON_DEEPSPEED_REPO=/gpfsssd/worksf/projects/rech/six/commun/code/eval/Megatron-DeepSpeed
export HF_DATASETS_OFFLINE=1

PP_SIZE=2
TP_SIZE=1
VOCAB_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-vocab.json
MERGE_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-merges.txt
SEQ_LEN=2048

# different from the training MICRO_BATCH_SIZE - no optim memory, so can do bigger BS
# make as big as it can fit into gpu w/o OOM, but not too close to 100%
EVAL_MICRO_BATCH_SIZE=12

#dummy arguments to make megatron happy.
MEGATRON_REQUIRED_ARGS=" \
    --num-layers -1 \
    --hidden-size -1 \
    --num-attention-heads -1 \
    --seq-length -1  \
    --max-position-embeddings -1
"


ZERO_STAGE=0

config_json="./ds_config.json"

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=512

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

#    --adaptive_seq_len \
CMD="./tasks/eval_harness/evaluate.py  \
    --load $CHECKPOINT_PATH \
    --tensor-model-parallel-size $TP_SIZE  \
    --pipeline-model-parallel-size $PP_SIZE \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --micro-batch-size $EVAL_MICRO_BATCH_SIZE \
    --no-load-optim \
    --no-load-rng \
    --inference \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --seq-length $SEQ_LEN \
    --eval_fp32 \
    --task_list hellaswag,mrpc,piqa \
    $MEGATRON_REQUIRED_ARGS \
    "

N_GPUS=1
LAUNCHER="deepspeed --num_gpus $N_GPUS"
$LAUNCHER $CMD

# want datasets
