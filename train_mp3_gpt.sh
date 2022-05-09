CHECKPOINT_PATH=data/checkpoints/gpt2
VOCAB_FILE=data/gpt2-vocab.json
MERGE_FILE=data/gpt2-merges.txt
DATA_PATH=data/t0-test
TENSORBOARD_PATH=data/checkpoints/tensorboard
CODECARBON_PATH=data/checkpoints/codecarbon

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
TP_SIZE=1
PP_SIZE=1

N_GPUS=2
SAVE_INTERVAL=100

#    --train-samples 10_000 \
#    --exit-interval $EXIT_INTERVAL \

#    --exit-interval 100 \
GPT_ARGS=" \
    --num-layers 2 \
    --hidden-size 64 \
    --num-attention-heads 2 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 2 2 1_000 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples 100 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --lr-warmup-samples 5 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --fp16 \
    "
#    --train-iters 500 \

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-activations \
    "

#    --codecarbon-dir $CODECARBON_PATH \
DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
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

ALL_ARGS="$GPT_ARGS $OUTPUT_ARGS $DATA_ARGS $DEEPSPEED_ARGS"

# if you can't stand pt-1.9 launcher noise
export LOGLEVEL=WARNING

LAUNCHER="deepspeed --num_gpus $N_GPUS"
export CMD=" \
    $LAUNCHER pretrain_mp3_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --distributed-backend nccl \
    $ALL_ARGS \
    "

echo $CMD

$CMD
