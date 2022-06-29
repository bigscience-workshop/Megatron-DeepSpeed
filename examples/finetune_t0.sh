#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

DATA_PATH="{ \
        'input_tokens': 'tests/data/t0/ag_news_prompt_inputs_document',  \
       'target_tokens': 'tests/data/t0/ag_news_prompt_targets_document'  \
       }"
CHECKPOINT_PATH="./checkpoints"
TOKENIZER_PATH=gpt2

deepspeed --num_gpus 1 pretrain_t0.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
