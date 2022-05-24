#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

DATA_PATH=data/t0-test_text_document
CHECKPOINT_PATH=data


deepspeed --num_gpus 2 pretrain_mp3_gpt.py \
       --num-layers 2 \
       --hidden-size 128 \
       --num-attention-heads 4 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 626 \
       --max-position-embeddings 1024 \
       --train-iters 10000 \
       --lr-decay-iters 5000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles \
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
       --fp16 \
       --tensorboard-dir GPT2
