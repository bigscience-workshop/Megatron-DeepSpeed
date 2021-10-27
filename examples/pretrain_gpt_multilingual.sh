#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

# paths to multilingual preprocessed datasets
DATA_PATH_EN=<Specify path and file prefix>_text_document
DATA_PATH_AR=<Specify path and file prefix>_text_document
DATA_PATH_KR=<Specify path and file prefix>_text_document
DATA_PATH_KR=<Specify path and file prefix>_text_document
DATA_PATH_JP=<Specify path and file prefix>_text_document

CHECKPOINT_PATH=<Specify path>


deepspeed --num_gpus 1 pretrain_gpt.py \
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
       --data-path 0.01 $DATA_PATH_EN 0.32 $DATA_PATH_KR 0.33 $DATA_PATH_JP 0.33 $DATA_PATH_AR \
       --extra-eval-data-path \
       VALID-EN 1.0 $DATA_PATH_EN, \
       VALID-FR 1.0 $DATA_PATH_FR, \
       VALID-KR-JP-AR 0.3 $DATA_PATH_KR 0.3 $DATA_PATH_JP 0.3 $DATA_PATH_AR \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
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
