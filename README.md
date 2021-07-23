# What is this fork of Megatron-LM and Megatron-DeepSpeed

This is a fork of https://github.com/microsoft/Megatron-DeepSpeed, which in itself is a fork of https://github.com/NVIDIA/Megatron-LM. The former integrates DeepSpeed into the original Megatron-LM code.

This fork in turn will include direct changes to the models needed for the BigScience project. This is the repo we use for this project.

In addition various code bits and lots of docs are to be found at https://github.com/bigscience-workshop/bigscience.

**Important**: **Forking this repo** If you need to fork this repo to your personal account, github won't let you if you already forked either https://github.com/microsoft/Megatron-DeepSpeed or https://github.com/NVIDIA/Megatron-LM, this is a [strange limitation of github](https://stackoverflow.com/questions/6675994/is-it-possible-to-fork-a-fork-in-github) that they don't seem to plan to fix. So if you have commit access to this repo you can use a PR branch instead. If you don't, then you will need to delete the previously existing fork (first making sure you don't lose any of your work in that fork), and then you can fork this repo. This also means that one can't PR only into all 3 repos becase PR requires a fork of each and github won't let you do it. That's a problem.

Please note that the rest of this page has been trimmed to only include the info relevant to the BigScience project and also updated to usage with the integrated Deepspeed. You will find the original page with all the tables and training info on Bert and T5 [here](https://github.com/NVIDIA/Megatron-LM).

# Setup

```
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed
cd Megatron-DeepSpeed
pip install -r requirements.txt
```

Then install `apex` and `deepspeed` either via their homepages, or as explained for [apex](
https://github.com/bigscience-workshop/bigscience/tree/master/jz/envs#apex) and
[deepspeed](https://github.com/bigscience-workshop/bigscience/tree/master/jz/envs#deepspeed). The instructions are for JZ, so you may need to adjust the scripts to your setup.


# Usage

After installation, there are several possible workflows. The most comprehensive is:
1. Data preprocessing
2. Pretraining
3. Finetuning (Optional for zero-shot tasks)
4. Downstream task evaluation or text generation

However, steps 1 and 2 can be replaced by using one of the pretrained models mentioned above.

We've provided several scripts for pretraining both BERT and GPT in [`examples`](./examples) directory, as well as scripts for both zero-shot and fine-tuned downstream tasks including MNLI, RACE, WikiText103, and LAMBADA evaluation. There is also a script for GPT interactive text generation.

# Training

## Vocab

The GPT [vocab file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json) and [merge table](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt) can be downloaded directly.

## Data Preprocessing

The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
```
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
```

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap, cached index file, or the lazy loader format use `preprocess_data.py`. Set the `--dataset-impl` flag to `mmap`, `cached`, or `lazy`, respectively (default is `mmap`).

An example script to prepare data for GPT training is:

```
python tools/preprocess_data.py \
    --input my-corpus.json \
    --output-prefix my-gpt2 \
    --vocab gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod
```

The output will be two files named, in this case, `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`. The `--data-path` specified in later GPT training is the full path and new filename, but without the file extension.

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).


## GPT Pretraining

The `examples/pretrain_gpt.sh` script runs single GPU 345M parameter GPT pretraining. As mentioned above, single GPU training is primarily intended for debugging purposes, as the code is optimized for distributed training.

It follows largely the same format as the previous BERT script with a few notable differences: the tokenization scheme used is BPE (which requires a merge table and a `json` vocabulary file) instead of WordPiece, the model architecture allows for longer sequences (note that the max position embedding must be greater than or equal to the maximum sequence length), and the `--lr-decay-style` has been set to cosine decay.  Note that the `--data-path` now includes the additional `_text_document` suffix added in preprocessing, but does not include the file extensions.

```
CHECKPOINT_PATH=checkpoints/gpt2
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=my-gpt2_text_document

GPT_ARGS=" \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --lr-warmup-fraction .01 \
    --fp16 \
    "

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-activations \
    "

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    "

CMD="pretrain_gpt.py $GPT_ARGS $OUTPUT_ARGS $DATA_ARGS"

N_GPUS=1

LAUNCHER="deepspeed --num_gpus $N_GPUS"

$LAUNCHER $CMD
```

Note, we replaced `python` with `deepspeed --num_gpus 1`. For multi-gpu training update `--num_gpus` to the number of GPUs you have.

For multi-node training you will either need to create a `hostfile` which defines all the nodes as explained [here](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) or in the SLURM environment it might not work and you will need to use:

```
CMD=<as above>

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000
GPUS_PER_NODE=4
NNODES=16

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'
```

For a single GPU the other approach is to emulate `distributed` with:
```
MASTER_ADDR=localhost MASTER_PORT=9994 RANK=0 LOCAL_RANK=0 python pretrain_gpt.py ...
```

Further command line arguments are described in the source file [`arguments.py`](./megatron/arguments.py).


## Using any pretrained tokenizer

Thanks to @sbmaruf, any HF pretrained tokenizer may be used instead of the Megatron-provided BERT/GPT/T5 tokenizers. You'll need to run preprocessing yourself (`tools/preprocess_data.py`), using `tokenizer-type=PretrainedFromHF` and `tokenizer-name-or-path=<your_tokenizer>`. For example, `python tools/preprocess_data.py --input ~/c4_en_train.jsonl --output-prefix c4_en_train --dataset-impl mmap --tokenizer-type PretrainedFromHF --tokenizer-name-or-path t5-small --workers 30 --append-eod`

## Distributed Pretraining

The `examples/pretrain_{bert,gpt,t5}_distributed.sh` scripts use the PyTorch distributed launcher for distributed training. As such, multi-node training can be achieved by properly setting environment variables and using `init_method='env://'` in the launcher. See the official PyTorch [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization). By default, multi-node training uses the [nccl](https://developer.nvidia.com/nccl) distributed backend. A simple set of additional arguments and the use of the PyTorch distributed module with the Python flag `-m torch.distributed.launch`, detailed below, are the only additional requirements to adopt distributed training.

We use two types of parallelism: data and model parallelism. We facilitate two distributed data parallel implementations: a simple one of our own that performs gradient all-reduce at the end of back propagation step, and Torch's distributed data parallel wrapper that overlaps gradient reduction with back propagation computation. To switch between these two options use `--DDP-impl local` or `--DDP-impl torch`, respectively. As expected, Torch distributed data parallelism is more efficient at larger model sizes. For example, for the 8.3 billion parameters model running on 512 GPUs, the scaling increases from 60% to 76% when Torch's distributed data parallel is used. However, the overlapping method requires more memory and for some configurations (e.g., 2.5 billion parameters using 2-way model parallel and 1.2 billion parameters with no model parallel) can make the overall training slower as a result. We empirically found that using a smaller model in those cases improves the training time.

Second, we developed a simple and efficient two-dimensional model-parallel approach. To use tensor model parallelism (splitting execution of a single transformer module over multiple GPUs), add the `--tensor-model-parallel-size` flag to specify the number of GPUs among which to split the model, along with the arguments passed to the distributed launcher as mentioned above. To use pipeline model parallelism (sharding the transformer modules into stages with an equal number of transformer modules on each stage, and then pipelining execution by breaking the batch into smaller microbatches), use the `--pipeline-model-parallel-size` flag to specify the number of stages to split the model into (e.g., splitting a model with 24 transformer layers across 4 stages would mean each stage gets 6 transformer layers each).

<!-- The number of microbatches in a per-pipeline minibatch is controlled by the `--num-microbatches-in-minibatch` argument. With `WORLD_SIZE` GPUs, `TENSOR_MP_SIZE` tensor-model-parallel size, `PIPELINE_MP_SIZE` pipeline-model-parallel-size, `WORLD_SIZE`/(`TENSOR_MP_SIZE` * `PIPELINE_MP_SIZE`) GPUs will be used for data parallelism. The default values for `--tensor-model-parallel-size` and `--pipeline-model-parallel-size` is 1, which will not implement either form of model parallelism. -->

We have examples of how to use these two different forms of model parallelism the example scripts ending in `distributed_with_mp.sh`, note that pipeline parallelism is not currently supported in the T5 model:

Other than these minor changes, the distributed training is identical to the training on a single GPU.

Distributed training:

**see the details on how to do distributed training with the `deepspeed` launcher a few sections up**
XXX: The following needs to be updated:

```
WORLD_SIZE=8
TENSOR_MP_SIZE=2
PIPELINE_MP_SIZE=2

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000"

CHECKPOINT_PATH=&#60;same as above&#62;
VOCAB_FILE=&#60;same as above&#62;
DATA_PATH=&#60;same as above&#62;
MODEL_ARGS=&#60;same as above&#62;
OUTPUT_ARGS=&#60;same as above&#62;

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./pretrain_<model>.py \
    $MODEL_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --tensor-model-parallel-size $TENSOR_MP_SIZE \
    --pipeline-model-parallel-size $PIPELINE_MP_SIZE \
    --DDP-impl torch
```

## GPT-3 Example

In `examples/pretrain_gpt3_175B.sh` we have provided an example of how to configure Megatron to run [GPT-3](https://arxiv.org/abs/2005.14165) with 175 billion parameters on 1024 GPUs. The script is designed for [slurm](https://slurm.schedmd.com/documentation.html) with [pyxis](https://github.com/NVIDIA/pyxis) plugin but can be easily adopted to any other scheduler. It uses 8-way and 16-way tensor and pipeline parallelism, respectively. With options `global-batch-size 1536` and `rampup-batch-size 16 16 5859375`, the training will start with global batch size 16 and linearly increase the global batch size to 1536 over 5,859,375 samples with incrmeental steps 16. The training dataset can be either a single set or a multiple datasets combined with a set of weights.

With full global batch size of 1536 on 1024 A100 GPUs, each iteration takes around 32 seconds resulting in 138 teraFLOPs per GPU which is 44% of the theoretical peak FLOPs.


# Evaluation and Tasks

We provide several command line arguments, detailed in the scripts listed below, to handle various zero-shot and fine-tuned downstream tasks. However, you can also finetune your model from a pretrained checkpoint on other corpora as desired. To do so, simply add the `--finetune` flag and adjust the input files and training parameters within the original training script. The iteration count will be reset to zero, and the optimizer and internal state will be reinitialized. If the fine-tuning is interrupted for any reason, be sure to remove the `--finetune` flag before continuing, otherwise the training will start again from the beginning.

Because evaluation requires substantially less memory than training, it may be advantageous to merge a model trained in parallel for use on a single GPU in downstream tasks. The following script accomplishes this. Currently only tensor model parallelism is supported on input and pipeline model parallelsim on the output. This example reads in a model with 2-way tensor model parallelism and writes out a model with 2-way pipeline model parallelism.

```
TENSOR_MODEL_PARALLEL_SIZE=2
TARGET_PIPELINE_MODEL_PARALLEL_SIZE=2

VOCAB_FILE=bert-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_345m

WORLD_SIZE=$TENSOR_MODEL_PARALLEL_SIZE python tools/merge_mp_partitions.py \
    --model-type BERT \
    --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE \
    --pipeline-model-parallel-size 1 \
    --target-pipeline-model-parallel-size $TARGET_PIPELINE_MODEL_PARALLEL_SIZE \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file $VOCAB_FILE \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH/merged

```

Several downstream tasks are described for both GPT and BERT models below. They can be run in distributed and model parallel modes with the same changes used in the training scripts.

## GPT Text Generation
`bash examples/generate_text.sh`

We generate text samples using largely the GPT pretraining script. Few changes need to make, such as we need to provide the path to the pretrained checkpoint, the length of the output samples, whether to generate texts unconditionally (`--num-samples` to denote how many samples to generate) or conditional (need to pass `--sample-input-file <filename>` where each line of the file will be used as the conditional texts). There are few optional parameters to play, e.g. `top-k`, `top-p`, or `greedy` (set top-k and top-p to 0) sampling..

```
CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
GPT_ARGS=&#60;same as those in <a href="#gpt-pretraining">GPT pretraining</a> above&#62;

MAX_OUTPUT_SEQUENCE_LENGTH=1024
TEMPERATURE=1.0
TOP_P=0.9
NUMBER_OF_SAMPLES=2
OUTPUT_FILE=samples.json

python tools/generate_samples_gpt.py \
    $GPT_ARGS \
    --load $CHECKPOINT_PATH \
    --out-seq-length $MAX_OUTPUT_SEQUENCE_LENGTH \
    --temperature $TEMPERATURE \
    --genfile $OUTPUT_FILE \
    --num-samples $NUMBER_OF_SAMPLES \
    --top_p $TOP_P \
    --recompute
```

## GPT Evaluation
We include example scripts for GPT evaluation on WikiText perplexity evaluation and LAMBADA Cloze accuracy.

### WikiText Perplexity Evaluation
For even comparison with prior works, we evaluate perplexity on the word-level [WikiText-103 test dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), and appropriately compute perplexity given the change in tokens when using our subword tokenizer.

We use the following command to run WikiText-103 evaluation on a 345M parameter model.
```
TASK="WIKITEXT103"

VALID_DATA=&#60;wikitext path&#62;.txt
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m

COMMON_TASK_ARGS=" \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --fp16 \
    --vocab-file $VOCAB_FILE"

python tasks/main.py \
    --task $TASK \
    $COMMON_TASK_ARGS \
    --valid-data $VALID_DATA \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file $MERGE_FILE \
    --load $CHECKPOINT_PATH \
    --micro-batch-size 8 \
    --checkpoint-activations \
    --log-interval 10 \
    --no-load-optim \
    --no-load-rng
```
