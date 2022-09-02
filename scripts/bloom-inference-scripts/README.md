# Inference scripts for BLOOM

## BLOOM Inference solutions

Here are some stats on JeanZay's 8x80GB A100 node w/ 512GB of CPU memory:

All benchmarks are doing greedy generation of 100 token outputs:
```
Generate args {'max_length': 100, 'do_sample': False}
```
The inputs are just a few tokens.

Throughput in msecs 8x80GB gpus:

| project      \ bs |      1 |     8 |    16 |    32 |   64 |  128 |  256 | 512  |
| :---------------- | :----- | :---- | :---- | :---- | :--- | :--- | :--- | :--- |
| accelerate   bf16 | 230.38 | 31.78 | 17.84 | 10.89 |  oom |      |      |      |
| accelerate   int8 | 286.56 | 40.92 | 22.65 | 13.27 |  oom |      |      |      |
| ds-inference fp16 |  44.02 |  5.70 |  3.01 |  1.68 | 1.00 | 0.69 |  oom |      |
| ds-inference int8 |  89.09 | 11.44 |  5.88 |  3.09 | 1.71 | 1.02 | 0.71 | oom  |
| ds-zero           |    283 | 34.88 |   oom |       |      |      |      |      |
|                   |        |       |       |       |      |      |      |      |

Start to ready to generate in secs (mainly loading and data preparation time):

| project                 |      |
| :---------------------- | :--- |
| accelerate              |  121 |
| ds-inference shard-int8 |   61 |
| ds-inference shard-fp16 |   60 |
| ds-inference unsharded  |  662 |
| ds-zero                 |  462 |

Now let's look at the actual power of int8, as it requires only half the original GPU memory.

Throughput in msecs 4x80GB A100:

| project      \ bs |      1 |     8 |    16 |    32 |   64 | 128  |
| :---------------- | :----- | :---- | :---- | :---- | :--- | :--- |
| accelerate   int8 | 284.15 | 40.14 | 21.97 |  oom  |      |      |
| ds-inference int8 | 156.51 | 20.11 | 10.38 |  5.50 | 2.96 | oom  |
|                   |        |       |       |       |      |      |



## Deepspeed-Inference

Tensor-Parallelism and efficient fused CUDA kernels:
https://www.deepspeed.ai/tutorials/inference-tutorial/

### Setup

```
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
pip install .
```

### Run

1. the fastest approach is to use a tp-pre-sharded checkpoint that takes only ~1min to load, as compared to 10min for non-presharded bloom checkpoint


```
deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-fp16
```
1a.
if you want to run the original bloom checkpoint, which will run at the same througput once loaded, but the loading will take 10-20min:

```
deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-inference.py --name bigscience/bloom
```

2a. The quantized version requires you to have only half the GPU memory of the normal version:


```
deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-int8 --dtype int8
```

Here we used `microsoft/bloom-deepspeed-inference-int8` and also told the script to run in `int8`.

And of course, just 4x80GB A100 gpus is now sufficient:

```
deepspeed --num_gpus 4 scripts/bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-int8 --dtype int8
```




Performance on a single node of 8x80GB A100 w/ 512GB CPU RAM (JeanZay) - just a batch of 1 (would be more efficient to run a larger batch)

Adding `--benchmark` to activate the benchmarks


BS=1
```
$ deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-inference.py --name bigscience/bloom --batch_size 1 --benchmark 2>&1 | tee bloom-ds-inference_bs=1.txt
[...]

```

While processing memory per process:

-  GPU: ~50GB
-  CPU: ~10GB


BS=8
```
$ deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-inference.py --name bigscience/bloom --batch_size 8 --benchmark 2>&1 | tee bloom-ds-inference_bs=8.txt
[...]
*** Performance stats:
Throughput per token including tokenize: 5.23 msecs
Start to ready to generate: 683.397 secs
Tokenize and generate 800 (bs=8) tokens: 4.241 secs
Start to finish: 687.638 secs
```

BS=64

```
$ deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-inference.py --name bigscience/bloom --batch_size 64 --benchmark 2>&1 | tee bloom-ds-inference_bs=64.txt




```

BS=128

```
$ deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-inference.py --name bigscience/bloom --batch_size 128 --benchmark 2>&1 | tee bloom-ds-inference_bs=128.txt




```


## HF Accelerate

https://github.com/huggingface/accelerate

### Setup

```
pip install transformers
```



### Run




BS=1
```
$ python scripts/bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --batch_size 1 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=1.txt
[...]


```

BS=8
```
$ python scripts/bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --batch_size 8 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=8.txt
[...]


```

BS=16
```
$ python scripts/bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --batch_size 16 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=16.txt
[...]


```



## Deepspeed ZeRO-Inference

https://www.deepspeed.ai/tutorials/zero/

### Setup

```
pip install deepspeed
```


### Run

Note that the script currently runs the same inputs on all GPUs, but you can run a different stream on each GPU, and get `n_gpu` times faster throughput. You can't do that with Deepspeed-Inference.


BS=1

```
$ deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 1 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=1.txt
[...]
*** Performance stats:
Throughput per token including tokenize: 282.93 msecs
Start to ready to generate: 501.871 secs
Tokenize and generate 800 (bs=1) tokens: 226.188 secs
Start to finish: 728.060 secs
```


BS=8

```
$ deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 8 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=8.txt
[...]

*** Performance stats:
Throughput per token including tokenize: 34.57 msecs
Start to ready to generate: 482.132 secs
Tokenize and generate 6400 (bs=8) tokens: 221.236 secs
Start to finish: 703.368 secs
```

BS=16 and higher OOMs

```
$ deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 16 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=16.txt
[...]
OOM

```
