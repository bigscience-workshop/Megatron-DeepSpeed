# Inference scripts for BLOOM


## Deepspeed-Inference

Tensor-Parallelism and efficient fused CUDA kernels:
https://www.deepspeed.ai/tutorials/inference-tutorial/

### Setup

```
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
git checkout ds-inference/bloom-support
pip install .
```

### Run

```
deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom --batch_size 1
```

Performance on a single node of 8x80GB A100 w/ 512GB CPU RAM (JeanZay) - just a batch of 1 (would be more efficient to run a larger batch)

Adding `--benchmark` to activate the benchmarks


BS=1
```
deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom --batch_size 1 --benchmark 2>&1 | tee bloom-ds-inference_bs=1.txt
[...]

```

While processing memory per process:

-  GPU: ~50GB
-  CPU: ~10GB


BS=8
```
$ deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom --batch_size 8 --benchmark 2>&1 | tee bloom-ds-inference_bs=8.txt
[...]
*** Performance stats:
Throughput per token including tokenize: 5.23 msecs
Start to ready to generate: 683.397 secs
Tokenize and generate 800 (bs=8) tokens: 4.241 secs
Start to finish: 687.638 secs
```

BS=64

```
$ deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom --batch_size 64 --benchmark 2>&1 | tee bloom-ds-inference_bs=64.txt


```


## Deepspeed ZeRO-Inference

https://www.deepspeed.ai/tutorials/zero/

### Setup

```
pip install deepspeed
```


### Run

Note that the script currently runs the same inputs on all GPUs, but you can run a different stream on each GPU, and get `n_gpu` times faster throughput. You can't do that with Deepspeed-Inference.


```
$ deepspeed --num_gpus 8 bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 1 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=1.txt
[...]

*** Performance stats:
Throughput per token including tokenize: 2258.26 msecs
Start to ready to generate: 463.870 secs
Tokenize and generate 100 (bs=1) tokens: 226.826 secs
Start to finish: 690.695 secs
```

divided by 8, it's about 282 msec / token

```
$ deepspeed --num_gpus 8 bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 8 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=8.txt
[...]

Throughput per token including tokenize: 275.91 msecs
Start to ready to generate: 468.254 secs
Tokenize and generate 800 (bs=8) tokens: 221.315 secs
Start to finish: 689.569 secs

```

divided by 8, it's about 4.6 msec / token



```
$ deepspeed --num_gpus 8 bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 64 --benchmark " 2>&1 | tee bloom-ds-zero-inference_bs=64.txt
[...]



```
