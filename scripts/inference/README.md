# Inference scripts for BLOOM

## Deepspeed-Inference

Tensor-Parallelism and efficient fused CUDA kernels:
https://www.deepspeed.ai/tutorials/inference-tutorial/

```
deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom --batch_size 1
```

Performance on a single node of 8x80GB A100 w/ 512GB CPU RAM (JeanZay) - just a batch of 1 (would be more efficient to run a larger batch)

Adding `--benchmark` to activate the benchmarks


BS=1
```
deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom --batch_size 1 --benchmark
[...]

```

While processing memory per process:

-  GPU: ~50GB
-  CPU: ~10GB


BS=8
```
$ deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom --batch_size 8 --benchmark
[...]

```

BS=64

```
$ deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom --batch_size 64 --benchmark


```


## Deepspeed ZeRO-Inference

https://www.deepspeed.ai/tutorials/zero/

```
$ deepspeed --num_gpus 8 bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 1 --benchmark
[...]




```

```
$ deepspeed --num_gpus 8 bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 8 --benchmark
[...]



```


```
$ deepspeed --num_gpus 8 bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 64 --benchmark
[...]



```
