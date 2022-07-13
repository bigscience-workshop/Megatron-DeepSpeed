# Inference scripts for BLOOM

## Deepspeed-Inference

Tensor-Parallelism and efficient fused CUDA kernels:
https://www.deepspeed.ai/tutorials/inference-tutorial/

```
deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom
```

Performance on a single node of 8x80GB A100 w/ 512GB CPU RAM (JeanZay) - just a batch of 1 (would be more efficient to run a larger batch)

Adding `--benchmark` to activate the benchmarks

```
deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom --benchmark

*** Performance stats:
Throughput per token: 40.73 msecs
Start to ready to generate: 673.429 secs
Tokenize and generate 100 tokens: 4.089 secs
Start to finish: 677.518 secs
```

While processing memory per process:

-  GPU: ~50GB
-  CPU: ~10GB


## Deepspeed ZeRO-Inference

https://www.deepspeed.ai/tutorials/zero/

```
deepspeed --num_gpus 8 bloom-ds-zero-inference.py --name bigscience/bloom
```
