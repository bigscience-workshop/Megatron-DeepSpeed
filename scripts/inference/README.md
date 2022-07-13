# Inference scripts for BLOOM

## Deepspeed-Inference

https://www.deepspeed.ai/tutorials/inference-tutorial/

```
deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom
```

Performance on a single node of 8x80GB A100 w/ 512GB CPU RAM (JeanZay):



```
deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom --benchmark

*** Performance stats:
Start to ready to generate: 698.697 secs
Generate 100 tokens: 23.008 secs
Start to finish 721.705 secs
Througput per token: 0.0412 secs
```

While processing memory per process:

-  GPU: ~50GB
-  CPU: ~10GB


## Deepspeed ZeRO-Inference

https://www.deepspeed.ai/tutorials/zero/

```
deepspeed --num_gpus 8 bloom-ds-zero-inference.py --name bigscience/bloom
```
