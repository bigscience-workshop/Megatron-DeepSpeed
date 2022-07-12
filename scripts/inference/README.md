# Inference scripts for BLOOM

## Deepspeed-Inference

https://www.deepspeed.ai/tutorials/inference-tutorial/

```
deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom
```

## Deepspeed ZeRO-Inference

https://www.deepspeed.ai/tutorials/zero/

```
deepspeed --num_gpus 8 bloom-ds-zero-inference.py --name bigscience/bloom
```
