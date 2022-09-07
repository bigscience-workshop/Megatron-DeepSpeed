# Inference scripts for BLOOM

## BLOOM Inference solutions

Here are some benchmark resuls on JeanZay's 8x80GB A100 node w/ 512GB of CPU memory:

All benchmarks are doing greedy generation of 100 token outputs:
```
Generate args {'max_length': 100, 'do_sample': False}
```
The input prompt is comprised of just a few tokens.

Throughput in msecs on 8x80GB gpus:

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

Now let's look at the power of quantized int8-based models provided by Deepspeed-Inference and BitsNBytes, as it requires only half the original GPU memory of inference in bfloat16 or float16.

Throughput in msecs 4x80GB A100:

| project      \ bs |      1 |     8 |    16 |    32 |   64 | 128  |
| :---------------- | :----- | :---- | :---- | :---- | :--- | :--- |
| accelerate   int8 | 284.15 | 40.14 | 21.97 |  oom  |      |      |
| ds-inference int8 | 156.51 | 20.11 | 10.38 |  5.50 | 2.96 | oom  |
|                   |        |       |       |       |      |      |

To get the benchmark results simply add `--benchmark` to any of these 3 scripts discussed below.


## Deepspeed-Inference

Deepspeed-Inference uses Tensor-Parallelism and efficient fused CUDA kernels:
https://www.deepspeed.ai/tutorials/inference-tutorial/

### Setup

```
pip install deepspeed>=0.7.3
```

### Run

1. the fastest approach is to use a tp-pre-sharded checkpoint that takes only ~1min to load, as compared to 10min for non-presharded bloom checkpoint


```
deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-fp16
```

1a.
if you want to run the original bloom checkpoint, which once loaded will run at the same throughput as the previous solution, but the loading will take 10-20min:

```
deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-inference.py --name bigscience/bloom
```

2a. The 8bit quantized version requires you to have only half the GPU memory of the normal half precision version:


```
deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-int8 --dtype int8
```

Here we used `microsoft/bloom-deepspeed-inference-int8` and also told the script to run in `int8`.

And of course, just 4x80GB A100 gpus is now sufficient:

```
deepspeed --num_gpus 4 scripts/bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-int8 --dtype int8
```



## HF Accelerate

HF Accelerate can use naive Pipeline Parallelism to load a huge model over multiple GPUs:
https://github.com/huggingface/accelerate

### Setup

```
pip install transformers>=4.21.3 accelerate>=0.12.0
```


### Run


```
python scripts/bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --batch_size 1 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=1.txt
```

To activate the 8bit quantized solution first install `bitsnbytes`:

```
pip install bitsandbytes
```

and then add `--dtype int8` to the previous command line:

```
python scripts/bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --dtype int8 --batch_size 1 --benchmark 2>&1 | tee bloom-int8-accelerate-inference_bs=4.txt
```

if you have more that 4 GPUs you can tell it to use only 4 with:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --dtype int8 --batch_size 1 --benchmark 2>&1 | tee bloom-int8-accelerate-inference_bs=4.txt
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
deepspeed --num_gpus 8 scripts/bloom-inference-scripts/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 1 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=1.txt
```

You can also try the offloading solutions with just one small GPU, which will take a long time to run, but if you don't have 8 huge GPUs this is as good as it gets.


CPU-Offload (1x gpus):
```
deepspeed --num_gpus 1 scripts/bloom-inference-scripts/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 8 --cpu_offload --benchmark 2>&1 | tee bloom-ds-zero-inference-cpu_offload_bs=8.txt
```

NVMe-Offload (1x gpus):
```
deepspeed --num_gpus 1 scripts/bloom-inference-scripts/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 8 --nvme_offload_path=/path/to/nvme_offload --benchmark 2>&1 | tee bloom-ds-zero-inference-nvme_offload_bs=8.txt
```

make sure to adjust `/path/to/nvme_offload` to somewhere you have ~400GB of free memory on a fast NVMe drive.
