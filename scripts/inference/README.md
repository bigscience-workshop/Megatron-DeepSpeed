# Inference scripts for BLOOM

To run a server using HuggingFace (requires [accelerate](https://github.com/huggingface/accelerate) to be installed):
```
python scripts/inference/bloom-accelerate-server.py --model_name bigscience/bloom --dtype bf16 --log_file data.log --host $ADDRESS --port $PORT
```

To run a server using deepspeed (requires [DeepSpeed MII](https://github.com/microsoft/DeepSpeed-mii) to be installed):
```
export DS_CACHE=<path where to dump pre-sharded 8-TP checkpoints>

deepspeed --num_gpus 8 scripts/inference/cache-ds-model.py --model_name bigscience/bloom --dtype fp16

python scripts/inference/bloom-ds-server.py --model_name bigscience/bloom --dtype fp16 --log_file data.log --host $ADDRESS --port $PORT
```

Usage:
Currently, the script supports 3 method:
1. The main generate method
```
curl -H "Content-Type: application/json" -X POST -d '{ "input_text": "India is a country of", "top_k": "5", "top_p": "0.9", "temperature": "0.7", "min_length": "1", "max_new_tokens": "40", "return_type": "output_only" }' http://$ADDRESS:$PORT/generate/
```
returns
```
{"num_output_tokens":40,"output_text":" many languages and cultures. The country is a melting pot of different cultures and languages. The country is a home to more than 1.2 billion people. The country is a home to more than 22","query_id":8,"throughput":"2.066 tokens/s","total_time_taken":"19.358 s"}
```
2. Method that returns the model description
```
curl -H "Content-Type: application/json" -X GET http://$ADDRESS:$PORT/about/
```
returns
```
Please don't send any personal information to this endpoint. We are logging your data.

Usage:
A request object should look like:
{
    input_text: "Hello, I'm a model",
    "top_k": 5,
    "top_p": 0.9,
    "temperature": 0.7,
    "min_length": 1,
    "max_new_tokens": 40,
    "return_type": "output_only"
}

Default values (use if not provided in request object):
top_k = 50
top_p = 1
temperature = 1
min_length = 1
max_new_tokens = 40
return_type = "both_input_output"
```
3. Method to check GPU usage
```
curl -H "Content-Type: application/json" -X GET http://$ADDRESS:$PORT/gpu/
```
returns the nvidia-smi output
## BLOOM Inference solutions

Here are some stats on JeanZay's 8x80GB A100 node w/ 512GB of CPU memory:

All benchmarks are doing greedy generation of 100 token outputs:
```
Generate args {'min_length': 100, 'max_length': 100, 'do_sample': False}
```
The inputs are just a few tokens.

Throughput in msecs:

| project \ bs |      1 |     8 |    16 |    32 |    64 |  128 |
| :----------- |  :---- | :---- | :---- | :---- | :---- | :--- |
| accelerate   | 230.38 | 31.78 | 17.84 | 10.89 |  oom  | omm  |
| ds-inference |  40.57 |  5.23 |       |       |  2.77 | 0.66 |
| ds-zero      |    283 | 34.88 | oom   |  oom  |  oom  | oom  |


Start to ready to generate in secs:

| project \ bs |    1 |    8 |   16 |   32 |   64 |  128 |
| :----------- | :--- | :--- | :--- | :--- | :--- | :--- |
| accelerate   |  121 |  120 |  113 |  118 |      |      |
| ds-inference |  662 |  673 |      |      |  685 |  654 |
| ds-zero      |  462 |  463 |      |      |      |      |
|              |      |      |      |      |      |      |


DS-Inference load time (start to ready to generate) will become much faster soon. Once we stop relying on ds-zero to instantiate the model on gpu. The plan is to pre-shard the weights TP-wise for 8x and 16x gpus and load them directly on each gpu. Will probably be under 1min.


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

```
deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom
```

Performance on a single node of 8x80GB A100 w/ 512GB CPU RAM (JeanZay) - just a batch of 1 (would be more efficient to run a larger batch)

Adding `--benchmark` to activate the benchmarks


BS=1
```
$ deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom --batch_size 1 --benchmark 2>&1 | tee bloom-ds-inference_bs=1.txt
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

BS=128

```
$ deepspeed --num_gpus 8 scripts/inference/bloom-ds-inference.py --name bigscience/bloom --batch_size 128 --benchmark 2>&1 | tee bloom-ds-inference_bs=128.txt




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
$ deepspeed --num_gpus 8 scripts/inference/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 1 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=1.txt
[...]
*** Performance stats:
Throughput per token including tokenize: 282.93 msecs
Start to ready to generate: 501.871 secs
Tokenize and generate 800 (bs=1) tokens: 226.188 secs
Start to finish: 728.060 secs
```


BS=8

```
$ deepspeed --num_gpus 8 scripts/inference/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 8 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=8.txt
[...]

*** Performance stats:
Throughput per token including tokenize: 34.57 msecs
Start to ready to generate: 482.132 secs
Tokenize and generate 6400 (bs=8) tokens: 221.236 secs
Start to finish: 703.368 secs
```

BS=16 and higher OOMs

```
$ deepspeed --num_gpus 8 scripts/inference/bloom-ds-zero-inference.py --name bigscience/bloom --batch_size 16 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=16.txt
[...]
OOM

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
$ python scripts/inference/bloom-accelerate-inference.py --name bigscience/bloom --batch_size 1 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=1.txt
[...]


```

BS=8
```
$ python scripts/inference/bloom-accelerate-inference.py --name bigscience/bloom --batch_size 8 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=8.txt
[...]


```

BS=16
```
$ python scripts/inference/bloom-accelerate-inference.py --name bigscience/bloom --batch_size 16 --benchmark 2>&1 | tee bloom-ds-zero-inference_bs=16.txt
[...]


```