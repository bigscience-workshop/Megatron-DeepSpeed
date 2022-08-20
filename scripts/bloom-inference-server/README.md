## Inference solutions for BLOOM 176B
We support HuggingFace accelerate and DeepSpeed Inference for generation.

Required packages:
1. [DeepSpeed](https://github.com/microsoft/DeepSpeed)
1. [DeepSpeed MII](https://github.com/microsoft/DeepSpeed-MII)
1. [HuggingFace accelerate](https://github.com/huggingface/accelerate)

All the provided scripts are tested on 8 A100 80GB GPUs for BLOOM 176B. These scripts might not work for other models or a different number of GPUs.
DS inference only supports fp16 for cli and server application. However, for benchmarking, it supports both fp16 and bf16. bf16 support will be added once DeepSpeed adds suitable CUDA kernels for these.

DS inference is deployed using the DeepSpeed MII library which requires the resharded checkpoints for 8 x Tensor Parallel. The HuggingFace checkpoints can be resharded and cached using the following command:
```shell
deepspeed --num_gpus 8 scripts/bloom-inference-server/cache_ds_checkpoints.py --model_name bigscience/bloom --dtype fp16 --save_mp_checkpoint_path <PATH TO DS CACHED MODEL>
```
Note: Running the above script will consume ~350 GB of disk space and will take some time (~30 minutes), depending on both the speed of your GPUs and storage.

Note: sometimes GPU memory is not freed when DS inference deployment is shutdown. You can free this memory by running:
```python
import mii
mii.terminate("ds_inference_grpc_server")
```
or alternatively, just doing a `killall python` in terminal.

#### BLOOM inference via command-line
This asks for generate_kwargs everytime.
Example: generate_kwargs =
```json
{"min_length": 100, "max_new_tokens": 100, "do_sample": false}
```

1. using HF accelerate
```shell
python scripts/bloom-inference-server/cli.py --model_name bigscience/bloom --dtype bf16 --deployment_framework hf_accelerate --generate_kwargs '{"min_length": 100, "max_new_tokens": 100, "do_sample": false}'
```

2. using DS inference
```shell
python scripts/bloom-inference-server/cli.py --model_name bigscience/bloom --dtype fp16 --deployment_framework ds_inference --save_mp_checkpoint_path <PATH TO DS CACHED MODEL> --generate_kwargs '{"min_length": 100, "max_new_tokens": 100, "do_sample": false}'
```

#### BLOOM server deployment
1. using HF accelerate
```shell
python scripts/bloom-inference-server/server.py --model_name bigscience/bloom --dtype bf16 --deployment_framework hf_accelerate --host <HOST ADDRESS> --port <PORT> --allowed_max_new_tokens 100
```

2. using DS inference
```shell
python scripts/bloom-inference-server/server.py --model_name bigscience/bloom --dtype fp16 --deployment_framework ds_inference --save_mp_checkpoint_path <PATH TO DS CACHED MODEL> --host <HOST ADDRESS> --port <PORT> --allowed_max_new_tokens 100
```

We provide an example [script](examples/server_request.py) to query the BLOOM server is provided.

#### Benchmark system for BLOOM inference
1. using HF accelerate
```shell
python scripts/bloom-inference-server/benchmark.py --model_name bigscience/bloom --dtype bf16 --deployment_framework hf_accelerate --benchmark_cycles 5
```

2. using DS inference
```shell
deepspeed --num_gpus 8 scripts/bloom-inference-server/benchmark.py --model_name bigscience/bloom --dtype fp16 --deployment_framework ds_inference --save_mp_checkpoint_path <PATH TO DS CACHED MODEL> --benchmark_cycles 5
```

3. using DS ZeRO
```shell
deepspeed --num_gpus 8 scripts/bloom-inference-server/benchmark.py --model_name bigscience/bloom --dtype bf16 --deployment_framework ds_zero --benchmark_cycles 5
```

Alternatively, the following shell script will benchmark different batch sizes for the model.
```shell
mkdir -p logs

for bs in {1,2,4,8,16,32,64,128}
do
    python scripts/bloom-inference-server/benchmark.py --model_name bigscience/bloom --dtype bf16 --deployment_framework hf_accelerate --benchmark_cycles 5 --batch_size $bs 2>&1 | tee logs/hf-$bs.log

    deepspeed --num_gpus 8 scripts/bloom-inference-server/benchmark.py --model_name bigscience/bloom --dtype fp16 --deployment_framework ds_inference --save_mp_checkpoint_path <PATH TO DS CACHED MODEL> --benchmark_cycles 5 --batch_size $bs 2>&1 | tee logs/ds-$bs.log

    deepspeed --num_gpus 8 scripts/bloom-inference-server/benchmark.py --model_name bigscience/bloom --dtype bf16 --deployment_framework ds_zero --benchmark_cycles 5 --batch_size $bs 2>&1 | tee logs/ds-zero-$bs.log
done
```

The following will benchmark sequence length for batch size = 1 on DS inference.
```shell
for sq in {1,10,50,100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4000,4500,5000}
do
    deepspeed --num_gpus 8 scripts/bloom-inference-server/benchmark.py --model_name bigscience/bloom --dtype fp16 --batch_size 1 --benchmark_cycles 5 --deployment_framework ds_inference --generate_kwargs '{"do_sample": false, "min_length": '$sq', "max_new_tokens": '$sq'}' 2>&1 | tee logs/ds_$sq.log
done
```
