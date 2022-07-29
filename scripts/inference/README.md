# Inference scripts for BLOOM

To run a server using HuggingFace (requires accelerate to be installed):
```
python scripts/inference/inference-server.py --model_name bigscience/bloom --dtype bf16 --log_file data.log --host 127.0.0.1 --port 5000 --inference_method hf_accelerate
```

To run a server using deepspeed:
```
deepspeed --num_gpus 8 scripts/inference/inference-server.py --model_name bigscience/bloom --dtype fp16 --log_file data.log --host 127.0.0.1 --port 5000 --inference_method deepspeed
```