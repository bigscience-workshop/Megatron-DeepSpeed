
# usage:
#
# direct HF
# deepspeed --num_gpus 1 bloom-test.py --name bigscience/bloom-350m
#
# via deepspeed/zero-3 inference
# deepspeed --num_gpus 1 bloom-test.py --name bigscience/bloom-350m --deepspeed
#


import torch
import deepspeed
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
from argparse import ArgumentParser
import os
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
from transformers.deepspeed import HfDeepSpeedConfig
import torch.distributed as dist

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str)
parser.add_argument("--local_rank", required=False, type=int)
#parser.add_argument("--deepspeed", action="store_true")
args = parser.parse_args()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank,
            world_size))

config = AutoConfig.from_pretrained(args.name)

model_hidden_size = config.hidden_size

train_batch_size = 1 * world_size
model_name = args.name
dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16

# Note: you need to edit nvme_path to an actual path on your filesystem where the model will be offloaded to
ds_config = {
    "fp16": {
        "enabled": dtype == torch.float16,
    },
    "bf16": {
        "enabled": dtype == torch.bfloat16,
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/mnt/nvme0/offload/",
            "pin_memory": True,
            "buffer_count": 4,
            "buffer_size": 4e9, # for bloom, otherwise the default 1e8 should be enough
            "fast_init": False
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}

deepspeed.runtime.utils.see_memory_usage('pre-init', force=True)
#if args.deepspeed:
dschf = HfDeepSpeedConfig(ds_config)

model = AutoModelForCausalLM.from_pretrained(model_name)

#generator = pipeline('text-generation', model=args.name, device=local_rank, framework="pt")
deepspeed.runtime.utils.see_memory_usage('post-init', force=True)

ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()
#    generator.model = ds_engine.module
deepspeed.runtime.utils.see_memory_usage('post-ds-init', force=True)

# if args.deepspeed:
#     ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
#     ds_engine.module.eval()
# #    generator.model = ds_engine.module
#     deepspeed.runtime.utils.see_memory_usage('post-ds-init', force=True)
# else:
#     dist.init_process_group("nccl")
#     model = model.to(device=local_rank)

#response = generator('DeepSpeed is', min_length=50, max_length=50, do_sample=False)

text_in = 'DeepSpeed is'

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(text_in, return_tensors="pt").to(device=local_rank)
with torch.no_grad():
    model = ds_engine.module if args.deepspeed else model
    outputs = model.generate(inputs, synced_gpus=True, min_length=50, max_length=50, do_sample=False)
text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"in={text_in}\nout={text_out}")
