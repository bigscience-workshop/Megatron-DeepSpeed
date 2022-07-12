
# usage:
# deepspeed --num_gpus 1 bloom-inference.py --name bigscience/bloom-350m
#

import glob
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
import deepspeed
import io
import sys
import json
import os
import gc
import torch
import torch.distributed as dist

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str)
parser.add_argument("--local_rank", required=False, type=int)
parser.add_argument("--deepspeed", action="store_true")
args = parser.parse_args()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

def get_checkpoint_files(pretrained_model_name_or_path):
    # XXX: I just hacked this one together to automatically handle the fetching of the model file or
    # shards into cache and returning the cached entries - note that I removed most arguments

    from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, cached_path, hf_bucket_url
    from transformers.utils.hub import EntryNotFoundError

    cache_dir = None
    is_sharded = False
    filename = WEIGHTS_NAME
    archive_file = hf_bucket_url(pretrained_model_name_or_path, filename=filename)

    try:
        resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        return [resolved_archive_file]

    except EntryNotFoundError:
        if filename == WEIGHTS_NAME:
            # Maybe the checkpoint is sharded, we try to grab the index name in this case.
            archive_file = hf_bucket_url(
                pretrained_model_name_or_path,
                filename=WEIGHTS_INDEX_NAME,
            )
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
            )
            is_sharded = True

    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
        )

        return resolved_archive_file


model_name = args.name

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# XXX: can't automatically derive dtype via config's `from_pretrained`
dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16

kernel_inject = True
#kernel_inject = False

if kernel_inject:
    # XXX: for now ds-inference only works with fp16
    dtype = torch.float16
else:
    dtype = torch.bfloat16

#dtype = config.dtype
print(dtype)

model_hidden_size = config.hidden_size
train_batch_size = 1 * world_size

ds_config = {
    "fp16": {
        "enabled": dtype == torch.float16,
    },
    "bf16": {
        "enabled": dtype == torch.bfloat16,
    },
    "zero_optimization": {
        "stage": 3,
#        "offload_param": {
#            "device": "none",
#            "pin_memory": True
#        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 0
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}

print(ds_config)

dschf = HfDeepSpeedConfig(ds_config)

torch.cuda.empty_cache()
gc.collect()
deepspeed.runtime.utils.see_memory_usage('pre-from-pretrained', force=True)

model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

deepspeed.runtime.utils.see_memory_usage('post-from-pretrained', force=True)

model = model.eval()


ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()
model = ds_engine.module

# a must to remove ZeRO hooks!
ds_engine.destroy()

from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
def ds_clear_params(ds_engine):
    for p in ds_engine.parameters():
        if hasattr(p, "ds_tensor"):
            p.ds_tensor = torch.empty(0, dtype=p.dtype, device=p.device)
            p.ds_status = ZeroParamStatus.NOT_AVAILABLE

# this frees the memory used by zero
ds_clear_params(ds_engine)

#ds_engine.module = None
del ds_engine

torch.cuda.empty_cache()
gc.collect()
deepspeed.runtime.utils.see_memory_usage('post-init-ds-zero-init', force=True)


checkpoints_json = "checkpoints.json"
def write_checkponts_json():

    with io.open(checkpoints_json, 'w', encoding='utf-8') as f:

        checkpoint_dir = "/gpfsscratch/rech/six/commun/uan68tv-model-conversion/bloom"
        checkpoint_files = glob.glob(f"{checkpoint_dir}/*bin")
        #checkpoint_files = get_checkpoint_files(model_name)

        print("Checkpoint files:", checkpoint_files)

        data = {
            "type": "BLOOM-176B",
            "checkpoints": checkpoint_files,
            "version": 1.0
        }
        json.dump(data, f)

rank = dist.get_rank()
if rank == 0:
    write_checkponts_json()
dist.barrier()

#print("before deepspeed.init_inference")
torch.cuda.empty_cache()
gc.collect()
deepspeed.runtime.utils.see_memory_usage('pre-ds-inference-init', force=True)

# use one of these args to `init_inference`
# 1. injection_policy is the slower version, but it's plain pytorch so it'll always work
# 2. replace_with_kernel_inject is the faster one (fast fused kernels)

if kernel_inject:
    kwargs = dict(replace_with_kernel_inject=True)
else:
    kwargs = dict(injection_policy={BloomBlock: ('self_attention.dense', 'mlp.dense_4h_to_h')})

#checkpoints_json=None
model = deepspeed.init_inference(model,
                                 mp_size=world_size,
                                 dtype=torch.half,
                                 checkpoint=checkpoints_json,
                                 **kwargs
                                 )
#                                 injection_policy={BloomBlock: ('self_attention.dense', 'mlp.dense_4h_to_h')},
#                                 #replace_with_kernel_inject=True

torch.cuda.empty_cache()
gc.collect()
deepspeed.runtime.utils.see_memory_usage('post-ds-inference-init', force=True)

#print("after deepspeed.init_inference")
model = model.module

text_in = 'DeepSpeed is'

tokens = tokenizer(text_in, return_tensors="pt")

for t in tokens:
    if torch.is_tensor(tokens[t]):
        tokens[t] = tokens[t].to(torch.cuda.current_device())

with torch.no_grad():
    gen_tokens = model.generate(
        **tokens,
        min_length=50,
        max_length=50,
        do_sample=False,
    )


text_out = tokenizer.batch_decode(gen_tokens)[0]

print(f"in={text_in}\nout={text_out}")

torch.cuda.empty_cache()
gc.collect()
deepspeed.runtime.utils.see_memory_usage('end-of-run', force=True)

