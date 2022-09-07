# usage:
# deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom
#
# to run benchmarks:
# deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom --benchmark
#


# This is going to improve, but at the moment, the process is a bit cumbersome - we first use
# 1. use Deepspeed-ZeRO to instantiate the model on GPUs, w/o loading the checkpoints,
# 2. free the allocated storage
# 3. start Deepspeed-Inference and only now load the checkpoint
# 4. run generate
# Done.
#


from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
import deepspeed
import gc
import glob
import io
import json
import math
import os
import sys
import time
import torch
import torch.distributed as dist

t_start = time.time()

num_tokens = 100

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
parser.add_argument("--cpu_offload", action="store_true", help="whether to activate CPU offload")
parser.add_argument("--nvme_offload_path", help="whether to activate NVME offload and the path on nvme")
args = parser.parse_args()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

deepspeed.init_distributed('nccl')
rank = dist.get_rank()

def print_rank0(*msg):
    if rank != 0: return
    print(*msg)

### Model loading and instantiating on GPU (via ZeRO)

model_name = args.name

print_rank0(f"*** Loading the model {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# XXX: can't automatically derive dtype via config's `from_pretrained`
dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16

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

if args.cpu_offload and args.nvme_offload_path:
    raise ValueError("Use one of --cpu_offload or --nvme_offload_path and not both")

if args.cpu_offload:
    ds_config["zero_optimization"]["offload_param"] = dict(
        device="cpu",
        pin_memory=True
    )

if args.nvme_offload_path:
    ds_config["zero_optimization"]["offload_param"] = dict(
        device="nvme",
        pin_memory=True,
        nvme_path=args.nvme_offload_path,
        buffer_size=6e8
    )

dschf = HfDeepSpeedConfig(ds_config) # this tells from_pretrained to instantiate directly on gpus

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('pre-from-pretrained', force=True)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

if args.benchmark:
    deepspeed.runtime.utils.see_memory_usage('post-from-pretrained', force=True)

model = model.eval()

print_rank0(ds_config)

ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()
model = ds_engine.module

if args.benchmark:
    t_ready = time.time()


### Generate

print_rank0(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way"
]

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)

print_rank0(f"Generate args {generate_kwargs}")
inputs = input_sentences[:args.batch_size]
def generate():
    """ returns a list of zipped inputs, outputs and number of new tokens """

    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

    outputs = model.generate(**input_tokens, **generate_kwargs)

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o-i for i,o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)

# XXX: this is currently doing world_size streams on world_size gpus, so we can feed it different inputs on each! and hence the time can be divided by world_size

# warmup is a must if measuring speed as it's when all the optimizations are performed
# e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
_ = generate()

t_generate_start = time.time()
pairs = generate()
t_generate_span = time.time() - t_generate_start
for i,o,_ in pairs:
    print_rank0(f"{'-'*60}\nin={i}\nout={o}\n")


if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('end-of-run', force=True)

### Benchmark

if args.benchmark:
    print_rank0(f"*** Running benchmark")

    # warm up
    for i in range(1):
        _ = generate()
    torch.cuda.synchronize()

    # benchmark
    t0 = time.time()
    cycles = 5
    total_new_tokens_generated = 0
    for i in range(cycles):
        generated = generate()
        total_new_tokens_generated += sum(new_tokens for _,_,new_tokens in generated)

    torch.cuda.synchronize()
    # note that we actually generate world_size unique streams (though the benchmark feeds the same inputs)
    total_new_tokens_generated *=  world_size
    througput = (time.time() - t0)/(total_new_tokens_generated)
    print_rank0(f"""
*** Performance stats:
Throughput per token including tokenize: {througput*1000:.2f} msecs
Start to ready to generate: {t_ready - t_start:.3f} secs
Tokenize and generate {total_new_tokens_generated} (bs={args.batch_size}) tokens: {t_generate_span:.3f} secs
Start to finish: {t_ready - t_start + t_generate_span:.3f} secs
""")
