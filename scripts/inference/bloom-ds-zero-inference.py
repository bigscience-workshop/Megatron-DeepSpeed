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


import glob
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
import deepspeed
import io
import math
import sys
import json
import os
import gc
import torch
import torch.distributed as dist
import time

t_start = time.time()

num_tokens = 100

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
parser.add_argument("--cpu_offload", action="store_true", help="whether to activate CPU offload")
args = parser.parse_args()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))


### Model loading and instantiating on GPU (via ZeRO)

model_name = args.name

if local_rank == 0:
    print(f"*** Loading the model {model_name}")

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

if args.cpu_offload:
    ds_config["zero_optimization"]["offload_param"] = dict(device="cpu", pin_memory=True)

dschf = HfDeepSpeedConfig(ds_config) # this tells from_pretrained to instantiate directly on gpus

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('pre-from-pretrained', force=True)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

if args.benchmark:
    deepspeed.runtime.utils.see_memory_usage('post-from-pretrained', force=True)

model = model.eval()

rank = dist.get_rank()

if rank == 0:
    print(ds_config)

ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()
model = ds_engine.module

if args.benchmark:
    t_ready = time.time()


### Generate

if rank == 0:
    print(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

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

generate_kwargs = dict(min_length=num_tokens, max_length=num_tokens, do_sample=False)
#generate_kwargs = dict(min_length=num_tokens, max_length=num_tokens, do_sample=True)
if rank == 0:
    print(f"Generate args {generate_kwargs}")
inputs = input_sentences[:args.batch_size]
def generate():
    """ returns a list of pairs of inputs and outputs """

    tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)

    for t in tokens:
        if torch.is_tensor(tokens[t]):
            tokens[t] = tokens[t].to(torch.cuda.current_device())

    greedy_output = model.generate(**tokens, **generate_kwargs)

    outputs = tokenizer.batch_decode(greedy_output, skip_special_tokens=True)

    return zip(inputs, outputs)

# XXX: this is currently doing world_size streams on world_size gpus, so we can feed it different inputs on each! and hence the time can be divided by world_size

# warmup is a must if measuring speed as it's when all the optimizations are performed
# e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
_ = generate()

t_generate_start = time.time()
pairs = generate()
t_generate_span = time.time() - t_generate_start
if rank == 0:
    for i,o in pairs:
        print(f"{'-'*60}\nin={i}\nout={o}\n")


if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('end-of-run', force=True)

### Benchmark

# benchmark it!
if args.benchmark:
    if rank == 0:
        print(f"*** Running benchmark")

    # warm up
    for i in range(1):
        _ = generate()
    torch.cuda.synchronize()

    # benchmark
    t0 = time.time()
    cycles = 5
    for i in range(cycles):
        _ = generate()
    torch.cuda.synchronize()
    if rank == 0:
        # note that dividing by world_size as well as we can have world_size streams
        tokens_in_cycle_total = num_tokens*args.batch_size*world_size
        througput = (time.time() - t0)/(cycles*tokens_in_cycle_total)
        print(f"""
*** Performance stats:
Throughput per token including tokenize: {througput*1000:.2f} msecs
Start to ready to generate: {t_ready - t_start:.3f} secs
Tokenize and generate {tokens_in_cycle_total} (bs={args.batch_size}) tokens: {t_generate_span:.3f} secs
Start to finish: {t_ready - t_start + t_generate_span:.3f} secs
""")
