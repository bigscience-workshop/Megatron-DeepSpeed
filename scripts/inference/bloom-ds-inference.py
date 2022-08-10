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
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
import deepspeed
import gc
import math
import os
import time
import torch
import torch.distributed as dist
from utils import generate_, write_checkponts_json


t_start = time.time()

num_tokens = 100

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
args = parser.parse_args()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

deepspeed.init_distributed('nccl')
rank = dist.get_rank()


### Model loading and instantiating on GPUs

model_name = args.name

#print(get_checkpoint_files(model_name))

if rank == 0:
    print(f"*** Loading the model {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# XXX: can't automatically derive dtype via config's `from_pretrained`
#dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16


# use one of these args to `init_inference`
# 1. injection_policy is the slower version, but it's plain pytorch so it'll always work
# 2. replace_with_kernel_inject is the faster one (fast fused kernels)
kernel_inject = True
#kernel_inject = False

if kernel_inject:
    # XXX: for now ds-inference only works with fp16
    dtype = torch.float16
else:
    dtype = torch.bfloat16

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('pre-from-pretrained', force=True)

# Construct model with fake meta tensors, later will be replaced during ds-inference ckpt load
with deepspeed.OnDevice(dtype=dtype, device='meta'):
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

if args.benchmark:
    deepspeed.runtime.utils.see_memory_usage('post-from-pretrained', force=True)

model = model.eval()


if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('post-init-ds-zero-init', force=True)

### Deepspeed-Inference Loading

checkpoints_json = "checkpoints.json"

if rank == 0:
    write_checkponts_json(model_name, checkpoints_json)
dist.barrier()

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('pre-ds-inference-init', force=True)

if kernel_inject:
    kwargs = dict(replace_with_kernel_inject=True)
else:
    kwargs = dict(injection_policy={BloomBlock: ('self_attention.dense', 'mlp.dense_4h_to_h')})

#checkpoints_json=None
model = deepspeed.init_inference(model,
                                 mp_size=world_size,
                                 dtype=torch.half,
                                 checkpoint=checkpoints_json,
                                 **kwargs,
                                 )

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('post-ds-inference-init', force=True)


model = model.module

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

generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)

if rank == 0:
    print(f"Generate args {generate_kwargs}")
inputs = input_sentences[:args.batch_size]


# warmup is a must if measuring speed as it's when all the optimizations are performed
# e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
_ = generate_(
    inputs,
    model,
    tokenizer,
    generate_kwargs,
    torch.cuda.current_device()
)

t_generate_start = time.time()
generated = generate_(
    inputs,
    model,
    tokenizer,
    generate_kwargs,
    torch.cuda.current_device()
)
t_generate_span = time.time() - t_generate_start
if rank == 0:
    for i,o,_ in generated:
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
        _ = generate_(
            inputs,
            model,
            tokenizer,
            generate_kwargs,
            torch.cuda.current_device()
        )
    torch.cuda.synchronize()

    # benchmark
    t0 = time.time()
    cycles = 5
    total_new_tokens_generated = 0
    for i in range(cycles):
        generated = generate_(
            inputs,
            model,
            tokenizer,
            generate_kwargs,
            torch.cuda.current_device()
        )
        total_new_tokens_generated += sum(new_tokens for _,_,new_tokens in generated)
    torch.cuda.synchronize()
    if rank == 0:
        througput = (time.time() - t0)/(total_new_tokens_generated)
        print(f"""
*** Performance stats:
Throughput per token including tokenize: {througput*1000:.2f} msecs
Start to ready to generate: {t_ready - t_start:.3f} secs
Tokenize and generate {total_new_tokens_generated} (bs={args.batch_size}) tokens: {t_generate_span:.3f} secs
Start to finish: {t_ready - t_start + t_generate_span:.3f} secs
""")
