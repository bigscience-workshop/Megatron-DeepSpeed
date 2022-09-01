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
from huggingface_hub import snapshot_download
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
from transformers.utils import is_offline_mode
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

# the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
tp_presharded_models = ["microsoft/bloom-deepspeed-inference-int8", "microsoft/bloom-deepspeed-inference-fp16"]

t_start = time.time()

num_tokens = 100

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--dtype", type=str, help="fp16 or int8", choices=["int8", "float16"], default="float16")
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
args = parser.parse_args()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

deepspeed.init_distributed('nccl')
rank = dist.get_rank()

def print_rank0(*msg):
    if rank != 0: return
    print(*msg)


### Model loading and instantiating on GPUs


def get_repo_root(model_name_or_path, revision=None):
    # checks if online or not
    if is_offline_mode():

        print_rank0("Offline mode: forcing local_files_only=True")
        local_files_only = True
    else:
        local_files_only = False

    # loads files from hub
    cached_repo_dir = snapshot_download(model_name_or_path, allow_patterns=["*"], local_files_only=local_files_only, revision=revision)

    return cached_repo_dir

def get_checkpoint_files(model_name_or_path, revision=None):
    # checks if online or not
    if is_offline_mode():
        print_rank0("Offline mode: forcing local_files_only=True")
        local_files_only = True
    else:
        local_files_only = False

    # loads files from hub
    cached_repo_dir = snapshot_download(model_name_or_path, allow_patterns=["*"], local_files_only=local_files_only, revision=revision)

    # extensions: .bin | .pt
    # creates a list of paths from all downloaded files in cache dir
    file_list = [str(entry) for entry in Path(cached_repo_dir).rglob('*.[bp][it][n]') if entry.is_file()]
    return file_list


model_name = args.name
infer_dtype = args.dtype

tp_presharded_mode = True if model_name in tp_presharded_models else False

#print(get_checkpoint_files(model_name))

print_rank0("*** Loading the model {model_name}")

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
def write_checkponts_json():

    with io.open(checkpoints_json, 'w', encoding='utf-8') as f:

        #checkpoint_dir = "/gpfsscratch/rech/six/commun/uan68tv-model-conversion/bloom"
        #checkpoint_files = glob.glob(f"{checkpoint_dir}/*bin")
        checkpoint_files = get_checkpoint_files(model_name)

        #print("Checkpoint files:", checkpoint_files)

        data = {
            "type": "BLOOM",
            "checkpoints": checkpoint_files,
            "version": 1.0
        }
        if world_size > 1:
            data["parallelization"] = "tp"

        json.dump(data, f)




if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('pre-ds-inference-init', force=True)

if kernel_inject:
    kwargs = dict(replace_with_kernel_inject=True)
else:
    kwargs = dict(injection_policy={BloomBlock: ('self_attention.dense', 'mlp.dense_4h_to_h')})

repo_root = get_repo_root(model_name)
if tp_presharded_mode:
    # tp presharded repos come with their own checkpoints config file
    checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
else:
     # for normal bloom repo we need to write the checkpoints config file
    if rank == 0:
        write_checkponts_json()
    dist.barrier()

#checkpoints_json=None
model = deepspeed.init_inference(model,
                                 mp_size=world_size,
                                 base_dir=repo_root,
                                 dtype=getattr(torch, infer_dtype),
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


# warmup is a must if measuring speed as it's when all the optimizations are performed
# e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
_ = generate()

t_generate_start = time.time()
generated = generate()
t_generate_span = time.time() - t_generate_start
for i,o,_ in generated:
    print_rank0(f"{'-'*60}\nin={i}\nout={o}\n")

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('end-of-run', force=True)

### Benchmark

# benchmark it!
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
    througput = (time.time() - t0)/(total_new_tokens_generated)
    print_rank0(f"""
*** Performance stats:
Throughput per token including tokenize: {througput*1000:.2f} msecs
Start to ready to generate: {t_ready - t_start:.3f} secs
Tokenize and generate {total_new_tokens_generated} (bs={args.batch_size}) tokens: {t_generate_span:.3f} secs
Start to finish: {t_ready - t_start + t_generate_span:.3f} secs
""")
