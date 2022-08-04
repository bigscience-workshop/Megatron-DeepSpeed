import argparse
import time
import os
import gc
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import generate_, get_max_memory_per_gpu_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    parser.add_argument("--name", type=str, help="Name path", required=True)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.)

    return parser.parse_args()

t_start = time.time()

num_tokens = 100

args = get_args()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

rank = local_rank

model_name = args.name
if rank == 0:
    print(f"Loading model {model_name}")


tokenizer = AutoTokenizer.from_pretrained(model_name)

# XXX: can't automatically derive dtype via config's `from_pretrained`
dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16

#print(get_max_memory_per_gpu_dict())


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory=get_max_memory_per_gpu_dict(dtype, model_name),
    torch_dtype=dtype,
)


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
#generate_kwargs = dict(max_new_tokens=num_tokens, use_cache=False, do_sample=False)
#generate_kwargs = dict(min_length=num_tokens, max_length=num_tokens, do_sample=False) 

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
    "cuda:0"
)

t_generate_start = time.time()
generated = generate_(
    inputs,
    model,
    tokenizer,
    generate_kwargs,
    "cuda:0"
)
t_generate_span = time.time() - t_generate_start
if rank == 0:
    for i,o,_ in generated:
        print(f"{'-'*60}\nin={i}\nout={o}\n")


if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()

### Benchmark

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
            "cuda:0"
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
            "cuda:0"
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
