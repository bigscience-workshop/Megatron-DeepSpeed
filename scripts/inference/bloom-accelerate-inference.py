import argparse
import time
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def get_max_memory_per_gpu_dict():
    max_memory_per_gpu =  torch.cuda.get_device_properties(0).total_memory // 2**30
    return {i: f"{max_memory_per_gpu}GIB" for i in range(torch.cuda.device_count())}

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

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory=get_max_memory_per_gpu_dict(),
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

generate_kwargs = dict(min_length=num_tokens, max_length=num_tokens, do_sample=False)
#generate_kwargs = dict(min_length=num_tokens, max_length=num_tokens, do_sample=True)

#top_k=None if greedy else top_k,
#top_p=None if greedy else top_p

if rank == 0:
    print(f"Generate args {generate_kwargs}")
inputs = input_sentences[:args.batch_size]
def generate():
    """ returns a list of pairs of inputs and outputs """

    tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in tokens:
        if torch.is_tensor(tokens[t]):
            tokens[t] = tokens[t].to("cuda:0")

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
        tokens_in_cycle = num_tokens * args.batch_size
        througput = (time.time() - t0)/(cycles * tokens_in_cycle)
        print(f"""
*** Performance stats:
Throughput per token including tokenize: {througput*1000:.2f} msecs
Start to ready to generate: {t_ready - t_start:.3f} secs
Tokenize and generate {tokens_in_cycle} (bs={args.batch_size}) tokens: {t_generate_span:.3f} secs
Start to finish: {t_ready - t_start + t_generate_span:.3f} secs
""")
