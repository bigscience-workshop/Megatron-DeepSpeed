import gc

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import utils
from utils import (
    Execute,
    benchmark_generation,
    generate,
    get_argument_parser,
    get_benchmark_results,
    get_dummy_batch,
    print_rank_n,
    run_and_log_time,
)


def get_args():
    parser = get_argument_parser()

    group = parser.add_argument_group(title="launch config")
    group.add_argument("--benchmark_cycles", type=int,
                       default=0, help="additionally run benchmark")

    args = utils.get_args(parser)

    return args


def get_max_memory_per_gpu_dict(dtype, model_name):
    """ try to generate the memory map based on what we know about the model and the available hardware """

    # figure out the memory map - the minimum per gpu required to load the model
    n_gpus = torch.cuda.device_count()

    if model_name == "bigscience/bloom" and n_gpus == 8 and torch.cuda.get_device_properties(0).total_memory > 79*2**30:
        # hand crafted optimized memory map for 8x80 setup over BLOOM
        # this works with bs=40
        return {0: '0GIB', 1: '51GIB', 2: '51GIB', 3: '51GIB', 4: '51GIB', 5: '51GIB', 6: '51GIB', 7: '51GIB'}

    try:
        # model_params calculation, as we don't have a model yet to do:
        #model_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

        config = AutoConfig.from_pretrained(model_name)
        h = config.n_embed
        l = config.n_layer
        v = config.vocab_size
        # from https://github.com/bigscience-workshop/bigscience/tree/6917a3b5fefcf439d3485ca184b4d9f6ab605150/math#model-sizing
        model_params = l*(12*h**2 + 13*h) + v*h + 4*h
    except:
        print(f"The model {model_name} has a broken config file. Please notify the owner")
        raise

    bytes = torch.finfo(dtype).bits / 8
    param_memory_total_in_bytes = model_params * bytes
    # add 5% since weight sizes aren't the same and some GPU may need more memory
    param_memory_per_gpu_in_bytes = int(param_memory_total_in_bytes / n_gpus * 1.05)
    print(f"Estimating {param_memory_per_gpu_in_bytes/2**30:0.2f}GB per gpu for weights")

    # check the real available memory
    # load cuda kernels first and only measure the real free memory after loading (shorter by ~2GB)
    torch.ones(1).cuda()
    max_memory_per_gpu_in_bytes = torch.cuda.mem_get_info(0)[0]
    if max_memory_per_gpu_in_bytes < param_memory_per_gpu_in_bytes:
        raise ValueError(f"Unable to generate the memory map automatically as the needed estimated memory per gpu ({param_memory_per_gpu_in_bytes/2**30:0.2f}GB) is bigger than the available per gpu memory ({max_memory_per_gpu_in_bytes/2**30:0.2f}GB)")

    return {i: param_memory_per_gpu_in_bytes for i in range(torch.cuda.device_count())}


def main():
    args = get_args()
    print_rank_n(f"Loading model {args.model_name}")

    (tokenizer, model), initialization_time = run_and_log_time(
        [
            Execute(
                AutoTokenizer.from_pretrained,
                {
                    "pretrained_model_name_or_path": args.model_name,
                }
            ),
            Execute(
                AutoModelForCausalLM.from_pretrained,
                {
                    "pretrained_model_name_or_path": args.model_name,
                    "device_map": "auto",
                    "max_memory": get_max_memory_per_gpu_dict(args.dtype, args.model_name),
                    "torch_dtype": args.dtype
                }
            )
        ]
    )

    print_rank_n(
        f"*** Starting to generate {args.max_new_tokens} tokens with bs={args.batch_size}")

    input_sentences = get_dummy_batch(args.batch_size)
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    print_rank_n(f"Generate args {generate_kwargs}")

    # warmup is a must if measuring speed as it's when all the optimizations are performed
    # e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
    _ = generate(
        input_sentences,
        model,
        tokenizer,
        generate_kwargs,
        "cuda:0"
    )

    generated, generation_time = run_and_log_time(
        Execute(
            generate,
            {
                "inputs": input_sentences,
                "model": model,
                "tokenizer": tokenizer,
                "generate_kwargs": generate_kwargs,
                "input_device": "cuda:0"
            }
        )
    )
    for i, (o, _) in zip(input_sentences, generated):
        print_rank_n(f"{'-' * 60}\nin = {i}\nout = {o}\n")

    if (args.benchmark_cycles > 0):
        print_rank_n(f"*** Running benchmark")

        torch.cuda.empty_cache()
        gc.collect()

        # warm up
        _ = generate(
            input_sentences,
            model,
            tokenizer,
            generate_kwargs,
            "cuda:0"
        )
        torch.cuda.synchronize()

        # benchmark
        total_new_tokens_generated, benchmark_time = run_and_log_time(
            Execute(
                benchmark_generation,
                {
                    "input_sentences": input_sentences,
                    "model": model,
                    "tokenizer": tokenizer,
                    "generate_kwargs": generate_kwargs,
                    "input_device": "cuda:0",
                }
            )
        )
        print_rank_n(
            get_benchmark_results(
                benchmark_time,
                initialization_time,
                generation_time,
                total_new_tokens_generated,
                args.batch_size
            )
        )

if (__name__ == "__main__"):
    main()
