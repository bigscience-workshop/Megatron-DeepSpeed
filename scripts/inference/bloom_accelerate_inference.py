import gc
from typing import List, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import utils
from utils import (
    Execute,
    benchmark_generation,
    get_argument_parser,
    get_benchmark_results,
    get_dummy_batch,
    print_rank_n,
    run_and_log_time,
)


class HFAccelerateModel:
    def __init__(self, model_name: str, dtype: torch.dtype) -> None:
        print_rank_n("Loading model...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory=get_max_memory_per_gpu_dict(
                dtype, model_name),
            torch_dtype=dtype
        )

        self.model.eval()
        self.input_device = "cuda:0"

        print_rank_n("Model loaded")

    def generate(self,
                 text: Union[str, List[str]],
                 generate_kwargs: dict,
                 remove_input_from_output: bool = False) -> Union[str, List[str]]:
        if (type(text) == str):
            text = [text]

        input_tokens = self.tokenizer(text, return_tensors="pt", padding=True)

        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.input_device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **input_tokens,
                **generate_kwargs
            )

        input_token_lengths = [x.shape[0] for x in input_tokens.input_ids]
        output_token_lengths = [x.shape[0] for x in output_tokens]
        generated_tokens = [
            o - i for i, o in zip(input_token_lengths, output_token_lengths)]

        if (remove_input_from_output):
            output_tokens = [x[-i:]
                             for x, i in zip(output_tokens, generated_tokens)]

        output_text = self.tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True)

        return output_text, generated_tokens


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
        print_rank_n(
            f"The model {model_name} has a broken config file. Please notify the owner")
        raise

    bytes = torch.finfo(dtype).bits / 8
    param_memory_total_in_bytes = model_params * bytes
    # add 5% since weight sizes aren't the same and some GPU may need more memory
    param_memory_per_gpu_in_bytes = int(
        param_memory_total_in_bytes / n_gpus * 1.05)
    print_rank_n(
        f"Estimating {param_memory_per_gpu_in_bytes/2**30:0.2f}GB per gpu for weights")

    # check the real available memory
    # load cuda kernels first and only measure the real free memory after loading (shorter by ~2GB)
    torch.ones(1).cuda()
    max_memory_per_gpu_in_bytes = torch.cuda.mem_get_info(0)[0]
    if max_memory_per_gpu_in_bytes < param_memory_per_gpu_in_bytes:
        raise ValueError(
            f"Unable to generate the memory map automatically as the needed estimated memory per gpu ({param_memory_per_gpu_in_bytes/2**30:0.2f}GB) is bigger than the available per gpu memory ({max_memory_per_gpu_in_bytes/2**30:0.2f}GB)")

    return {i: param_memory_per_gpu_in_bytes for i in range(torch.cuda.device_count())}


def main():
    args = get_args()

    model, initialization_time = run_and_log_time(
        Execute(
            HFAccelerateModel,
            {
                "model_name": args.model_name,
                "dtype": args.dtype,
            }
        )
    )

    if (args.generate_kwargs):
        generate_kwargs = args.generate_kwargs
    else:
        generate_kwargs = {
            "max_new_tokens": 100,
            "do_sample": False
        }

    print_rank_n(
        f"*** Starting to generate {generate_kwargs['max_new_tokens']} tokens with bs={args.batch_size}")

    input_sentences = get_dummy_batch(args.batch_size)

    print_rank_n(f"Generate args {generate_kwargs}")

    # warmup is a must if measuring speed as it's when all the optimizations are performed
    # e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
    model.generate(
        input_sentences,
        generate_kwargs
    )

    (output_text, num_generated_tokens), generation_time = run_and_log_time(
        Execute(
            model.generate,
            {
                "text": input_sentences,
                "generate_kwargs": generate_kwargs
            }
        )
    )
    for i, (o, _) in zip(input_sentences, zip(output_text, num_generated_tokens)):
        print_rank_n(f"{'-' * 60}\nin = {i}\nout = {o}\n")

    if (args.benchmark_cycles > 0):
        print_rank_n(f"*** Running benchmark")

        torch.cuda.empty_cache()
        gc.collect()

        # warm up
        model.generate(input_sentences, generate_kwargs)
        torch.cuda.synchronize()

        # benchmark
        total_new_tokens_generated, benchmark_time = run_and_log_time(
            Execute(
                benchmark_generation,
                {
                    "input_sentences": input_sentences,
                    "model": model,
                    "generate_kwargs": generate_kwargs
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
