import argparse
import copy
import math
import time
from typing import Any, List, Tuple, Union

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer


dummy_input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way"
]


class MaxTokensError(Exception):
    def __init__(self, max_new_tokens: int, allowed_max_new_tokens: int) -> None:
        super().__init__("max_new_tokens = {} > {} is not supported.".format(
            max_new_tokens, allowed_max_new_tokens))


class Execute:
    def __init__(self, func: callable, kwargs: dict) -> None:
        self.func = func
        self.kwargs = kwargs

    def __call__(self) -> Any:
        return self.func(**self.kwargs)


def get_argument_parser():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="model")
    group.add_argument("--model_name", type=str,
                       required=True, help="model to use")
    group.add_argument("--dtype", type=str, required=True,
                       choices=["bf16", "fp16"], help="dtype for model")
    group.add_argument("--batch_size", default=1, type=int, help="batch size")

    group = parser.add_argument_group(title="default values")
    group.add_argument("--greedy", action="store_true")
    group.add_argument("--top_k", type=int, default=0, help="default top_k")
    group.add_argument("--top_p", type=float, default=0, help="default top_p")
    group.add_argument("--temperature", type=float,
                       default=1, help="default temperature")
    group.add_argument("--min_length", type=int, default=1, help="min length")
    group.add_argument("--max_new_tokens", type=int,
                       default=100, help="max new tokens")

    return parser


def get_args(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    args.dtype = get_torch_dtype(args.dtype)
    return args


def run_rank_n(func: callable,
               kwargs: dict,
               barrier: bool = False,
               rank: int = 0) -> Any:
    if (dist.is_initialized()):
        if (dist.get_rank() == rank):
            output = func(**kwargs)
            if (barrier):
                dist.barrier()
            return output
        else:
            if (barrier):
                dist.barrier()
    else:
        return func(**kwargs)


def print_rank_n(*values, rank: int = 0) -> None:
    if (dist.is_initialized()):
        if (dist.get_rank() == rank):
            print(*values)
    else:
        print(*values)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    if (dtype_str == "bf16"):
        return torch.bfloat16
    elif (dtype_str == "fp16"):
        return torch.float16


def get_dummy_batch(batch_size: int, input_sentences: List[str] = None) -> List[str]:
    if (input_sentences == None):
        input_sentences = copy.deepcopy(dummy_input_sentences)

    if (batch_size > len(input_sentences)):
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    input_sentences = input_sentences[:batch_size]

    return input_sentences


def generate(inputs: List[str],
             model: AutoModelForCausalLM,
             tokenizer: AutoTokenizer,
             generate_kwargs: dict,
             input_device) -> Tuple[List[str], List[int]]:
    """ returns a list of zipped outputs and number of new tokens """

    input_tokens = tokenizer(
        inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(input_device)

    outputs = model.generate(**input_tokens, **generate_kwargs)

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o-i for i,
                        o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(outputs, total_new_tokens)


def run_and_log_time(execs: Union[List[Execute], Execute]) -> Union[List[Any], float]:
    """
    runs a list of Execute objects and returns a list of outputs and the time taken
    """
    start_time = time.time()

    if (type(execs) == list):
        results = []
        for e in execs:
            results.append(e())
    else:
        results = execs()

    time_elapsed = time.time() - start_time
    return results, time_elapsed


def benchmark_generation(input_sentences,
                         model,
                         tokenizer,
                         generate_kwargs,
                         input_device,
                         cycles: int = 5):
    total_new_tokens_generated = 0
    for _ in range(cycles):
        generated = generate(
            input_sentences,
            model,
            tokenizer,
            generate_kwargs,
            input_device
        )
        total_new_tokens_generated += sum(new_tokens for _,
                                          new_tokens in generated)
    return total_new_tokens_generated


def get_benchmark_results(benchmark_time: float,
                          initialization_time: float,
                          generation_time: float,
                          total_new_tokens_generated: int,
                          batch_size: int) -> str:
    throughput = total_new_tokens_generated / benchmark_time
    return f"""
*** Performance stats:
Throughput (including tokenization) = {throughput:.2f} tokens/sec
Throughput (including tokenization) = {1000 / throughput:.2f} msecs/token
Model loading time = {initialization_time:.2f} secs
Total tokens generated = {total_new_tokens_generated} with batch size = {batch_size}
Generation time per batch = {generation_time:.2f} secs
Model loading time + generation time per batch = {initialization_time + generation_time:.2f} secs
"""
