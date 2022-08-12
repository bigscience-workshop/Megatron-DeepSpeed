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
    group.add_argument("--generate_kwargs", type=dict, default={},
                       help="generate parameters. look at https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate to see the supported parameters")

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
                         generate_kwargs,
                         cycles: int = 5):
    total_new_tokens_generated = 0
    for _ in range(cycles):
        _, num_generated_tokens = model.generate(
            input_sentences,
            generate_kwargs
        )
        total_new_tokens_generated += sum(
            new_tokens for new_tokens in num_generated_tokens)
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
