import argparse
import copy
import json
import math
from typing import Any, List

import torch
import torch.distributed as dist

from .requests import GenerateRequest


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


class Execute:
    def __init__(self, func: callable, kwargs: dict) -> None:
        self.func = func
        self.kwargs = kwargs

    def __call__(self) -> Any:
        return self.func(**self.kwargs)


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="model")
    group.add_argument("--model_name", type=str,
                       required=True, help="model to use")
    group.add_argument("--dtype", type=str, required=True,
                       choices=["bf16", "fp16"], help="dtype for model")
    group.add_argument(
        "--generate_kwargs",
        type=str,
        default='{"min_length": 100, "max_new_tokens": 100, "do_sample": False}',
        help="generate parameters. look at https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate to see the supported parameters"
    )

    return parser


def get_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    args.dtype = get_torch_dtype(args.dtype)
    args.generate_kwargs = json.loads(args.generate_kwargs)
    return args


def run_rank_n(func: callable,
               kwargs: dict,
               barrier: bool = False,
               rank: int = 0,
               other_rank_output: Any = None) -> Any:
    if (dist.is_initialized()):
        if (dist.get_rank() == rank):
            output = func(**kwargs)
            if (barrier):
                dist.barrier()
            return output
        else:
            if (barrier):
                dist.barrier()
            return other_rank_output
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


def get_str_dtype(dtype_str: str) -> torch.dtype:
    if (dtype_str == torch.bfloat16):
        return "bf16"
    elif (dtype_str == torch.float16):
        return "fp16"


def get_dummy_batch(batch_size: int, input_sentences: List[str] = None) -> List[str]:
    if (input_sentences == None):
        input_sentences = copy.deepcopy(dummy_input_sentences)

    if (batch_size > len(input_sentences)):
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    input_sentences = input_sentences[:batch_size]

    return input_sentences


def get_num_tokens_to_generate(max_new_tokens: int,
                               allowed_max_new_tokens: int) -> int:
    if (max_new_tokens == None):
        return allowed_max_new_tokens
    else:
        return min(max_new_tokens, allowed_max_new_tokens)
