import argparse
import copy
import json
import math
import time
from functools import partial
from typing import Any, List, Tuple, Union

import torch
import torch.distributed as dist

from .constants import (
    BIGSCIENCE_BLOOM,
    DS_INFERENCE,
    DS_INFERENCE_BLOOM_FP16,
    DS_INFERENCE_BLOOM_INT8,
    DS_ZERO,
    HF_ACCELERATE,
    SCRIPT_FRAMEWORK_MODEL_DTYPE_ALLOWED
)


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


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="model")
    group.add_argument(
        "--deployment_framework",
        type=str,
        choices=[
            HF_ACCELERATE,
            DS_INFERENCE,
            DS_ZERO
        ],
        default=HF_ACCELERATE
    )
    group.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=[
            BIGSCIENCE_BLOOM,
            DS_INFERENCE_BLOOM_FP16,
            DS_INFERENCE_BLOOM_INT8
        ],
        help="model to use"
    )
    group.add_argument("--dtype", type=str, required=True,
                       choices=["bf16", "fp16", "int8"], help="dtype for model")
    group.add_argument(
        "--generate_kwargs",
        type=str,
        default='{"min_length": 100, "max_new_tokens": 100, "do_sample": false}',
        help="generate parameters. look at https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate to see the supported parameters"
    )

    return parser


def get_args(parser: argparse.ArgumentParser, script: str) -> argparse.Namespace:
    args = parser.parse_args()

    assert is_script_framework_model_dtype_allowed(
        script,
        args.deployment_framework,
        args.model_name,
        args.dtype
    ), f"{script} is not supported with {args.deployment_framework}, {args.model_name} and {args.dtype} dtype"

    args.dtype = get_torch_dtype(args.dtype)
    args.generate_kwargs = json.loads(args.generate_kwargs)
    args.use_pre_sharded_checkpoints = args.model_name in [
        DS_INFERENCE_BLOOM_FP16, DS_INFERENCE_BLOOM_INT8]
    return args


def run_rank_n(func: partial,
               barrier: bool = False,
               rank: int = 0,
               other_rank_output: Any = None) -> Any:
    if (dist.is_initialized()):
        if (dist.get_rank() == rank):
            output = func()
            if (barrier):
                dist.barrier()
            return output
        else:
            if (barrier):
                dist.barrier()
            return other_rank_output
    else:
        return func()


def print_rank_n(*values, rank: int = 0) -> None:
    if (dist.is_initialized()):
        if (dist.get_rank() == rank):
            print(*values)
    else:
        print(*values)


def get_dtype_from_model_name(model_name: str) -> str:
    if (model_name == BIGSCIENCE_BLOOM):
        return "bf16"
    elif (model_name == DS_INFERENCE_BLOOM_FP16):
        return "fp16"
    elif (model_name == DS_INFERENCE_BLOOM_INT8):
        return "int8"


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    if (dtype_str == "bf16"):
        return torch.bfloat16
    elif (dtype_str == "fp16"):
        return torch.float16
    elif (dtype_str == "int8"):
        return torch.int8


def get_str_dtype(dtype_str: str) -> torch.dtype:
    if (dtype_str == torch.bfloat16):
        return "bf16"
    elif (dtype_str == torch.float16):
        return "fp16"
    elif (dtype_str == torch.int8):
        return "int8"


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


def run_and_log_time(execs: Union[List[partial], partial]) -> Tuple[Union[List[Any], Any], float]:
    start_time = time.time()

    if (type(execs) == list):
        results = []
        for f in execs:
            results.append(f())
    else:
        results = execs()

    time_elapsed = time.time() - start_time
    return results, time_elapsed


def pad_ids(arrays, padding, max_length=-1):
    if (max_length < 0):
        max_length = max(list(map(len, arrays)))

    arrays = [[padding] * (max_length - len(array)) +
              array for array in arrays]

    return arrays


def is_script_framework_model_dtype_allowed(script: str,
                                            deployment_framework: str,
                                            model_name: str,
                                            dtype: str) -> bool:
    if (script in SCRIPT_FRAMEWORK_MODEL_DTYPE_ALLOWED):
        if (deployment_framework in SCRIPT_FRAMEWORK_MODEL_DTYPE_ALLOWED[script]):
            if (model_name in SCRIPT_FRAMEWORK_MODEL_DTYPE_ALLOWED[script][deployment_framework]):
                if (dtype in SCRIPT_FRAMEWORK_MODEL_DTYPE_ALLOWED[script][deployment_framework][model_name]):
                    return True
    return False
