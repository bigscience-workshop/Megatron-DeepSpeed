import argparse
import copy
import math
from typing import Any, List

import torch
import torch.distributed as dist


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


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="model")
    group.add_argument("--model_name", type=str,
                       required=True, help="model to use")
    group.add_argument("--dtype", type=str, required=True,
                       choices=["bf16", "fp16"], help="dtype for model")
    group.add_argument("--batch_size", default=1, type=int, help="batch size")
    group.add_argument(
        "--generate_kwargs",
        type=dict,
        default={
            "max_new_tokens": 100,
            "do_sample": False
        },
        help="generate parameters. look at https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate to see the supported parameters"
    )

    return parser


def get_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    args.dtype = get_torch_dtype(args.dtype)
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


def parse_generate_kwargs(kwargs: dict) -> dict:
    if ("max_length" in kwargs):
        kwargs["max_length"] = int(kwargs["max_length"])
    if ("min_length" in kwargs):
        kwargs["min_length"] = int(kwargs["min_length"])
    if ("do_sample" in kwargs):
        kwargs["do_sample"] = bool(kwargs["do_sample"])
    if ("early_stopping" in kwargs):
        kwargs["early_stopping"] = bool(kwargs["early_stopping"])
    if ("num_beams" in kwargs):
        kwargs["num_beams"] = int(kwargs["num_beams"])
    if ("temperature" in kwargs):
        kwargs["temperature"] = float(kwargs["temperature"])
    if ("top_k" in kwargs):
        kwargs["top_k"] = int(kwargs["top_k"])
    if ("top_p" in kwargs):
        kwargs["top_p"] = float(kwargs["top_p"])
    if ("typical_p" in kwargs):
        kwargs["typical_p"] = float(kwargs["typical_p"])
    if ("repitition_penalty" in kwargs):
        kwargs["repitition_penalty"] = float(kwargs["repitition_penalty"])
    if ("bos_token_id" in kwargs):
        kwargs["bos_token_id"] = int(kwargs["bos_token_id"])
    if ("pad_token_id" in kwargs):
        kwargs["pad_token_id"] = int(kwargs["pad_token_id"])
    if ("eos_token_id" in kwargs):
        kwargs["eos_token_id"] = int(kwargs["eos_token_id"])
    if ("length_penalty" in kwargs):
        kwargs["length_penalty"] = float(kwargs["length_penalty"])
    if ("no_repeat_ngram_size" in kwargs):
        kwargs["no_repeat_ngram_size"] = int(kwargs["no_repeat_ngram_size"])
    if ("encoder_no_repeat_ngram_size" in kwargs):
        kwargs["encoder_no_repeat_ngram_size"] = int(
            kwargs["encoder_no_repeat_ngram_size"])
    if ("num_return_sequences" in kwargs):
        kwargs["num_return_sequences"] = int(kwargs["num_return_sequences"])
    if ("max_time" in kwargs):
        kwargs["max_time"] = float(kwargs["max_time"])
    if ("max_new_tokens" in kwargs):
        kwargs["max_new_tokens"] = int(kwargs["max_new_tokens"])
    if ("decoder_start_token_id" in kwargs):
        kwargs["decoder_start_token_id"] = int(
            kwargs["decoder_start_token_id"])
    if ("num_beam_groups" in kwargs):
        kwargs["num_beam_groups"] = int(kwargs["num_beam_groups"])
    if ("diversity_penalty" in kwargs):
        kwargs["diversity_penalty"] = float(kwargs["diversity_penalty"])
    if ("forced_bos_token_id" in kwargs):
        kwargs["forced_bos_token_id"] = int(kwargs["forced_bos_token_id"])
    if ("forced_eos_token_id" in kwargs):
        kwargs["forced_eos_token_id"] = int(kwargs["forced_eos_token_id"])
    if ("exponential_decay_length_penalty" in kwargs):
        kwargs["exponential_decay_length_penalty"] = float(
            kwargs["exponential_decay_length_penalty"])

    # i was being lazy :)
    if ("bad_words_ids" in kwargs):
        del kwargs["bad_words_ids"]
    if ("force_words_ids" in kwargs):
        del kwargs["force_words_ids"]

    # so people don't slow down the server
    if ("use_cache" in kwargs):
        del kwargs["use_cache"]
    if ("remove_invalid_values" in kwargs):
        del kwargs["remove_invalid_values"]
    if ("synced_gpus" in kwargs):
        del kwargs["synced_gpus"]

    # no idea how to support this in a server setting
    if ("prefix_allowed_tokens_fn" in kwargs):
        del kwargs["prefix_allowed_tokens_fn"]
    if ("logits_processor" in kwargs):
        del kwargs["logits_processor"]
    if ("renormalize_logits" in kwargs):
        del kwargs["renormalize_logits"]
    if ("stopping_criteria" in kwargs):
        del kwargs["stopping_criteria"]
    if ("constraints" in kwargs):
        del kwargs["constraints"]
    if ("output_attentions" in kwargs):
        del kwargs["output_attentions"]
    if ("output_hidden_states" in kwargs):
        del kwargs["output_hidden_states"]
    if ("output_scores" in kwargs):
        del kwargs["output_scores"]
    if ("return_dict_in_generate" in kwargs):
        del kwargs["return_dict_in_generate"]

    return kwargs
