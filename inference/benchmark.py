import argparse
import gc
import os
import time
from typing import Any, List, Union

import deepspeed
import torch

import constants
import utils
from ds_inference import DSInferenceModel
from ds_zero import DSZeROModel
from hf_accelerate import HFAccelerateModel
from utils import Execute, Model, get_argument_parser, get_dummy_batch, print_rank_n


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


def benchmark_end_to_end(args: argparse.Namespace,
                         model_class: Model,
                         zero_activated: bool = False) -> None:
    model, initialization_time = run_and_log_time(
        Execute(model_class, {"args": args})
    )

    print_rank_n(
        f"*** Starting to generate {args.generate_kwargs['max_new_tokens']} tokens with bs={args.batch_size}")

    input_sentences = get_dummy_batch(args.batch_size)

    print_rank_n(f"Generate args {args.generate_kwargs}")

    # warmup is a must if measuring speed as it's when all the optimizations are performed
    # e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
    model.generate(
        input_sentences,
        args.generate_kwargs
    )

    (output_text, num_generated_tokens), generation_time = run_and_log_time(
        Execute(
            model.generate,
            {
                "text": input_sentences,
                "generate_kwargs": args.generate_kwargs
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
        model.generate(input_sentences, args.generate_kwargs)
        torch.cuda.synchronize()

        # benchmark
        total_new_tokens_generated, benchmark_time = run_and_log_time(
            Execute(
                benchmark_generation,
                {
                    "input_sentences": input_sentences,
                    "model": model,
                    "generate_kwargs": args.generate_kwargs,
                    "cycles": args.benchmark_cycles
                }
            )
        )

        # with ZeRO every GPU is generating batch_size * sequence_length tokens
        if (zero_activated):
            world_size = int(os.getenv('WORLD_SIZE', '1'))
            total_new_tokens_generated *= world_size

        print_rank_n(
            get_benchmark_results(
                benchmark_time,
                initialization_time,
                generation_time,
                total_new_tokens_generated,
                args.batch_size
            )
        )


def get_args() -> argparse.Namespace:
    parser = get_argument_parser()

    group = parser.add_argument_group(title="launch config")
    group.add_argument(
        "--deployment_framework",
        type=str,
        choices=[
            constants.HF_ACCELERATE,
            constants.DS_INFERENCE,
            constants.DS_ZERO
        ],
        default=constants.HF_ACCELERATE
    )
    group.add_argument("--benchmark_cycles", type=int,
                       default=0, help="additionally run benchmark")
    group.add_argument("--local_rank", required=False,
                       type=int, help="used by dist launchers")
    group.add_argument("--save_mp_checkpoint_path", required=False,
                       type=str, help="MP checkpoints path for DS inference")
    group.add_argument("--cpu_offload", action="store_true",
                       help="whether to activate CPU offload for DS ZeRO")

    args = utils.get_args(parser)

    launched_with_deepspeed = args.deployment_framework in [
        constants.DS_INFERENCE, constants.DS_ZERO]

    if (not launched_with_deepspeed):
        assert args.local_rank == None, "local_rank must be None if not launched with DeepSpeed"

    if (args.save_mp_checkpoint_path):
        assert args.deployment_framework == constants.DS_INFERENCE, "save_mp_checkpoint_path only works with DS inference"

    if (args.cpu_offload):
        assert args.deployment_framework == constants.DS_ZERO, "cpu_offload only works with DS_ZeRO"

    return args


def main() -> None:
    args = get_args()

    if (args.deployment_framework == constants.HF_ACCELERATE):
        benchmark_end_to_end(args, HFAccelerateModel)
    elif (args.deployment_framework == constants.DS_INFERENCE):
        deepspeed.init_distributed("nccl")
        benchmark_end_to_end(args, DSInferenceModel)
    elif (args.deployment_framework == constants.DS_ZERO):
        benchmark_end_to_end(args, DSZeROModel, zero_activated=True)
    else:
        raise ValueError(
            f"Unknown deployment framework {args.deployment_framework}")


if (__name__ == "__main__"):
    main()
