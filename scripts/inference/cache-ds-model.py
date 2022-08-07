import argparse
import os

import deepspeed
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

import mii
from utils import get_torch_dtype, print_rank_n, run_rank_n, write_checkponts_json


def ParseArgs():
    parser = argparse.ArgumentParser(description="Text generation server")

    group = parser.add_argument_group(title="model")
    group.add_argument("--model_name", type=str,
                       required=True, help="model to use")
    group.add_argument("--dtype", type=str, required=True,
                       choices=["bf16", "fp16"], help="dtype for model")

    group = parser.add_argument_group(title="launch config")
    group.add_argument("--local_rank", required=False,
                       type=int, help="used by dist launchers")

    args = parser.parse_args()

    args.dtype = get_torch_dtype(args.dtype)

    return args


def main():
    deepspeed.init_distributed("nccl")
    args = ParseArgs()

    checkpoints_json = "checkpoints.json"

    if (os.path.isdir(os.getenv("DS_CACHE"))):
        print_rank_n("Found cached model at {}".format(os.getenv("DS_CACHE")))
        exit()

    # ensure all processes didn't find cache
    dist.barrier()

    print_rank_n("Caching model at {}".format(os.getenv("DS_CACHE")))

    run_rank_n(
        write_checkponts_json,
        {
            "model_name": args.model_name,
            "checkpoints_json": checkpoints_json
        }
    )

    run_rank_n(
        os.makedirs,
        {
            "name": os.getenv("DS_CACHE")
        }
    )

    with deepspeed.OnDevice(dtype=args.dtype, device="meta"):
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(args.model_name),
            torch_dtype=args.dtype
        )
    model.eval()

    deepspeed.init_inference(
        model,
        mp_size=int(os.getenv("WORLD_SIZE", "1")),
        dtype=args.dtype,
        checkpoint=checkpoints_json,
        replace_with_kernel_inject=True,
        save_mp_checkpoint_path=os.getenv("DS_CACHE")
    )


if (__name__ == "__main__"):
    main()
