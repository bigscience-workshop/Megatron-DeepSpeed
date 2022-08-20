import argparse
import os
import shutil

import deepspeed
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from utils import print_rank_n, run_rank_n

from .model import write_checkponts_json


def cache_ds_checkpoints(args: argparse.Namespace) -> None:
    if (args.local_rank == 0):
        print_rank_n("Loading model...")
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    # Load model
    with deepspeed.OnDevice(dtype=args.dtype, device="meta"):
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(args.model_name),
            torch_dtype=torch.bfloat16
        )
    model = model.eval()

    # Write checkpoints.json
    tmp_directory = "tmp"
    run_rank_n(
        os.makedirs,
        {
            "name": tmp_directory,
            "exist_ok": True
        }
    )
    checkpoints_json = os.path.join(tmp_directory, "checkpoints.json")
    run_rank_n(
        write_checkponts_json,
        {
            "checkpoints_json": checkpoints_json,
            "model_name": args.model_name
        },
        barrier=True
    )

    run_rank_n(
        os.makedirs,
        {
            "name": args.save_mp_checkpoint_path,
            "exist_ok": True
        },
        barrier=True
    )

    if (args.dtype == torch.float16):
        model = deepspeed.init_inference(
            model,
            mp_size=world_size,
            dtype=args.dtype,
            checkpoint=checkpoints_json,
            replace_with_kernel_inject=True,
            save_mp_checkpoint_path=args.save_mp_checkpoint_path
        )
    elif (args.dtype == torch.bfloat16):
        raise NotImplementedError("bfloat16 is not yet supported")

    run_rank_n(shutil.rmtree, {"path": tmp_directory})
    print_rank_n("Model loaded")
