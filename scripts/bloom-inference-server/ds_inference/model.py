import io
import json
import os
import shutil
from argparse import Namespace
from pathlib import Path

import deepspeed
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_offline_mode

from huggingface_hub import snapshot_download
from utils import Model, print_rank_n, run_rank_n


class DSInferenceModel(Model):
    def __init__(self, args: Namespace) -> None:
        if (args.local_rank == 0):
            # print_rank_n won't work here since deepspeed is not initialized yet
            print_rank_n("Loading model...")
        world_size = int(os.getenv("WORLD_SIZE", "1"))

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.pad = self.tokenizer.pad_token_id

        # Load model
        with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
            self.model = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(args.model_name),
                torch_dtype=torch.bfloat16
            )
        self.model = self.model.eval()

        if (args.dtype in [torch.float16, torch.int8]):
            if (args.use_pre_sharded_checkpoints):
                model_path = snapshot_download(
                    args.model_name,
                    allow_patterns=["*"],
                    local_files_only=is_offline_mode(),
                    revision=None
                )
                checkpoints_json = os.path.join(
                    model_path, "BLOOM_ds-inference_config.json")

                self.model = deepspeed.init_inference(
                    self.model,
                    mp_size=world_size,
                    base_dir=model_path,
                    dtype=args.dtype,
                    checkpoint=checkpoints_json,
                    replace_with_kernel_inject=True
                )
            else:
                # Write checkpoints.json
                tmp_directory = "tmp"
                run_rank_n(
                    os.makedirs,
                    {
                        "name": tmp_directory,
                        "exist_ok": True
                    }
                )
                checkpoints_json = os.path.join(
                    tmp_directory, "checkpoints.json")
                run_rank_n(
                    write_checkponts_json,
                    {
                        "checkpoints_json": checkpoints_json,
                        "model_name": args.model_name
                    },
                    barrier=True
                )

                self.model = deepspeed.init_inference(
                    self.model,
                    mp_size=world_size,
                    dtype=args.dtype,
                    checkpoint=checkpoints_json,
                    replace_with_kernel_inject=True
                )

                run_rank_n(shutil.rmtree, {"path": tmp_directory})
        elif (args.dtype == torch.bfloat16):
            raise NotImplementedError("bfloat16 is not yet supported")

        self.model = self.model.module
        self.input_device = torch.cuda.current_device()

        print_rank_n("Model loaded")


def get_checkpoint_files(model_name_or_path, revision=None):
    # loads files from hub
    cached_repo_dir = snapshot_download(
        model_name_or_path,
        allow_patterns=["*"],
        local_files_only=is_offline_mode(),
        revision=revision
    )

    # creates a list of paths from all downloaded files in cache dir, matching the regex *.pt
    file_list = [str(entry) for entry in Path(
        cached_repo_dir).rglob('*.pt') if entry.is_file()]
    return file_list


def write_checkponts_json(checkpoints_json: str, model_name: str) -> None:
    with io.open(checkpoints_json, "w", encoding="utf-8") as f:
        checkpoint_files = get_checkpoint_files(model_name)
        data = {
            "type": "BLOOM",
            "checkpoints": checkpoint_files,
            "version": 1.0
        }
        json.dump(data, f)
