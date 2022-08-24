import os
from argparse import Namespace

import deepspeed
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig

from utils import Model


class DSZeROModel(Model):
    def __init__(self, args: Namespace) -> None:
        if (args.local_rank == 0):
            print("Loading model...")

        config = AutoConfig.from_pretrained(args.model_name)

        world_size = int(os.getenv('WORLD_SIZE', '1'))
        train_batch_size = 1 * world_size

        ds_config = {
            "fp16": {
                "enabled": args.dtype == torch.float16,
            },
            "bf16": {
                "enabled": args.dtype == torch.bfloat16,
            },
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": config.hidden_size * config.hidden_size,
                "stage3_prefetch_bucket_size": 0.9 * config.hidden_size * config.hidden_size,
                "stage3_param_persistence_threshold": 0
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }

        if (args.cpu_offload):
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True
            }

        # this tells from_pretrained to instantiate directly on gpus
        dschf = HfDeepSpeedConfig(ds_config)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.pad = self.tokenizer.pad_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=args.dtype)
        self.model = self.model.eval()
        self.model = deepspeed.initialize(
            model=self.model, config_params=ds_config)[0]
        self.model.module.eval()
        self.model = self.model.module

        self.input_device = torch.cuda.current_device()
