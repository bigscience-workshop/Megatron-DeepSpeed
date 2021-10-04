import json
import os
import sys
from unittest.mock import patch

import deepspeed
import torch

torch.backends.cudnn.deterministic = True
from transformers import GPT2LMHeadModel

from megatron import get_args, get_tokenizer, global_vars, initialize_megatron
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.testing_utils import (
    TestCasePlus,
    mockenv_context,
    set_seed,
    torch_assert_close,
)
from megatron.training import setup_model_and_optimizer
from pretrain_gpt import get_batch_pipe as get_gpt_batch_pipe
from pretrain_gpt import model_provider as gpt_model_provider

# fix the relative path
sys.path.append("tools/convert_checkpoint/")
from test_model import flatten_arguments, get_default_args

from tools.convert_checkpoint import deepspeed_to_megatron, deepspeed_to_transformers
# FIXME: additional package requirement
from parameterized import parameterized


def reset_global_variables():
    global_vars._GLOBAL_ARGS = None
    global_vars._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    global_vars._GLOBAL_TOKENIZER = None
    global_vars._GLOBAL_TENSORBOARD_WRITER = None
    global_vars._GLOBAL_ADLR_AUTORESUME = None
    global_vars._GLOBAL_TIMERS = None


class TestCheckpointConversion(TestCasePlus):
    def setUp(self) -> None:
        super().setUp()
        set_seed()
        reset_global_variables()

        self.dist_env_1_gpu = dict(
            MASTER_ADDR="localhost",
            MASTER_PORT="1123",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )

    def get_megatron_ds_args(self, fp16):
        # get megatron and deepspeed args
        megatron_args = get_default_args()

        # reset the number of layers
        # megatron_args['--num-layers'] = '50'

        ds_config_path = f"{self.test_file_dir_str}/ds_config.json"
        if not fp16:
            megatron_args.pop("--fp16")
            ds_config = json.load(open(f"{self.test_file_dir_str}/ds_config.json"))
            ds_config["fp16"]["enabled"] = False
            ds_config_path = os.path.join(
                self.get_auto_remove_tmp_dir(), "ds_config.json"
            )
            with open(ds_config_path, "w") as f:
                json.dump(ds_config, f)

        ds_args = f"""
            --deepspeed
            --deepspeed_config {ds_config_path}
            --zero-stage 1
            --deepspeed-activation-checkpointing
        """.split()

        return megatron_args, ds_args

    def save_and_get_ds_megatron_output(
        self, args, ds_megatron_model, ds_megatron_token, ds_megatron_dir
    ):
        # args for checkpointing
        args.save, args.no_save_optim = ds_megatron_dir, True
        # save deepspeed megatron model
        save_checkpoint(1, ds_megatron_model, None, None)

        ds_megatron_model = ds_megatron_model[0]

        # get processed batch
        _, _, attention_mask = ds_megatron_token

        # disable activation checkpoint
        ds_megatron_model.module.activation_checkpoint_interval = 0
        # resemble _exec_schedule in /deepspeed/runtine/pipe/engine.py
        ds_megatron_model._compute_loss = False
        ds_megatron_model.fwd_outputs = []

        ds_megatron_model.eval()
        # due to the hack in https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/68b46f201aad8803d0699f76b100663553e89e1b/pretrain_gpt.py#L74
        args.attn_mask = attention_mask

        # use deepspeed pipeline thing
        ds_megatron_model.pipe_buffers["inputs"].append(ds_megatron_token)
        ds_megatron_model.pipe_buffers["outputs"].append(None)

        with torch.no_grad():
            ds_megatron_model._exec_forward_pass(buffer_id=0)

        # gather output
        ds_megatron_output = ds_megatron_model.pipe_buffers["outputs"][0]
        return ds_megatron_output


    @parameterized.expand([
        ("fp32", False),
        ("fp16", True),
    ])
    def test_megatron_ds_to_megatron(self, name, fp16):
        # run deepspeed to megatron first, then convert to transformers, then compare the difference
        
        megatron_args, ds_args = self.get_megatron_ds_args(fp16)
        # run deepspeed megatron
        with patch("sys.argv", flatten_arguments(megatron_args) + ds_args):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()

                # get deepspeed megatron model
                ds_megatron_model, _, _ = setup_model_and_optimizer(gpt_model_provider)

                megatron_tokenizer = get_tokenizer()
                # always use this token_ids
                token_ids = torch.randint(
                    args.padded_vocab_size, (args.micro_batch_size, args.seq_length)
                )
                # process batch
                token_ids[token_ids == megatron_tokenizer.eod] += 1
                token_ids[token_ids == megatron_tokenizer.eod] %= args.padded_vocab_size

                # get processed batch
                megatron_token, _ = get_gpt_batch_pipe({"text": token_ids})

                # save directory
                ds_megatron_dir = self.get_auto_remove_tmp_dir()

                # gather deepspeed model output
                ds_megatron_output = self.save_and_get_ds_megatron_output(
                    args, ds_megatron_model, megatron_token, ds_megatron_dir
                )

        # run conversion file
        megatron_dir = self.get_auto_remove_tmp_dir()
        conversion_args = {
            "--input_folder": f"{ds_megatron_dir}",
            "--output_folder": f"{megatron_dir}",
        }
        with patch("sys.argv", flatten_arguments(conversion_args)):
            deepspeed_to_megatron.main()

        # run megatron
        with patch("sys.argv", flatten_arguments(megatron_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                args.deepspeed = False
                assert args.deepspeed is False
                megatron_model, _, _ = setup_model_and_optimizer(gpt_model_provider)

                args.load, args.no_load_rng = megatron_dir, True
                load_checkpoint(megatron_model, None, None)

                megatron_model = megatron_model[0]

                megatron_model.eval()
                megatron_output = megatron_model(*megatron_token)

        # compare the difference
        # torch_assert_equal(ds_megatron_output.data.cpu(), megatron_output.data.cpu())
        torch_assert_close(ds_megatron_output.data.cpu(), megatron_output.data.cpu())

    @parameterized.expand([
        ("fp32", False),
        ("fp16", True),
    ])
    def test_megatron_ds_to_transformers(self, name, fp16):
        # run megatron first, then convert to transformers, then compare the difference
        megatron_args, ds_args = self.get_megatron_ds_args(fp16)

        # run deepspeed megatron
        with patch("sys.argv", flatten_arguments(megatron_args) + ds_args):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()

                # get deepspeed megatron model
                ds_megatron_model, _, _ = setup_model_and_optimizer(gpt_model_provider)

                megatron_tokenizer = get_tokenizer()
                # always use this token_ids
                token_ids = torch.randint(
                    args.padded_vocab_size, (args.micro_batch_size, args.seq_length)
                )
                # process batch
                token_ids[token_ids == megatron_tokenizer.eod] += 1
                token_ids[token_ids == megatron_tokenizer.eod] %= args.padded_vocab_size

                # get processed batch
                megatron_token, _ = get_gpt_batch_pipe({"text": token_ids})

                # save directory
                ds_megatron_dir = self.get_auto_remove_tmp_dir()

                # gather deepspeed model output
                ds_megatron_output = self.save_and_get_ds_megatron_output(
                    args, ds_megatron_model, megatron_token, ds_megatron_dir
                )

        # run conversion file
        transformer_dir = self.get_auto_remove_tmp_dir()
        conversion_args = {
            "--input_folder": f"{ds_megatron_dir}",
            "--output_folder": f"{transformer_dir}",
        }
        with patch("sys.argv", flatten_arguments(conversion_args)):
            deepspeed_to_transformers.main()

        torch_dtype = torch.get_default_dtype() if not fp16 else torch.float16
        transformers_model = GPT2LMHeadModel.from_pretrained(transformer_dir, torch_dtype=torch_dtype)

        transformers_model.cuda()
        transformers_model.eval()
        # FIXME: potential error here, do not have tokenizer saved in deepspeed_to_transformers file
        # transformers_tokenizer = GPT2Tokenizer.from_pretrained(transformer_dir)
        # transformers_tokens = transformers_tokenizer(token_ids)

        # FIXME: we do not have attention mask here
        tokens, position_ids, attention_mask = megatron_token
        transformers_output = transformers_model(input_ids=tokens)
        transformers_output = transformers_output.logits

        if fp16:
            # from megatron.model.module import float16_to_fp32
            # transformers_output = float16_to_fp32( transformers_output )
            ds_megatron_output = ds_megatron_output.half()

        # compare the difference
        # FIXME: it could now pass torch_assert_close, but not torch_assert_equal
        # torch_assert_equal(ds_megatron_output.data.cpu(), transformers_output.data.cpu())
        torch_assert_close(
            ds_megatron_output.data.cpu(), transformers_output.data.cpu()
        )