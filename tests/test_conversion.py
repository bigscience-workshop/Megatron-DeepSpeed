import os
import sys
from contextlib import ContextDecorator
from functools import wraps
from unittest.mock import patch

import deepspeed
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from megatron import get_args, get_tokenizer, global_vars, initialize_megatron
from megatron.testing_utils import (
    TestCasePlus,
    mockenv_context,
    set_seed,
    torch_assert_equal,
)
from megatron.training import setup_model_and_optimizer
from megatron.checkpointing import load_checkpoint, save_checkpoint

from pretrain_gpt import get_batch_pipe, model_provider

# fix the relative path
sys.path.append('tools/convert_checkpoint/')
from tools.convert_checkpoint import deepspeed_to_transformers
from test_model import flatten_arguments, get_default_args


def reset_global_variables():
    global_vars._GLOBAL_ARGS = None
    global_vars._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    global_vars._GLOBAL_TOKENIZER = None
    global_vars._GLOBAL_TENSORBOARD_WRITER = None
    global_vars._GLOBAL_ADLR_AUTORESUME = None
    global_vars._GLOBAL_TIMERS = None


class TestConversion(TestCasePlus):
    def setUp(self) -> None:
        super().setUp()
        set_seed()
        reset_global_variables()
        self.dist_env_1_gpu = dict(
            MASTER_ADDR="localhost", MASTER_PORT="9994", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1"
        )

    def test_deepspeed_to_transformers(self):
        # run megatron first, then convert to transformers, then compare the difference
        command_args = get_default_args()

        ds_args = f"""
            --deepspeed
            --deepspeed_config {self.test_file_dir_str}/ds_config.json
            --zero-stage 1
            --deepspeed-activation-checkpointing
        """.split()
        # run deepspeed megatron
        with patch('sys.argv', flatten_arguments(command_args) + ds_args):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()
                megatron_dir = self.get_auto_remove_tmp_dir()
                # args for checkpointing
                args.save, args.no_save_optim = megatron_dir, True

                # fix this token_ids
                token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))

                megatron_tokenizer = get_tokenizer()

                # get deepspeed megatron model
                megatron_model, _, _ = setup_model_and_optimizer(model_provider)

                # save deepspeed megatron model
                save_checkpoint(0, megatron_model, _, _)

                megatron_model = megatron_model[0]

                # process batch
                token_ids[token_ids == megatron_tokenizer.eod] += 1
                token_ids[token_ids == megatron_tokenizer.eod] %= args.padded_vocab_size

                # get processed batch
                megatron_token, _ = get_batch_pipe({"text": token_ids})
                tokens, position_ids, attention_mask = megatron_token

                # disable activation checkpoint
                megatron_model.module.activation_checkpoint_interval = 0
                # resemble _exec_schedule in /deepspeed/runtine/pipe/engine.py
                megatron_model._compute_loss = False
                megatron_model.fwd_outputs = []

                # due to the hack in https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/68b46f201aad8803d0699f76b100663553e89e1b/pretrain_gpt.py#L74
                args.attn_mask = attention_mask

                # use deepspeed pipeline thing 
                megatron_model.pipe_buffers["inputs"].append( megatron_token )
                megatron_model.pipe_buffers["outputs"].append( None )

                with torch.no_grad():
                    megatron_model._exec_forward_pass(buffer_id=0)

                # gather output
                megatron_output = megatron_model.pipe_buffers["outputs"][0]

                # run conversion file 
                transformer_dir = self.get_auto_remove_tmp_dir()
                conversion_args = {
                    "--input_folder": f"{megatron_dir}", 
                    "--output_folder": f"{transformer_dir}"
                }
                with patch('sys.argv', flatten_arguments(conversion_args)):
                    deepspeed_to_transformers.main()

                transformers_model = GPT2LMHeadModel.from_pretrained(transformer_dir)

                transformers_model.cuda()
                # FIXME: potential error here, do not have tokenizer saved in deepspeed_to_transformers file
                # transformers_tokenizer = GPT2Tokenizer.from_pretrained(transformer_dir)
                # transformers_tokens = transformers_tokenizer(token_ids)

                # FIXME: we do not have attention mask here
                transformers_output = transformers_model(input_ids=tokens)
                transformers_output = transformers_output.logits

        # compare the difference 
        torch_assert_equal(megatron_output.cpu(), transformers_output.cpu())
