import os
import sys
from contextlib import ContextDecorator
from functools import wraps
from unittest.mock import patch

import deepspeed
import torch
from transformers import GPT2LMHead, GPT2Tokenizer

from megatron import get_args, get_tokenizer, global_vars, initialize_megatron
from megatron.testing_utils import (
    TestCasePlus,
    mockenv_context,
    set_seed,
    torch_assert_equal,
)
from megatron.training import setup_model_and_optimizer
from pretrain_gpt import get_batch_pipe, model_provider
from tools.convert_checkpoint import deepspeed_to_transformers

from .test_model import flatten_arguments, get_default_args


class run_megatron(ContextDecorator):
    """decorator & context manager for megatron code"""

    def __call__(self, func):
        @wraps(func)
        def decorator():
            command_args = get_default_args()
            with patch("sys.argv", flatten_arguments(command_args)):
                # NOTE: perhaps should not hard code distributed config?
                with mockenv_context(
                    MASTER_ADDR="localhost",
                    MASTER_PORT="9994",
                    RANK="0",
                    LOCAL_RANK="0",
                    WORLD_SIZE="1",
                ):
                    return func()

        return decorator()


def reset_global_variables():
    global_vars._GLOBAL_ARGS = None
    global_vars._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    global_vars._GLOBAL_TOKENIZER = None
    global_vars._GLOBAL_TENSORBOARD_WRITER = None
    global_vars._GLOBAL_ADLR_AUTORESUME = None
    global_vars._GLOBAL_TIMERS = None


class TestConversion(TestCasePlus):
    def setUp(self):
        super().setUp()
        set_seed()
        reset_global_variables()
        self.args = get_args()
        self.token_ids = torch.randint(
            self.args.padded_vocab_size,
            (self.args.micro_batch_size, self.args.seq_length),
        )
        self.megatron_output = None
        self.transformers_output = None

    def test_megatron(self):
        with run_megatron():
            deepspeed.init_distributed()
            initialize_megatron()
            megatron_tokenizer = get_tokenizer()

            megatron_model, _, _ = setup_model_and_optimizer(model_provider)
            megatron_model = megatron_model[0]

            # process batch
            self.token_ids[self.token_ids == megatron_tokenizer.eod] += 1
            self.token_ids[
                self.token_ids == megatron_tokenizer.eod
            ] %= self.args.padded_vocab_size
            tokens = get_batch_pipe({"text": self.token_ids})[0]

            # cache result
            self.megatron_output = megatron_model(*tokens)

            # save model
            state_dict = megatron_input.state_dict_for_save_checkpoint()
            temp_dir = self.get_auto_remove_tmp_dir()
            torch.save(state_dict, os.path.join(temp_dir, "megatron.pt"))

    def test_huggingface(self):
        temp_dir = self.get_auto_remove_tmp_dir()
        transformer_dir = os.path.join(temp_dir, "transformers")

        conversion_args = f"""
            --input_folder {os.path.join(temp_dir, "megatron.pt")}
            --output_folder {transformer_dir}
        """

        with patch.object(sys, "argv", conversion_args):
            deepspeed_to_transformers.main()

        transformers_model = GPT2LMHead.from_pretrained(transformer_dir)
        transformers_tokenizer = GPT2Tokenizer.from_pretrained(transformer_dir)

        tokens = transformers_tokenizer(self.token_ids)
        self.transformers_output = transformers_model(**tokens)

    def test_equal_output(self):
        torch_assert_equal(self.megatron_output, self.transformers_output)
