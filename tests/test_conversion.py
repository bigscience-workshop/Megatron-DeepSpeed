from contextlib import ContextDecorator
from functools import wraps

import torch

from transformers import GPT2LMHeadModel

from megatron import initialize_megatron, get_args, get_tokenizer, global_vars
from megatron.testing_utils import set_seed, torch_assert_equal, TestCasePlus, mockenv_context
from pretrain_gpt import model_provider, get_batch_pipe


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
            args.padded_vocab_size, (args.micro_batch_size, args.seq_length)
        )
        token_ids[token_ids == tokenizer.eod] += 1
        token_ids[token_ids == tokenizer.eod] %= args.padded_vocab_size
        self.token_ids = token_ids

    def test_megatron(self):
        with run_megatron():
            deepspeed.init_distributed()
            initialize_megatron()
            megatron_tokenizer = get_tokenizer()

            megatron_model, _, _ = setup_model_and_optimizer(model_provider)
            megatron_model = megatron_model[0]

            # process batch
            megatron_input = get_batch_pipe({"text": self.token_ids})[0]
            megatron_output = megatron_model(*megatron_input)

            state_dict = megatron_input.state_dict_for_save_checkpoint()
    
    def test_huggingface(self):
        



        


    def test_equal_output(self):
        hf_output = hf_model(self.input)
        megatron_output = megatron_model(self.input)
        torch_assert_equal(hf_output, megatron_output)
