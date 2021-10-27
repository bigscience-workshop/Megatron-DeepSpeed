import unittest
from random import randint
from unittest.mock import patch

import deepspeed
import torch

from megatron import initialize_megatron, get_args, get_tokenizer, global_vars
from megatron.testing_utils import TestCasePlus, mockenv_context
from megatron.training import setup_model_and_optimizer
from pretrain_gpt import model_provider as gpt_model_provider, get_batch_pipe as get_gpt_batch_pipe
from pretrain_prefix_lm import model_provider as prefix_lm_model_provider, get_batch_pipe as get_prefix_lm_batch_pipe


def get_default_args():
    """return a dictionary with key as argument name and value as additional arguments"""
    return {
        # GPT_ARGS
        "--num-layers": "2",
        "--hidden-size": "128",
        "--num-attention-heads": "4",
        "--seq-length": "256",
        "--max-position-embeddings": "256",
        "--micro-batch-size": "4",
        "--global-batch-size": "8",
        "--lr-decay-iters": "320000",
        "--lr-decay-style": "cosine",
        "--lr": "0.00015",
        "--min-lr": "1.0e-5",
        "--train-iters": "5000",
        "--tokenizer-type": "PretrainedFromHF",
        "--tokenizer-name-or-path": "gpt2",
        "--data-impl": "mmap",
        "--split": "949,50,1",
        "--distributed-backend": "nccl",
        "--weight-decay": "1e-2",
        "--clip-grad": "1.0",
        "--lr-warmup-fraction": ".01",
        "--fp16": "",

        "--attention-dropout": "0",
        "--hidden-dropout": "0",

        # OUTPUT_ARGS
        "--log-interval": "10",
        "--save-interval": "500",
        "--eval-interval": "100",
        "--eval-iters": "10",
        "--checkpoint-activations": "",

        # DATA_ARGS
    }


def flatten_arguments(args):
    """
    Converts dictionary argument to a list.

    Note: we add "IGNORED" at the beginning as this value is ignored by the argparser

    Example: {"arg1": "value1", "arg2": "value2"} -> ["IGNORED", "arg1", "value1", "arg2", "value2"]
    """
    return ["IGNORED"] + [item for key_value in args.items() for item in key_value if item != ""]


def equal_vectors(tensor1, tensor2, dim=-1):
    """View tensor1 and tensor2 as a list of vectors, and compute equality"""
    return torch.linalg.norm(tensor1 - tensor2, dim=dim) == 0


class MyTestCase(TestCasePlus):
    def setUp(self) -> None:
        super().setUp()

        # We reset all global variables
        global_vars._GLOBAL_ARGS = None
        global_vars._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
        global_vars._GLOBAL_TOKENIZER = None
        global_vars._GLOBAL_TENSORBOARD_WRITER = None
        global_vars._GLOBAL_ADLR_AUTORESUME = None
        global_vars._GLOBAL_TIMERS = None

        self.dist_env_1_gpu = dict(
            MASTER_ADDR="localhost", MASTER_PORT="9994", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1"
        )

    def test_gpt(self):
        """Test causal invariance, ie past token don't depend on future tokens."""
        command_args = get_default_args()

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()
                tokenizer = get_tokenizer()

                model, _, _ = setup_model_and_optimizer(gpt_model_provider)
                model = model[0]

                token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))

                # eod is a special token
                token_ids[token_ids == tokenizer.eod] += 1
                token_ids[token_ids == tokenizer.eod] %= args.padded_vocab_size

                # process batch
                input_batch = get_gpt_batch_pipe({"text": token_ids})[0]

                # get a modified version of the first batch, we change a specific index
                changed_index = randint(0, args.seq_length - 2)
                input_token_ids_changed = input_batch[0].clone()
                # We increment the token_id by one for that index in order to artificially change the sequence.
                input_token_ids_changed[:, changed_index] = \
                    (input_token_ids_changed[:,changed_index] + 1) % args.padded_vocab_size

                output = model(*input_batch)
                output_changed = model(input_token_ids_changed, *input_batch[1:])

                # All token in past should be unchanged
                self.assertTrue(
                    torch.all(equal_vectors(output[:, :changed_index], output_changed[:, :changed_index]))
                )
                # All tokens in the future should have changed
                self.assertFalse(
                    torch.any(equal_vectors(output[:, changed_index:], output_changed[:, changed_index:]))
                )

    def test_prefix_lm_reset_attention_mask(self):
        """
        Test prefix invariances when `reset_attention_mask=True`:
            - Past target tokens don't depend on future target tokens.
            - Target tokens depend on input tokens.
            - Input tokens depend on all other input tokens, but never target tokens.
        """
        command_args = get_default_args()

        command_args["--reset-attention-mask"] = ""
        command_args["--loss-on-targets-only"] = ""

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()
                tokenizer = get_tokenizer()

                model, _, _ = setup_model_and_optimizer(prefix_lm_model_provider)
                model = model[0]

                token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))

                # eod is a special token, this also guarantees that the whole row is considered as a document.
                token_ids[token_ids == tokenizer.eod] += 1
                token_ids[token_ids == tokenizer.eod] %= args.padded_vocab_size

                # process batch to have non empty prefix
                input_batch, (_, loss_mask), prefix_indices = get_prefix_lm_batch_pipe({"text": token_ids})

                for batch_id in range(len(prefix_indices)):
                    for id in prefix_indices[batch_id]:
                        self.assertTrue(loss_mask[batch_id, id] == 1)
                        self.assertTrue(id > 0)
                        # Make sure that the last prefix token predicts the first token.
                        self.assertTrue(loss_mask[batch_id, id -1] == 1)

                output = model(*input_batch)

                ## --------------- CHANGE A TARGET TOKEN ---------------------------
                # get a modified version of the first batch
                # guaranteed to exist as each row has at least one partial document
                changed_target_index = prefix_indices[0][0]
                token_ids_changed_target = input_batch[0].clone()
                # We increment the token id on the changed index.
                token_ids_changed_target[0, changed_target_index] = \
                    (token_ids_changed_target[0, changed_target_index] + 1) % args.padded_vocab_size
                # make sure we're not changing a token to eod as it's a special token
                token_ids_changed_target[token_ids_changed_target == tokenizer.eod] += 1
                token_ids_changed_target[token_ids_changed_target == tokenizer.eod] %= args.padded_vocab_size

                # Test change
                output_changed_target = model(token_ids_changed_target, *input_batch[1:])

                # All token in past should be unchanged
                self.assertTrue(
                    torch.all(
                        equal_vectors(output[0, :changed_target_index], output_changed_target[0, :changed_target_index])
                    )
                )
                # All tokens in the future should have changed
                self.assertFalse(
                    torch.any(
                        equal_vectors(output[0, changed_target_index:], output_changed_target[0, changed_target_index:])
                    )
                )
                # Unchanged changed rows should not change either
                self.assertTrue(
                    torch.all(
                        equal_vectors(output[1, :], output_changed_target[1, :])
                    )
                )

                ## --------------- CHANGE AN INPUT TOKEN ---------------------------
                # Let's change the the last prefix token and make sure that the first token changed
                # guaranteed to be positive as we avoid pathological case previously
                last_prefix_index = prefix_indices[0][0] - 1
                token_ids_changed_input = input_batch[0].clone()
                #  We increment the token id on the changed index.
                token_ids_changed_input[0, last_prefix_index] = \
                    (token_ids_changed_input[0, last_prefix_index] + 1) % args.padded_vocab_size
                # make sure we're not changing a token to eod as it's a special token
                token_ids_changed_input[token_ids_changed_input == tokenizer.eod] += 1
                token_ids_changed_input[token_ids_changed_input == tokenizer.eod] %= args.padded_vocab_size

                output_changed_input = model(token_ids_changed_input, *input_batch[1:])

                # All tokens should be changed
                self.assertFalse(
                    torch.any(
                        equal_vectors(output[0, :], output_changed_input[0, :])
                    )
                )
                # Unchanged changed rows should not change either
                self.assertTrue(
                    torch.all(
                        equal_vectors(output[1, :], output_changed_input[1, :])
                    )
                )

    def test_prefix_lm_wo_reset_attention_mask(self):
        """
        Test prefix invariances when `reset_attention_mask=False`:
            - Past target tokens don't depend on future target tokens.
            - Target tokens depend on input tokens.
            - Input tokens depend on all other input tokens, but never target tokens.
        """
        command_args = get_default_args()

        command_args["--loss-on-targets-only"] = ""

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()

                model, _, _ = setup_model_and_optimizer(prefix_lm_model_provider)
                model = model[0]

                token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))
                input_batch, (_, loss_mask), prefix_indices = get_prefix_lm_batch_pipe({"text": token_ids})

                for batch_id in range(len(prefix_indices)):
                    id = prefix_indices[batch_id]
                    self.assertTrue(loss_mask[batch_id, id] == 1)
                    self.assertTrue(id > 0)
                    # Make sure that the last prefix token predicts the first token.
                    self.assertTrue(loss_mask[batch_id, id -1] == 1)

                model(*input_batch)

                #TODO: Check all invariants

    def test_gpt_rotary_embeddings(self):
        """Test rotary embeddings"""
        command_args = get_default_args()

        del command_args["--max-position-embeddings"]
        command_args["--position-embedding-type"] = "rotary"

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()
                tokenizer = get_tokenizer()

                model, _, _ = setup_model_and_optimizer(gpt_model_provider)
                model = model[0]

                token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))

                # eod is a special token
                token_ids[token_ids == tokenizer.eod] += 1
                token_ids[token_ids == tokenizer.eod] %= args.padded_vocab_size

                # process batch
                input_batch = get_gpt_batch_pipe({"text": token_ids})[0]

                model(*input_batch)

                #TODO: Check all invariants


if __name__ == '__main__':
    unittest.main()
