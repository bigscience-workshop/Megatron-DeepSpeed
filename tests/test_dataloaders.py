import itertools
from unittest.mock import patch

import deepspeed

from megatron import global_vars, get_tokenizer, initialize_megatron, get_args
from megatron.data import mlm_dataset, mtf_dataset
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.testing_utils import TestCasePlus, flatten_arguments, mockenv_context


def get_default_args():
    """return a dictionary with key as argument name and value as additional arguments"""
    return {
        # GPT_ARGS
        "--num-layers": "2",
        "--hidden-size": "128",
        "--num-attention-heads": "4",
        "--seq-length": "512",
        "--max-position-embeddings": "512",
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

class TestDataLoading(TestCasePlus):
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

    def test_mlm_dataset(self):
        command_args = get_default_args()
        command_args["--data-path"] = f"{self.data_dir}/gpt2/meg-gpt2-openwebtext_text_document"
        command_args["--noise_density"] = "0.15"
        command_args["--mean_noise_span_length"] = "3"
        command_args["--vocab-extra-ids"] = "100"

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()

                # tokenizer
                tokenizer = get_tokenizer()
                # SEP is required to put in MLM preprocessed.
                tokenizer.tokenizer.add_special_tokens({"sep_token": "<s>"})

                args = get_args()
                train_val_test_num_samples = [
                    args.train_iters * args.global_batch_size,
                    args.eval_iters * args.global_batch_size,
                    0
                ]
                train_ds, valid_ds, test_ds = mlm_dataset.build_train_valid_test_datasets(
                    data_prefix=args.data_path,
                    data_impl=args.data_impl,
                    splits_string=args.split,
                    # TODO @thomasw21 figure how that value works
                    train_valid_test_num_samples=train_val_test_num_samples,
                    sequence_length=args.seq_length,
                    noise_density=args.noise_density,
                    mean_noise_span_length=args.mean_noise_span_length,
                    seed=args.seed,
                    skip_warmup=(not args.mmap_warmup)
                )

                sample = train_ds[0]
                # +1 is needed to compute labels. As inputs and targets are just concatenated.
                self.assertEqual(len(sample["input_tokens"]) + len(sample["target_tokens"]), args.seq_length + 1)

                # We make sure that inputs/targets end with <sep>
                self.assertEqual(sample["input_tokens"][-1], tokenizer.sep)
                self.assertEqual(sample["target_tokens"][-1], tokenizer.sep)

    def test_mtf_dataset(self):
        command_args = get_default_args()
        command_args["--data-path"] = f"{self.data_dir}/gpt2/ag_news_prompt"
        command_args["--dataloader-type"] = "decoder_packed"

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()

                args = get_args()
                train_val_test_num_samples = [
                    args.train_iters * args.global_batch_size,
                    args.eval_iters * args.global_batch_size,
                    0
                ]
                train_ds, valid_ds, test_ds = mtf_dataset.build_train_valid_test_datasets(
                    data_prefix=args.data_path,
                    data_impl=args.data_impl,
                    splits_string=args.split,
                    # TODO @thomasw21 figure how that value works
                    train_valid_test_num_samples=train_val_test_num_samples,
                    seed=args.seed,
                    skip_warmup=(not args.mmap_warmup)
                )

                # TODO @thomasw21 make sure that input and target are aligned.


    def test_mtf_packed_dataloader(self):
        command_args = get_default_args()
        command_args["--data-path"] = f"{self.data_dir}/gpt2/ag_news_prompt"
        command_args["--dataloader-type"] = "decoder_packed"

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()

                args = get_args()
                train_val_test_num_samples = [
                    args.train_iters * args.global_batch_size,
                    args.eval_iters * args.global_batch_size,
                    0
                ]
                train_ds, valid_ds, test_ds = mtf_dataset.build_train_valid_test_datasets(
                    data_prefix=args.data_path,
                    data_impl=args.data_impl,
                    splits_string=args.split,
                    # TODO @thomasw21 figure how that value works
                    train_valid_test_num_samples=train_val_test_num_samples,
                    seed=args.seed,
                    skip_warmup=(not args.mmap_warmup)
                )

                batch_sampler = build_pretraining_data_loader(
                    train_ds, consumed_samples=0, num_workers=4
                )

                last_padding_size = 0
                for i, items in enumerate(batch_sampler):
                    micro_batch_size, seq_length = items["decoder_target_tokens"].shape

                    # `micro_batch_size` correspond to the one in argument
                    self.assertEqual(micro_batch_size, args.micro_batch_size)
                    # `seq_length` correspond to the one in argument + 1 in order to get tokens/labels
                    self.assertEqual(seq_length, args.seq_length + 1)

                    original_samples_count = 0
                    for batch_id in range(micro_batch_size):
                        segment_ids = [k for k, _ in itertools.groupby(items["decoder_segment_ids"][batch_id])]
                        # `segment_ids` is [1,2,...]
                        self.assertEqual(segment_ids[:-1], list(range(1, len(segment_ids))))
                        # `0` signify that the tokens are padding
                        self.assertEqual(segment_ids[-1], 0)
                        original_samples_count += len([segment_id for segment_id in segment_ids if segment_id != 0])

                    # Test that we actually pack, ie we have more samples than the `batch_size`
                    self.assertGreater(original_samples_count, micro_batch_size)

                    # Test that the first sample of each batch couldn't fit inside the previous batch
                    first_sample_segment_ids = next(itertools.groupby(items["decoder_segment_ids"][0]))[1]
                    first_sample_size = len(first_sample_segment_ids)
                    self.assertGreater(first_sample_size, last_padding_size)

                    # update `last_padding_size`
                    last_padding_size = len([None for segment_id in items["decoder_segment_ids"][micro_batch_size - 1] if segment_id == 0])

