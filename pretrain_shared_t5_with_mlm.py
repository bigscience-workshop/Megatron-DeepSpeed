import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.mlm_dataset import build_train_valid_test_datasets, build_dataset_group
from megatron.model import SharedT5ModelPipe
from megatron.training import pretrain
from megatron.utils import get_attention_masks_and_position_ids

import deepspeed
from deepspeed.runtime.utils import see_memory_usage

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        return fn

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed:
            # TODO @thomasw21: fix this for PP > 1 (the issue is that you're passing two values that require grad)
            assert mpu.get_pipeline_model_parallel_world_size() == 1, "PP > 1 is not supported yet"

            # TODO @thomasw21 hack to bypass a specific check
            args.attn_mask = None

            model = SharedT5ModelPipe(
                num_tokentypes=0,
                parallel_output=True
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe
        else:
            assert False, "Require deepspeed to be running"
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["input_tokens", "target_tokens"]
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    input_tokens = data_b["input_tokens"].long()
    target_tokens = data_b["target_tokens"].long()[:,:-1].contiguous()
    label_tokens = data_b["target_tokens"].long()[:,1:].contiguous()

    # Get the masks and position ids.
    input_attention_mask, _, input_position_ids = get_attention_masks_and_position_ids(
        input_tokens,
        tokenizer.eod,
        reset_position_ids=False, # TODO @thomasw21 doesn't work out of the box
        reset_attention_mask=False, # TODO @thomasw21 doesn't work out of the box
        eod_mask_loss=False, # TODO @thomasw21 doesn't work out of the box
        prefix_indices=None,
        loss_on_targets_only=False,
        ltor=False
    )
    target_attention_mask, target_loss_mask, target_position_ids = get_attention_masks_and_position_ids(
        target_tokens,
        tokenizer.eod,
        reset_position_ids=False,  # TODO @thomasw21 doesn't work out of the box
        reset_attention_mask=False,  # TODO @thomasw21 doesn't work out of the box
        eod_mask_loss=False,  # TODO @thomasw21 doesn't work out of the box
        prefix_indices=None,
        loss_on_targets_only=args.loss_on_targets_only,
        ltor=True
    )

    return (input_tokens, input_attention_mask, input_position_ids, target_tokens, target_attention_mask, target_position_ids), (label_tokens, target_loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""

    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    print_rank_0('> building train, validation, and test datasets for GPT ...')
    # Option 1 of data loading using --data-path

    if args.data_path:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            sequence_length=args.encoder_seq_length + args.decoder_seq_length,
            noise_density=args.noise_density,
            mean_noise_span_length=args.mean_noise_span_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup)
        )
    # Option 2 of data loading using --(train|valid|test)-weighted-split-paths
    elif args.train_weighted_split_paths:
        assigned_train_valid_test = []
        if args.train_weighted_split_paths is not None:
            train_ds = []
            assigned_train_valid_test.append("train")
        if args.valid_weighted_split_paths is not None:
            valid_ds = []
            assigned_train_valid_test.append("valid")
        if args.test_weighted_split_paths is not None:
            test_ds = []
            assigned_train_valid_test.append("test")

        for s in assigned_train_valid_test:
            data_groups = zip(eval(f"args.{s}_weighted_split_paths"),
                                eval(f"args.{s}_weighted_split_weights"),
                                eval(f"args.{s}_weighted_split_splits"),
                                eval(f"args.{s}_weighted_split_names"))
            for paths, weights, splits, name in data_groups:
                d = build_dataset_group(
                    dataset_group_name=name,
                    paths=paths,
                    weights=weights,
                    splits=splits,
                    data_impl=args.data_impl,
                    train_valid_test_num_samples=train_val_test_num_samples,
                    seq_length=args.encoder_seq_length + args.decoder_seq_length,
                    noise_density=args.noise_density,
                    mean_noise_span_length=args.mean_noise_span_length,
                    seed=args.seed,
                    skip_warmup=(not args.mmap_warmup),
                    train_valid_test=s
                )
                eval(f"{s}_ds").append(d)
    else:
        raise NotImplementedError("No dataloading argument passed")

    print_rank_0("> finished creating MLM datasets ...")
    return train_ds, valid_ds, test_ds

@record
def main():
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        # TODO @thomasw21: make it work without DS.
        forward_step_func=None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

if __name__ == "__main__":
    main()
