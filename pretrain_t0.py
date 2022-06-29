"""Multitask Finetuning T0"""

import torch
from functools import partial

from megatron import get_args, get_tokenizer, get_timers, print_rank_0, mpu
from megatron.data.non_causal_mtf_dataset import build_train_valid_test_datasets, build_dataset_group
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, get_packed_attention_mask

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import os

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
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe_packed
        else:
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch_pipe_packed(data):
    """
    Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator` & in packed fashion
    
    data:
    decoder_target_tokens = [[6, 7, 8, 3, 4, 5, 0]]
    decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
    decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]
    """
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['decoder_target_tokens', 'decoder_segment_ids', 'decoder_causal_attention']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    tokens_ = data_b['decoder_target_tokens'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    
    segment_ids = data_b['decoder_segment_ids'].long()[:, :-1]
    decoder_causal_attention = data_b['decoder_causal_attention'].long()[:, :-1]

    # Get the masks and position ids.
    causal_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        prefix_indices=None,
        loss_on_targets_only=False # This is done below
    )
    # Only compute loss over causal target tokens, i.e. ignore input_tokens & padding
    loss_mask *= torch.logical_and((decoder_causal_attention - 1) * -1, tokens)
    loss_mask = loss_mask.to(datatype)

    attention_mask = get_packed_attention_mask(
        causal_mask=causal_mask,
        tokens=tokens,
        decoder_causal_attention=decoder_causal_attention,
        segment_ids=segment_ids,
        datatype=datatype,
    )

    if args.curriculum_learning and args.curriculum_seqlen < tokens.size()[1]:
        # seqlen-based curriculum learning
        # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)
    if args.curriculum_learning and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    print_rank_0('> building train, validation, and test datasets for GPT ...')
    # Option 1 of data loading using --data-path

    if args.data_path:

        import json
        data_path_dict = [json.loads(args.data_path)]

        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=data_path_dict,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))

    # # Option 2 of data loading using --(train|valid|test)-weighted-split-paths
    # elif args.train_weighted_split_paths:
    #     assigned_train_valid_test = []
    #     if args.train_weighted_split_paths is not None:
    #         train_ds = []
    #         assigned_train_valid_test.append("train")
    #     if args.valid_weighted_split_paths is not None:
    #         valid_ds = []
    #         assigned_train_valid_test.append("valid")
    #     if args.test_weighted_split_paths is not None:
    #         test_ds = []
    #         assigned_train_valid_test.append("test")

    #     for s in assigned_train_valid_test:
    #         data_groups = zip(eval(f"args.{s}_weighted_split_paths"),
    #                             eval(f"args.{s}_weighted_split_weights"),
    #                             eval(f"args.{s}_weighted_split_splits"),
    #                             eval(f"args.{s}_weighted_split_names"))
    #         for paths, weights, splits, name in data_groups:
    #             d = build_dataset_group(name, paths, weights, splits,
    #                                     args.data_impl,
    #                                     train_val_test_num_samples,
    #                                     args.seq_length, args.seed,
    #                                     (not args.mmap_warmup),
    #                                     train_valid_test=s)
    #             eval(f"{s}_ds").append(d)
    else:
        raise NotImplementedError("No dataloading argument passed")

    print_rank_0("> finished creating T0 datasets ...")
    return train_ds, valid_ds, test_ds

@record
def main():
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

if __name__ == "__main__":
    main()
