"""Multitask Finetuning T0"""

from multiprocessing.sharedctypes import Value
import torch

from megatron import get_args, get_tokenizer, print_rank_0, mpu
from megatron.data.non_causal_mtf_dataset import build_train_valid_test_datasets, build_dataset_group
from megatron.model import GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, get_packed_attention_mask
from megatron.utils import average_losses_across_data_parallel_group

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
            raise NotImplementedError("DeepSpeed is required for T0")

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


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    print_rank_0('> building train, validation, and test datasets for T0 ...')
    # Option 1 of data loading using --data-path
    # For T0, data has to be provided in the form --data-path input-data target-data input-data2 target-data2 ...
    if args.data_path:

        # Turn into list of pairs; Overwrite args.data_path to keep len = 1
        # TODO: Not yet compatible with dataset weights (Will break at prefixes, weights = analyze_data_prefix(args.data_path))
        assert len(args.data_path) > 1, "Please provide data in pairs of two: input_tokens target_tokens"
        args.data_path = [{"input_tokens": args.data_path[i], "target_tokens": args.data_path[i+1]} for i in range(0, len(args.data_path), 2)]

        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))
    else:
        raise NotImplementedError("No dataloading argument passed")

    print_rank_0("> finished creating T0 datasets ...")
    return train_ds, valid_ds, test_ds

@record
def main():
    pretrain(train_valid_test_datasets_provider, 
            model_provider, 
            forward_step_func=None,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

if __name__ == "__main__":
    main()
