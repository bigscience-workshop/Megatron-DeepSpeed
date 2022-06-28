"""Pretrain T0"""

import torch

from megatron import get_args, get_tokenizer, mpu
from megatron.utils import get_ltor_masks_and_position_ids, get_packed_attention_mask



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
