# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain UL2"""

from functools import partial

import torch

from megatron import (
    get_args,
    get_timers,
    mpu,
    print_rank_0
)
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.data.ul2_dataset import (
    is_decoder_only as _is_decoder_only,
    is_prefix_lm as _is_prefix_lm,
)
from megatron.model.gpt_model import GPTModel
from megatron.model.t5_model import T5Model, t5_position_ids
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group


def is_decoder_only():
    """Return whether we use a decoder-only model."""
    args = get_args()
    return _is_decoder_only(args.ul2_model_type)


def is_prefix_lm():
    """Return whether we use a non-causal decoder-only model."""
    args = get_args()
    return _is_prefix_lm(args.ul2_model_type)


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    assert pre_process and post_process, "UL2 doesn't yet support pipelining"

    print_rank_0('building UL2 model ...')
    if is_decoder_only():
        print_rank_0('Using decoder-only UL2 model.')
        model = GPTModel(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            prefix_lm=is_prefix_lm(),
        )
    else:
        print_rank_0('Using encoder-decoder UL2 model.')
        model = T5Model(num_tokentypes=0, parallel_output=True)
    return model


def get_batch(data_iterator):
    """Build the batch."""

    if is_decoder_only():
        keys = ['text', 'labels', 'loss_mask', 'dec_mask']
    else:
        keys = ['text_enc', 'text_dec', 'labels', 'loss_mask',
                'enc_mask', 'dec_mask', 'enc_dec_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    if is_decoder_only():
        tokens = data_b['text'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        dec_mask = (data_b['dec_mask'] < 0.5)
        dec_mask = dec_mask.unsqueeze(1)
        return tokens, loss_mask, labels, dec_mask
    else:
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = (data_b['enc_mask'] < 0.5)
        dec_mask = (data_b['dec_mask'] < 0.5)
        enc_dec_mask = (data_b['enc_dec_mask'] < 0.5)

        return tokens_enc, tokens_dec, loss_mask, labels, \
               enc_mask, dec_mask, enc_dec_mask


def loss_func(loss_mask, output_tensor):
    if is_decoder_only():
        lm_loss_ = output_tensor
    else:
        lm_loss_, _ = output_tensor

    lm_loss_ = lm_loss_.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group([lm_loss])

    return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    if is_decoder_only():
        (tokens, loss_mask, lm_labels, dec_mask) = get_batch(data_iterator)
    else:
        (
            tokens_enc, tokens_dec, loss_mask, lm_labels,
            enc_mask, dec_mask, enc_dec_mask,
        ) = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model lm_labels
    if is_decoder_only():
        position_ids = t5_position_ids(tokens)
        output_tensor = model(tokens, position_ids, dec_mask,
                              labels=lm_labels)
    else:
        output_tensor = model(tokens_enc,
                              tokens_dec,
                              enc_mask,
                              dec_mask,
                              enc_dec_mask,
                              tokentype_ids=None,
                              lm_labels=lm_labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for UL2 ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.encoder_seq_length,
        max_seq_length_dec=args.decoder_seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        dataset_type='ul2')
    print_rank_0("> finished creating UL2 datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
