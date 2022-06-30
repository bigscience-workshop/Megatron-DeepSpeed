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

"""Dataloaders."""

from functools import partial
import logging

import numpy as np
import torch

from megatron import get_args
from megatron import mpu

logger = logging.get_logger(__name__)


def pack_samples(items, max_seq_len=2049):
    """
    Items:
        [{'input_tokens': array([ 6, 7, 8, 3]), 
        'target_tokens': array([4, 5])}, {'input_tokens'...
    
    Output:
        decoder_target_tokens = [[6, 7, 8, 3, 4, 5, 0]]
        decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
        decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]
    """

    decoder_target_tokens = [[]]
    decoder_segment_ids = [[]]
    decoder_causal_attention = [[]]

    batch_num = 0
    item_num = 0
    cur_len = 0
    for token_dict in items:
        input_token_len = len(token_dict["input_tokens"])
        target_token_len = len(token_dict["target_tokens"])
        total_len = input_token_len + target_token_len
        if cur_len + total_len > max_seq_len:
            len_diff = max_seq_len - cur_len
            logger.info(f"Loosing {len_diff} tokens to padding.")
            # Padding
            if len_diff > 0:
                decoder_target_tokens[batch_num].append(np.zeros((len_diff)))
                decoder_segment_ids[batch_num].append(np.zeros((len_diff)))
                decoder_causal_attention[batch_num].append(np.zeros((len_diff)))
            batch_num += 1
            item_num = 0
            cur_len = 0
            decoder_target_tokens.append([])
            decoder_segment_ids.append([])
            decoder_causal_attention.append([])

        decoder_target_tokens[batch_num].append(token_dict["input_tokens"])
        decoder_target_tokens[batch_num].append(token_dict["target_tokens"])
        cur_len += total_len

        decoder_segment_ids[batch_num].append(np.ones((total_len)) + item_num)
        decoder_causal_attention[batch_num].append(np.ones((input_token_len)))
        decoder_causal_attention[batch_num].append(np.zeros((target_token_len)))
        item_num += 1
    # Padding
    len_diff = max_seq_len - cur_len
    if len_diff > 0:
        decoder_target_tokens[batch_num].append(np.zeros((len_diff)))
        decoder_segment_ids[batch_num].append(np.zeros((len_diff)))
        decoder_causal_attention[batch_num].append(np.zeros((len_diff)))

    # Normally the default collate_fn handles torch tensor conversion; As we use a custom collate_fn, do it here
    return {
        "decoder_target_tokens": torch.as_tensor(np.stack([np.concatenate(arr) for arr in decoder_target_tokens]), dtype=torch.int64),
        "decoder_segment_ids": torch.as_tensor(np.stack([np.concatenate(arr) for arr in decoder_segment_ids]), dtype=torch.int64),
        "decoder_causal_attention": torch.as_tensor(np.stack([np.concatenate(arr) for arr in decoder_causal_attention]), dtype=torch.int64),
    }



def build_pretraining_data_loader(dataset, consumed_samples, num_workers=None):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'packed':
        batch_sampler = MegatronPackedRandomSampler(
            sequence_length=args.seq_length + 1,
            dataset=dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    if num_workers is None:
        num_workers = args.num_workers

    collate_fn = None
    if args.dataloader_type == 'packed':
        collate_fn = partial(pack_samples, max_seq_len=args.seq_length + 1)

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       collate_fn=collate_fn,
                                       pin_memory=True)

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                       * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class MegatronPackedRandomSampler(object):
    """
    To be used with pack_samples collate_fn
    """
    def __init__(self, sequence_length, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                       * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.epoch)

        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        batch_count = 0
        token_lens = 0
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            tok_len = len(self.dataset[idx]['input_tokens']) + len(self.dataset[idx]['target_tokens'])
            if token_lens + tok_len > self.sequence_length:
                batch_count += 1
                token_lens = 0

            if batch_count == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch_count = 0
                batch = []
            else:
                token_lens += tok_len
                batch.append(idx)
