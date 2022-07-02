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
import torch

from megatron import get_args, get_tokenizer
from megatron import mpu
from megatron.data.mtf_dataset import MTFDataset


def pack_samples(items, max_seq_len: int, micro_batch_size: int, pad_token: int):
    """
    Greedily packs samples.

    Items:
        [
            {
                'input_tokens': array([6, 7]),
                'target_tokens': array([8])
            },
            {
                'input_tokens': array([3, 4]),
                'target_tokens': array([5])
            }
        ]

    Output:
        decoder_tokens = [[6, 7, 8, 3, 4, 5, <pad>]]: Concatenation of tokens followed with padding tokens.
        decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]: Segment ids determine original documents.
        decoder_is_inputs = [[1, 1, 0, 1, 1, 0, 0]]: `1` depicts inputs, `0` depicts target.
    """

    decoder_tokens = torch.full((micro_batch_size, max_seq_len), pad_token, dtype=torch.int64)
    decoder_segment_ids = torch.zeros((micro_batch_size, max_seq_len), dtype=torch.int64)
    decoder_is_inputs = torch.full((micro_batch_size, max_seq_len), False, dtype=torch.bool)

    batch_num = 0
    # `0` is reserved for padding
    item_num = 1
    cur_len = 0
    for token_dict in items:
        input_token_len = len(token_dict["input_tokens"])
        target_token_len = len(token_dict["target_tokens"])
        total_len = input_token_len + target_token_len
        if cur_len + total_len > max_seq_len:
            len_diff = max_seq_len - cur_len
            # Padding
            if len_diff > 0:
                decoder_tokens[batch_num][cur_len: max_seq_len] = pad_token
                decoder_segment_ids[batch_num][cur_len: max_seq_len] = 0
                # padded values are already 0, no need to update `decoder_is_inputs`
            batch_num += 1
            assert batch_num < micro_batch_size
            item_num = 1
            cur_len = 0

        decoder_tokens[batch_num][cur_len: cur_len + input_token_len] = torch.from_numpy(token_dict["input_tokens"])
        decoder_tokens[batch_num][cur_len + input_token_len: cur_len + total_len] = torch.from_numpy(token_dict["target_tokens"])
        decoder_segment_ids[batch_num][cur_len: cur_len + total_len] = item_num
        decoder_is_inputs[batch_num][cur_len: cur_len + input_token_len] = 1  # inputs
        # targets are already 0 at init, no need to update `decoder_is_inputs`

        item_num += 1
        cur_len += total_len
        assert cur_len < max_seq_len

    # Normally the default collate_fn handles torch tensor conversion; As we use a custom collate_fn, do it here
    return {
        "decoder_token_ids": decoder_tokens,
        "decoder_segment_ids": decoder_segment_ids,
        "decoder_is_inputs": decoder_is_inputs,
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
    elif args.dataloader_type == 'decoder_packed':
        assert isinstance(dataset, MTFDataset)
        batch_sampler = MegatronDecoderPackedText2TextRandomSampler(
            sequence_length=args.seq_length + 1,
            dataset=dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'packed':
        batch_sampler = MegatronPackedRandomSampler(
            sequence_length=args.seq_length,
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
    if args.dataloader_type == 'decoder_packed':
        assert isinstance(dataset, MTFDataset)
        pad_token = get_tokenizer().pad
        collate_fn = partial(pack_samples, max_seq_len=args.seq_length + 1, micro_batch_size=args.micro_batch_size,
                             pad_token=pad_token)

    # Torch dataloader.
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

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


class MegatronPackedRandomSampler(object):
    """docstring for MegatronPackedRandomSampler"""
    def __init__(self, sequence_length, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size):
        # Keep a copy of input params for later use.
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
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class MegatronDecoderPackedText2TextRandomSampler(object):
    """
    Converts a two stream dataset with `input_tokens` and `target_tokens` and creates a batch that should be greedily
    packed to be passed onto the decoder model.

    To be used with `pack_samples` as collate_fn
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
        self.active_total_samples = self.total_samples - self.last_batch_size

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
        current_epoch_samples = self.consumed_samples % self.active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                      * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()

        # Infinite loader
        while True:
            g.manual_seed(self.epoch)

            # Randomly shuffle the dataset
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

    @property
    def epoch(self):
        return self.consumed_samples // self.active_total_samples
