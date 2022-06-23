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

"""GPT Non-Causal Mask Language Model Finetune Style dataset."""

import os
import time
import random
import collections

import numpy as np
import torch

from megatron import mpu, print_rank_0, get_tokenizer
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples, create_masked_lm_predictions
from megatron.data.dataset_utils import get_train_valid_test_split_, get_split_by_range_, get_indexed_dataset_
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset


def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    max_seq_length,
                                    noise_density,
                                    mean_noise_span_length,
                                    seed,
                                    skip_warmup
                                    ):
    assert noise_density is not None
    assert mean_noise_span_length is not None

    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(data_prefix[0],
                                                data_impl, splits_string,
                                                train_valid_test_num_samples,
                                                max_seq_length,
                                                noise_density,
                                                mean_noise_span_length,
                                                seed, skip_warmup
                                                )
    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(data_prefix,
                                                  train_valid_test_num_samples)
    prefixes, weights, datasets_train_valid_test_num_samples = output

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            prefixes[i], data_impl, splits_string,
            datasets_train_valid_test_num_samples[i],
            max_seq_length,
            noise_density,
            mean_noise_span_length,
            seed, skip_warmup)
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

        # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights)
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights)
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights)

    return (blending_train_dataset, blending_valid_dataset,
            blending_test_dataset)


def _build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                     train_valid_test_num_samples,
                                     max_seq_length,
                                     noise_density,
                                     mean_noise_span_length,
                                     seed,
                                     skip_warmup):
    """Build train, valid, and test datasets."""


    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0] - 1
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
        start_index = indexed_dataset.doc_idx[splits[index]]
        end_index = indexed_dataset.doc_idx[splits[index + 1]]
        print_rank_0('     sentence indices in [{}, {}) total of {} '
                     'sentences'.format(start_index, end_index,
                                        end_index - start_index))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            # Get the pointer to the original doc-idx so we can set it later.
            doc_idx_ptr = indexed_dataset.get_doc_idx()
            # Slice the doc-idx
            start_index = splits[index]
            # Add +1 so we can index into the dataset to get the upper bound.
            end_index = splits[index + 1] + 1
            # New doc_idx view.
            indexed_dataset.set_doc_idx(doc_idx_ptr[start_index:end_index])
            # Build the dataset accordingly.
            dataset = MLMDataset(
                    indexed_dataset=indexed_dataset,
                    noise_density=noise_density,
                    mean_noise_span_length=mean_noise_span_length,
                    name=name,
                    data_prefix=data_prefix,
                    sequence_length=max_seq_length,
                    seed=seed,
            )
            indexed_dataset.set_doc_idx(doc_idx_ptr)
            # Checks.
            # assert indexed_dataset.doc_idx[0] == 0
            # assert indexed_dataset.doc_idx.shape[0] == \
            #     (total_num_of_documents + 1)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


class MLMDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name,
        indexed_dataset,
        data_prefix,
        sequence_length,
        seed,
        noise_density = 0.15,
        mean_noise_span_length = 3
    ):

        # Params to store.
        self.name = name
        self.seed = seed
        self.sequence_length = sequence_length

        # Dataset.
        self.indexed_dataset = indexed_dataset

        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
        # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
        # according to `masked_lm_prob` and `max_ngrams`. We can also define the label length accordingly.
        number_of_raw_tokens, inputs_length, targets_length = compute_input_and_target_lengths(
            self.max_seq_length,
            self.noise_density,
            self.mean_noise_span_length
        )
        self.number_of_raw_tokens = number_of_raw_tokens
        self.inputs_length = inputs_length
        self.targets_length = targets_length

        # Build the samples mapping.
        self.shuffle_idx, self.samples_idx = get_samples_mapping(
            self.indexed_dataset,
            data_prefix,
            self.name,
            number_of_raw_tokens=number_of_raw_tokens,
            seed=seed
        )

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad
        self.bos_id = tokenizer.bos_token_id
        self.sep_id = tokenizer.sep_token_id
        self.sentinel_token_ids = tokenizer.additional_special_tokens_ids
        assert len(self.sentinel_tokens) > 0, "Provide the argument --vocab-extra-ids 100 to the script"

    def __len__(self):
        return len(self.samples_mapping)

    def __getitem__(self, idx):
        idx = self.shuffle_idx[idx]

        indices = self.samples_mapping[idx]
        sample = []
        for doc_idx, start_index, end_index in indices:
            sample.append(self.indexed_dataset.get(doc_idx)[start_index:end_index])

        return build_training_sample(
            sample,
            self.inputs_length,
            self.targets_length,
            self.number_of_raw_tokens,
            self.sentinel_tokens,
        )


def build_training_sample(
    sample,
    inputs_length,
    targets_length,
    num_noise_spans,
    all_sentinel_token_ids,
):
    """Build training sample.

    Arguments:
        TODO: Add description
    """

    # flatten sentences into one list
    tokens = [token for sentence in sample for token in sentence]

    spans_start, mask_indices = np.asarray([random_spans_noise_mask(
        inputs_length=inputs_length,
        targets_length=targets_length,
        num_noise_spans=num_noise_spans,
    )])
    spans_end = np.concatenate([
        spans_start[1:], np.full((1,), len(tokens), dtype=np.int32)]
    )
    labels_mask = ~mask_indices

    sentinel_token_ids = all_sentinel_token_ids[:len(spans_start[1::2])]

    input_token_ids = np.concatenate([
        elt
        for start, end, sentinel_token in zip(spans_start[::2], spans_end[::2], sentinel_token_ids)
        for elt in [tokens[start: end], np.full((1,), sentinel_token, dtype=np.int32)]
    ])
    target_token_ids = np.concatenate([
        elt
        for start, end, sentinel_token in zip(spans_start[1::2], spans_end[1::2], sentinel_token_ids)
        for elt in [np.full((1,), sentinel_token, dtype=np.int32), tokens[start: end]]
    ])

    return {
        'input_tokens': input_token_ids,
        'target_tokens': target_token_ids
    }


def get_samples_mapping(indexed_dataset, data_prefix, name, number_of_raw_tokens, seed):
    # Set random seed
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mapping
    indexmap_filename = f'{data_prefix}_{name}_mlm_dataset_indexmap_{number_of_raw_tokens}_rt_{seed}_seed.npy'
    shuffle_idx_filename = f"{data_prefix}_{name}_mlm_dataset_shuffle_idx_{number_of_raw_tokens}_rt_{seed}_seed.npy"
    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0 and \
       not os.path.isfile(indexmap_filename):
        print(' > WARNING: could not find index map file {}, building '
              'the indices on rank 0 ...'.format(indexmap_filename))

        # Make sure the types match the helpers input types.
        assert indexed_dataset.doc_idx.dtype == np.int64
        assert indexed_dataset.sizes.dtype == np.int32

        # Build samples mapping
        start_time = time.time()
        print_rank_0(' > building samples index mapping for {} ...'.format(
            name))
        samples_mapping = []
        sample_indices = []
        current_len = 0
        _idx = 0
        for doc_idx, sample_len in zip(indexed_dataset.doc_idx, indexed_dataset.sizes):
            # if document doesn't fit entirely.
            sample_left = sample_len
            while current_len + sample_left > number_of_raw_tokens:
                start_idx = sample_len - sample_left
                span_length = number_of_raw_tokens - current_len
                assert span_length <= number_of_raw_tokens
                sample_indices.append([doc_idx, start_idx, span_length + start_idx])
                samples_mapping.append(sample_indices)
                sample_indices = []
                current_len = 0
                sample_left -= span_length

            # If there's nothing else in the sample, we continue
            if sample_left == 0:
                continue

            # If there's something else in the sample, we add the rest as we know current_len + sample_left <= number_of_raw_tokens
            start_idx = sample_len - sample_left
            span_length = sample_left
            sample_indices.append([doc_idx, start_idx, start_idx + span_length])
            current_len += span_length
            assert span_length <= number_of_raw_tokens
            assert current_len <= number_of_raw_tokens
            # If we filled up a sample we add is to `samples_mapping` and empty the buffer.
            if current_len == number_of_raw_tokens:
                samples_mapping.append(sample_indices)
                sample_indices = []
                current_len = 0


        print_rank_0(' > done building sapmles index maping')
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        print_rank_0(' > saved the index mapping in {}'.format(
            indexmap_filename))

        # shuffle-idx.
        start_time = time.time()
        num_samples_ = len(samples_mapping)
        shuffle_idx = _build_shuffle_idx(num_samples_, np_rng)
        np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)

        # Make sure all the ranks have built the mapping
        print_rank_0(' > elasped time to build and save samples mapping '
                     '(seconds): {:4f}'.format(
                         time.time() - start_time))

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    # Load indexed dataset.
    print_rank_0(' > loading indexed mapping from {}'.format(
        indexmap_filename))
    start_time = time.time()
    samples_idx = np.load(indexmap_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        samples_idx.shape[0]))

    return shuffle_idx, samples_idx


def compute_input_and_target_lengths(sequence_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have SEP appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one SEP token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = sequence_length
    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)
    while inputs_length + targets_length > tokens_length:
        tokens_length -= 1
        inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1

    # tokens_length is the number of raw tokens we need to get
    # inputs_length will be the input
    # targets_length will be the target
    return tokens_length, inputs_length, targets_length


# TODO @thomasw21 handle random state correctly.
def random_spans_noise_mask(
    inputs_length,
    targets_length,
    num_noise_spans,
):

    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        inputs_length: int32 scalar
        targets_length: int32 scalar
        num_noise_spans: int32 scalar
    Returns:
        a int8 tensor with shape [num_noise_spans]
        a boolean tensor with shape [length]
    """
    # pick the lengths of the noise spans and the non-noise spans
    num_noise_tokens = targets_length - num_noise_spans - 1
    num_nonnoise_tokens = inputs_length - num_noise_tokens - 1
    number_of_raw_tokens = num_noise_tokens + num_nonnoise_tokens

    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]], constant_values=0)
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((number_of_raw_tokens,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return span_starts, is_noise

def _build_shuffle_idx(num_samples, np_rng):
    """Build the range [0, size) and shuffle."""
    print(f' > building shuffle index with split [0, {num_samples}) '
          f'...', flush=True)

    dtype_ = np.uint32
    if num_samples >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                  step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    return shuffle_idx_first
