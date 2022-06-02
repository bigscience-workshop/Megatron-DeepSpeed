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
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples, get_samples_mapping, create_masked_lm_predictions
from megatron.data.dataset_utils import get_train_valid_test_split_, get_split_by_range_, get_indexed_dataset_
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset


def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    max_seq_length,
                                    masked_lm_prob, short_seq_prob, seed,
                                    skip_warmup, binary_head=False,
                                    max_seq_length_dec=None,
                                    dataset_type='standard_bert'):
    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(data_prefix[0],
                                                data_impl, splits_string,
                                                train_valid_test_num_samples,
                                                max_seq_length, masked_lm_prob,
                                                short_seq_prob, seed,
                                                skip_warmup,
                                                binary_head,
                                                max_seq_length_dec,
                                                dataset_type=dataset_type)
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
            max_seq_length, masked_lm_prob, short_seq_prob,
            seed, skip_warmup, binary_head, dataset_type=dataset_type)
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
                                     masked_lm_prob, short_seq_prob, seed,
                                     skip_warmup, binary_head,
                                     max_seq_length_dec,
                                     dataset_type='standard_bert'):
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
            kwargs = dict(
                name=name,
                data_prefix=data_prefix,
                num_epochs=None,
                max_num_samples=train_valid_test_num_samples[index],
                max_seq_length=max_seq_length,
                seed=seed,
            )
            dataset = NonCausalMLMDataset(
                    indexed_dataset=indexed_dataset,
                    masked_lm_prob=masked_lm_prob,
                    max_seq_length_dec=max_seq_length_dec,
                    short_seq_prob=short_seq_prob,
                    **kwargs
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


class NonCausalMLMDataset(torch.utils.data.Dataset):

    def __init__(self, name, indexed_dataset, data_prefix,
                 num_epochs, max_num_samples, masked_lm_prob,
                 max_seq_length, max_seq_length_dec,
                 short_seq_prob, seed):

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.max_seq_length_dec = max_seq_length_dec

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        self.samples_mapping = get_samples_mapping(self.indexed_dataset,
                                                   data_prefix,
                                                   num_epochs,
                                                   max_num_samples,
                                                   self.max_seq_length - 2, # account for added tokens
                                                   short_seq_prob,
                                                   self.seed,
                                                   self.name,
                                                   False)

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.sentinel_tokens = tokenizer.additional_special_tokens_ids
        assert len(self.sentinel_tokens) > 0, "Provide the argument --vocab-extra-ids 100 to the script"

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):

        start_index, end_index, seq_length = self.samples_mapping[idx]
        sample = []
        for index in range(start_index, end_index):
            sample.append(self.indexed_dataset[index])

        #concat more to avoid padding
        while seq_length < (self.max_seq_length/self.masked_lm_prob):
            _idx = random.randint(idx, self.__len__())
            start_index, end_index, _seq_length = self.samples_mapping[_idx]
            for index in range(start_index, end_index):
                sample.append(self.indexed_dataset[index])

            seq_length += _seq_length

        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        return build_training_sample(sample, seq_length,
                                     self.max_seq_length,  # needed for padding
                                     self.max_seq_length_dec,
                                     self.vocab_id_list,
                                     self.vocab_id_to_token_dict,
                                     self.cls_id, self.sep_id,
                                     self.mask_id, self.pad_id,
                                     self.masked_lm_prob, np_rng,
                                     self.bos_id, self.eos_id,
                                     self.sentinel_tokens)


def build_training_sample(sample, target_seq_length,
                          max_seq_length, max_seq_length_dec,
                          vocab_id_list, vocab_id_to_token_dict,
                          cls_id, sep_id, mask_id, pad_id,
                          masked_lm_prob, np_rng, bos_id=None,
                          eos_id=None, sentinel_tokens=None):
    """Build training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
        bos_id: start of decoder example id
        eos_id: end of generation id
        sentinel_tokens: unique value to be substituted for every replaced span
    """

    # assert target_seq_length <= max_seq_length

    # flatten sentences into one list
    tokens = [token for sentence in sample for token in sentence]

    # # Truncate to `target_sequence_length`.
    # max_num_tokens = target_seq_length
    # truncated = len(tokens) > max_num_tokens
    # tokens = tokens[:max_num_tokens]

    max_ngrams = 3
    # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
    # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
    # according to `masked_lm_prob` and `max_ngrams`. We can also define the label length accordingly.
    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        max_seq_length,
        masked_lm_prob,
        max_ngrams
        )

    mask_indices = np.asarray([random_spans_noise_mask(expanded_inputs_length)])
    labels_mask = ~mask_indices
    
    input_ids_sentinel = create_sentinel_ids(mask_indices.astype(np.int8), vocab_len=len(vocab_id_list))
    labels_sentinel = create_sentinel_ids(labels_mask.astype(np.int8), vocab_len=len(vocab_id_list))


    if len(tokens) <= expanded_inputs_length:
        tokens = pad_and_convert_to_numpy(
            tokens,
            pad_id,
            expanded_inputs_length
            )
    else:
        tokens = tokens[:expanded_inputs_length]

    tokens = np.asarray([tokens])
    input_tokens_ids = filter_input_ids(tokens, input_ids_sentinel, eos_id)[0]
    output_tokens_ids = filter_input_ids(tokens, labels_sentinel, eos_id)[0]

    # # Masking.
    # max_predictions_per_seq = masked_lm_prob * max_num_tokens
    # (tokens, masked_positions, masked_labels, _, masked_spans) = create_masked_lm_predictions(
    #     tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
    #     cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng,
    #     max_ngrams=max_ngrams, geometric_dist=True, masking_style="t5")

    # sentinel_tokens = collections.deque(sentinel_tokens)
    # input_tokens_ids = []
    # output_tokens_ids = [] #[bos_id]
    # (start_index, end_index) = (0, None)
    # for span in masked_spans:
    #     flag = sentinel_tokens.popleft()

    #     output_tokens_ids.append(flag)
    #     output_tokens_ids.extend(span.label)

    #     end_index = span.index[0]
    #     input_tokens_ids.extend(tokens[start_index: end_index])
    #     input_tokens_ids.append(flag)

    #     # the next start index is the token after the last span token
    #     start_index = span.index[-1] + 1


    # # Add the remaining tokens to input_tokens_ids
    # input_tokens_ids.extend(tokens[start_index:])
    # input_tokens_ids.append(eos_id)
    # # Add <eos> token to the output_tokens_ids
    # output_tokens_ids.append(eos_id)
    
    # text_tokens_ids = pad_and_convert_to_numpy(
    #     input_tokens_ids+output_tokens_ids,
    #     pad_id,
    #     max_seq_length+max_seq_length_dec
    #     )

    text_tokens_ids = np.concatenate((input_tokens_ids, output_tokens_ids))

    prefix_len = len(input_tokens_ids)

    return {
        'text': text_tokens_ids,
        'prefix_len': prefix_len
    }


def pad_and_convert_to_numpy(tokens, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0

    # Tokens and token types.
    filler = np.array([pad_id] * padding_length)
    tokens_np = np.concatenate((tokens, filler), dtype=np.int64)

    return tokens_np


def create_sentinel_ids(mask_indices, vocab_len):
    """
    Sentinel ids creation given the indices that should be masked.
    The start indices of each mask are replaced by the sentinel ids in increasing
    order. Consecutive mask indices to be deleted are replaced with `-1`.
    """
    start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
    start_indices[:, 0] = mask_indices[:, 0]

    sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
    sentinel_ids = np.where(sentinel_ids != 0, (vocab_len - sentinel_ids), 0)
    sentinel_ids -= mask_indices - start_indices

    return sentinel_ids


def filter_input_ids(input_ids, sentinel_ids, eos_id):
    """
    Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
    This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
    """
    batch_size = input_ids.shape[0]

    input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
    # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
    # masked tokens coming after sentinel tokens and should be removed
    input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
    input_ids = np.concatenate(
        [input_ids, np.full((batch_size, 1), eos_id, dtype=np.int32)], axis=-1
    )
    return input_ids


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
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
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


def random_spans_noise_mask(
    length,
    noise_density=0.15,
    mean_noise_span_length=3
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
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
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
        first_in_segment = np.pad(mask_indices, [[1, 0]])
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
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]