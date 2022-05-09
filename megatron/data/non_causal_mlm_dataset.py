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
import collections

import numpy as np
import torch

from megatron import mpu, print_rank_0, get_tokenizer
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from megatron.data.dataset_utils import get_train_valid_test_split_, get_split_by_range_
from megatron.data.dataset_utils import create_masked_lm_predictions, get_samples_mapping
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset


def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    seq_length, seed, skip_warmup):
    """Build train, valid, and test datasets."""

    # Single dataset.
    if len(data_prefix) == 1:
        all_train_datasets, all_valid_datasets, all_test_datasets =  _build_train_valid_test_datasets(data_prefix[0],
                                                data_impl, splits_string,
                                                train_valid_test_num_samples,
                                                seq_length, seed, skip_warmup)
    # Blending dataset.
    else:

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
                                            seq_length, seed, skip_warmup)
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        all_train_datasets = BlendableDataset(train_datasets, weights) \
                            if train_datasets else None
        all_valid_datasets = BlendableDataset(valid_datasets, weights) \
                            if valid_datasets else None
        all_test_datasets = BlendableDataset(test_datasets, weights) \
                            if test_datasets else None

    return all_train_datasets, all_valid_datasets, all_test_datasets


def build_dataset_group(dataset_group_name, paths, weights, splits, data_impl,
                        train_valid_test_num_samples,
                        seq_length, seed, skip_warmup, train_valid_test):
    '''
    Build a single dataset group corresponding to Option 2 of data loading see arguments.py
    a dataset group is passed on the following form
    GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT2 START:END PATH2
    or alternatively
    GIVEN_NAME PATH1    # for a single dataset to be used fully
    '''

    assert train_valid_test in ["train","valid","test"]

    # Single dataset.
    if len(paths) == 1:
        dataset =  _build_single_datasets(paths[0],
                                          splits[0],
                                          data_impl,
                                          train_valid_test_num_samples,
                                          seq_length, seed, skip_warmup,
                                          dataset_group_name, train_valid_test)
        return dataset
    # Blending dataset.
    else:

        data_prefix = []
        # data_prefix is on the shape:
        # ["WEIGHT1", "PATH1", "WEIGHT2", "PATH2", "WEIGHT3", "PATH3"]
        for w,p in zip(weights, paths):
            data_prefix += [w,p]

        output = get_datasets_weights_and_num_samples(data_prefix,
                                                    train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_single_datasets(prefixes[i],
                                        splits[i],
                                        data_impl,
                                        datasets_train_valid_test_num_samples[i],
                                        seq_length,
                                        seed, skip_warmup,
                                        dataset_group_name, train_valid_test)

            datasets.append(ds)
        all_datasets = BlendableDataset(datasets, weights)

        return all_datasets

def _build_single_datasets(data_prefix, range_string, data_impl, train_valid_test_num_samples,
                            seq_length, seed, skip_warmup, dataset_group_name, train_valid_test):
    """Build a single dataset"""

    assert train_valid_test in ["train","valid","test"]
    index = ["train","valid","test"].index(train_valid_test)

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    # this corresponds to option2 for data loading on the form
    # WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT3 START:END PATH3
    # splits here is an array of size 2  [start_index, end_index]
    splits = get_split_by_range_(range_string=range_string, size=total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    print_rank_0('    {}:'.format(dataset_group_name))
    print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[0], splits[1],
                                        splits[1] - splits[0]))

    def build_dataset(name):
        dataset = None
        if splits[1] > splits[0]:
            documents = np.arange(start=splits[0], stop=splits[1],
                                  step=1, dtype=np.int32)
            dataset = NonCausalMLMDataset(name, data_prefix,
                                  documents, indexed_dataset,
                                  train_valid_test_num_samples[index],
                                  seq_length, seed)
        return dataset

    dataset = build_dataset(dataset_group_name)

    return dataset


def _build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                     train_valid_test_num_samples,
                                     seq_length, seed, skip_warmup):
    """Build train, valid, and test datasets."""


    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    # splits here is an array of size 4  [train_start_index, valid_start_index, test_start_index, test_end_index]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = NonCausalMLMDataset(name, data_prefix,
                                  documents, indexed_dataset,
                                  train_valid_test_num_samples[index],
                                  seq_length, seed)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(path, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(' > building dataset index ...')
    start_time = time.time()
    indexed_dataset = make_indexed_dataset(path,
                                           data_impl,
                                           skip_warmup)
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print_rank_0('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset


class NonCausalMLMDataset(torch.utils.data.Dataset):

    def __init__(self, name, indexed_dataset, data_prefix,
                 num_epochs, max_num_samples, masked_lm_prob,
                 max_seq_length,
                 short_seq_prob, seed):

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length

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
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        return build_training_sample(sample,
                                     self.max_seq_length,  # needed for padding
                                     self.vocab_id_list,
                                     self.vocab_id_to_token_dict,
                                     self.cls_id, self.sep_id,
                                     self.mask_id, self.pad_id,
                                     self.masked_lm_prob, np_rng,
                                     self.bos_id, self.eos_id,
                                     self.sentinel_tokens)


def build_training_sample(sample,
                          max_seq_length,
                          vocab_id_list, vocab_id_to_token_dict,
                          cls_id, sep_id, mask_id, pad_id,
                          masked_lm_prob, np_rng, bos_id=None,
                          eos_id=None, sentinel_tokens=None):
    """Build training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
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

    # flatten sentences into one list
    tokens = [token for sentence in sample for token in sentence]

    # Truncate to `target_sequence_length`.
    max_num_tokens = max_seq_length
    truncated = len(tokens) > max_num_tokens
    tokens = tokens[:max_num_tokens]

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masks, labels, _, masked_spans) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng
        )

    # Padding.
    padded_tokens = pad_and_convert_to_numpy(tokens, max_seq_length)
    padded_labels = pad_and_convert_to_numpy(labels, max_seq_length)
    padded_masks = pad_and_convert_to_numpy(masks, max_seq_length)

    print(padded_tokens)
    print(padded_labels)
    import sys
    sys.exit()

    train_sample = {
        'text': padded_tokens,
        'labels': padded_labels,
        'mask': padded_masks,
        'prefix_len': 0
    }
    return train_sample


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