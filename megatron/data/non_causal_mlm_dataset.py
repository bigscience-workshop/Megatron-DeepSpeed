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

    assert target_seq_length <= max_seq_length

    # flatten sentences into one list
    tokens = [token for sentence in sample for token in sentence]

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    truncated = len(tokens) > max_num_tokens
    tokens = tokens[:max_num_tokens]

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masked_positions, masked_labels, _, masked_spans) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng,
        max_ngrams=10, geometric_dist=True, masking_style="t5")

    sentinel_tokens = collections.deque(sentinel_tokens)
    input_tokens_ids = []
    output_tokens_ids = [] #[bos_id]
    (start_index, end_index) = (0, None)
    for span in masked_spans:
        flag = sentinel_tokens.popleft()

        output_tokens_ids.append(flag)
        output_tokens_ids.extend(span.label)

        end_index = span.index[0]
        input_tokens_ids.extend(tokens[start_index: end_index])
        input_tokens_ids.append(flag)

        # the next start index is the token after the last span token
        start_index = span.index[-1] + 1


    # Add the remaining tokens to input_tokens_ids
    input_tokens_ids.extend(tokens[start_index:])
    # Add <eos> token to the output_tokens_ids
    output_tokens_ids.append(eos_id)
    prefix_len = len(input_tokens_ids)

    # # Padding.
    # input_tokens_ids, _, output_tokens_ids, enc_mask, \
    # dec_mask, enc_dec_mask, loss_mask \
    #     = pad_and_convert_to_numpy(tokens, masked_positions,
    #                                masked_labels, pad_id, max_seq_length,
    #                                max_seq_length_dec, masked_spans,
    #                                bos_id, eos_id, sentinel_tokens)

    # text_tokens_ids = np.array(input_tokens_ids+output_tokens_ids)

    # text_tokens_ids = input_tokens_ids + output_tokens_ids
    print("input_tokens_ids")
    print(len(input_tokens_ids))
    print(input_tokens_ids)
    print("output_tokens_ids")
    print(len(output_tokens_ids))
    print(output_tokens_ids)
    # print("text_tokens_ids")
    # # print(text_tokens_ids)
    # print(len(text_tokens_ids))

    # input_tokens_ids = pad_and_convert_to_numpy(
    #     input_tokens_ids,
    #     self.tokenizer.pad,
    #     self.seq_length
    #     )

    import sys
    sys.exit()

    return {
        'text': input_tokens_ids,
        'prefix_len': prefix_len
    }


    # train_sample = {
    #     'text_enc': tokens_enc,
    #     'text_dec': tokens_dec_in,
    #     'labels': labels,
    #     'loss_mask': loss_mask,
    #     'truncated': int(truncated),
    #     'enc_mask': enc_mask,
    #     'dec_mask': dec_mask,
    #     'enc_dec_mask': enc_dec_mask,
    # }
    # return train_sample


def pad_and_convert_to_numpy(tokens, masked_positions,
                             masked_labels, pad_id,
                             max_seq_length, max_seq_length_dec,
                             masked_spans=None, bos_id=None,
                             eos_id=None, sentinel_tokens=None):
    """Pad sequences and convert them to numpy."""

    sentinel_tokens = collections.deque(sentinel_tokens)
    t5_input = []
    (t5_decoder_in, t5_decoder_out) = ([bos_id], [])
    (start_index, end_index) = (0, None)
    for span in masked_spans:
        flag = sentinel_tokens.popleft()

        # Append the same tokens in decoder input and output
        t5_decoder_in.append(flag)
        t5_decoder_in.extend(span.label)
        t5_decoder_out.append(flag)
        t5_decoder_out.extend(span.label)

        end_index = span.index[0]
        t5_input.extend(tokens[start_index: end_index])
        t5_input.append(flag)

        # the next start index is the token after the last span token
        start_index = span.index[-1] + 1

    # Add <eos> token to the t5_decoder_out
    t5_decoder_out.append(eos_id)

    # Add the remaining tokens to the t5 input
    t5_input.extend(tokens[start_index:])

    # assert (len(t5_input) - len(masked_spans)) + \
    #        (len(t5_decoder_in) - (len(masked_spans) + 1)) == len(tokens)

    # Some checks.

    # Encoder-side padding mask.
    num_tokens = len(t5_input)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(masked_positions) == len(masked_labels)

    # Tokens..
    filler = [pad_id] * padding_length
    tokens_enc = np.array(t5_input + filler, dtype=np.int64)

    # Decoder-side padding mask.
    num_tokens_dec = len(t5_decoder_in)
    padding_length_dec = max_seq_length_dec - num_tokens_dec
    assert padding_length_dec >= 0
    filler_dec = [pad_id] * padding_length_dec
    tokens_dec_in = np.array(t5_decoder_in + filler_dec, dtype=np.int64)

    # Create attention masks
    enc_mask = make_attention_mask(tokens_enc, tokens_enc)
    enc_dec_mask = make_attention_mask(tokens_dec_in, tokens_enc)
    dec_mask = make_attention_mask(tokens_dec_in, tokens_dec_in)
    dec_mask = dec_mask * make_history_mask(tokens_dec_in)

    # Labels mask.
    labels = t5_decoder_out + ([-1] * padding_length_dec)
    labels = np.array(labels, dtype=np.int64)

    # Loss mask
    loss_mask = ([1] * num_tokens_dec) + ([0] * padding_length_dec)
    loss_mask = np.array(loss_mask, dtype=np.int64)

    return tokens_enc, tokens_dec_in, labels, enc_mask, \
           dec_mask, enc_dec_mask, loss_mask


def make_attention_mask(source_block, target_block):
    """
    Returns a 2-dimensional (2-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
    mask = mask.astype(np.int64)
    # (source_length, target_length)
    return mask


def make_attention_mask_3d(source_block, target_block):
    """
    Returns a 3-dimensional (3-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[:, None, :] >= 1) * (source_block[:, :, None] >= 1)
    # (batch, source_length, target_length)
    # mask = mask.astype(np.int64)
    return mask


def make_history_mask(block):
    length = block.shape[0]
    arange = np.arange(length)
    history_mask = (arange[None, ] <= arange[:, None])
    history_mask = history_mask.astype(np.int64)
    return history_mask


def make_history_mask_3d(block):
    batch, length = block.shape
    arange = torch.arange(length, device=block.device)
    history_mask = (arange[None, ] <= arange[:, None])[None, ]
    history_mask = history_mask.expand(batch, length, length)
    return history_mask
