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

"""Processing data for pretraining.

This builds data files from a source HuggingFace dataset, e.g,

  from datasets import load_dataset
  dset = load_dataset('openwebtext')

The implementation requires MPI and mpi4py, and it assumes that
files are written to a global file system, such that one process
can read a file written by another process.

A list of sample index values for the source dataset are shuffled by rank 0,
and the shuffled sequence is then broadcast to all ranks.
Each process tokenizes a subset of samples and writes its output to a part file.
After all ranks have finished, rank 0 merges and deletes the part files.

To run:

mpiexec -np 320 python preprocess_dataset_mpi.py \
       --input openwebtext \
       --output-prefix openwebtext-bert \
       --vocab bert-large-uncased-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
"""

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from datasets import load_dataset, logging

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from megatron.data.indexed_dataset import infer_dataset_impl, MMapIndexedDataset, data_file_path, index_file_path

import random
from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# compute start and end index values based on rank
# evenly divide file count among procs
def get_start_end(num, rank, num_ranks):
    """Compute start and end index values to evenly divide num items
    among ranks.

    If num is not evenly divisible by num_ranks, ranks from
    [0,remainder) will each be assigned one extra item.
    Returns a (start, end) tuple, such that the calling rank
    should take items in a list[start:end]

    Parameters
    ----------
    num : int
      Number of items to be divided
    rank : int
      Rank of the calling process
    num_ranks : int
      Number of processes among which to divide items

    Returns
    -------
    int
      start index value
    int
      end index value
    """

    num_per_rank = num // num_ranks
    remainder = num - num_per_rank * num_ranks
    if rank < remainder:
        start = (num_per_rank + 1) * rank;
        end = start + (num_per_rank+1)
    else:
        start = (num_per_rank + 1) * remainder + num_per_rank * (rank - remainder);
        end = start + num_per_rank
    if end > num:
        end = num
    return start, end

# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def encode_text(self, text):
        ids = {}
        for key in ['text']:
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text):
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0:
                if self.args.append_eod:
                    doc_ids[-1].append(Encoder.tokenizer.eod)
                ids[key] = doc_ids
        return ids, len(text)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Dataset name')
    group.add_argument('--columns', nargs='+', default=['text'],
                       help='space separate listed of column names to extract from dataset')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group.add_argument('--shuffle', action='store_true',
                       help='Shuffle samples before writing output files.')
    group.add_argument('--seed', type=int, default=None,
                       help='Seed to pass to random.seed for shuffle operations.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')


    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args

def main():
    args = get_args()
    startup_start = time.time()

    # typically: ['text']
    columns = args.columns

    # some functions like build_tokenizer use args.rank to filter stdout messages
    args.rank = mpi_rank

    # silence info messages from all procs except rank 0 
    if mpi_rank != 0:
        logging.set_verbosity(logging.ERROR)

    # load the specified HuggingFace dataset and get its size
    dsetname = args.input
    if mpi_rank == 0:
        print("Opening dataset", dsetname)
    dset = load_dataset(dsetname, split="train", keep_in_memory=None)
    dset_size = len(dset)
    mpi_comm.barrier()
    if mpi_rank == 0:
        print(dset)

    # create sample index list on rank 0,
    # optionally shuffle the list,
    # and bcast to all ranks
    idx = []
    if mpi_rank == 0:
        idx = [int(x) for x in range(dset_size)]
        if args.shuffle:
            if args.seed is not None:
                random.seed(args.seed)
            random.shuffle(idx)
    idx = mpi_comm.bcast(idx, root=0)
    
    # divide index list evenly among ranks
    start, end = get_start_end(len(idx), mpi_rank, mpi_size)

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    # TODO: skip write phase and report success if rank has nothing to do

    # create data file for each rank
    if mpi_rank == 0:
        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in columns:
        filebase = "{}_{}_{}_{}".format(args.output_prefix, key, level, mpi_rank)
        output_bin_files[key] = data_file_path(filebase)
        output_idx_files[key] = index_file_path(filebase)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                               impl=args.dataset_impl,
                                               vocab_size=tokenizer.vocab_size)

    # wait for all ranks to create file before stopping timer
    mpi_comm.barrier()
    startup_end = time.time()
    if mpi_rank == 0:
        print("Seconds to startup:", startup_end - startup_start)

    # each rank tokenizes its samples and writes its own file
    proc_start = time.time()
    count = 0
    total_bytes_processed = 0
    encoder.initializer()
    for i in range(start, end):
        # tokenize text for the given sample index
        j = idx[i]
        text = dset[j]['text']
        doc, bytes_processed = encoder.encode_text(text)

        # add tokenized sequence to our data file
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()

    # finalize file of each rank
    for key in columns:
        builders[key].finalize(output_idx_files[key])
        del builders[key] # file closed in __del__

    # wait for all ranks to finish their files
    mpi_comm.barrier()
    proc_end = time.time()
    if mpi_rank == 0:
        print("Seconds to tokenize:", proc_end - proc_start)
        print("Documents:", dset_size, "docs/sec: ", dset_size / (proc_end - proc_start))

    # TODO: allreduce to ensure all ranks wrote their part successfully

    # rank 0 merges and deletes all per-rank files
    if mpi_rank == 0:
        print("Merging rank files ...", flush=True)
        merge_start = time.time()

        # define name of single file
        for key in columns:
            filebase = "{}_{}_{}".format(args.output_prefix, key, level)
            output_bin_files[key] = data_file_path(filebase)
            output_idx_files[key] = index_file_path(filebase)
            builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                   impl=args.dataset_impl,
                                                   vocab_size=tokenizer.vocab_size)

        # merge all ranks into one file
        for rank in range(mpi_size):
            infile = "{}_{}_{}_{}".format(args.output_prefix, key, level, rank)
            print("Merging file", infile, flush=True)
            builders[key].merge_file_(infile)

        # finalize the merged file
        print("Finalizing merged file ...", flush=True)
        for key in columns:
            builders[key].finalize(output_idx_files[key])
            del builders[key] # file closed in __del__

        # delete per ranks files
        print("Deleting rank files ...", flush=True)
        for rank in range(mpi_size):
            filebase = "{}_{}_{}_{}".format(args.output_prefix, key, level, rank)
            binfile = data_file_path(filebase)
            idxfile = index_file_path(filebase)
            os.remove(binfile)
            os.remove(idxfile)

        merge_end = time.time()
        print("Seconds to merge:", merge_end - merge_start)
        print(f"Merged {mpi_size} datasets to {args.output_prefix}")

    # hold everyone until rank 0 is done
    mpi_comm.barrier()

if __name__ == '__main__':
    main()
