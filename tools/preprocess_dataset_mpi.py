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

The implementation can use `mpi4py` or `torch.distributed` for node communication, and it assumes that
files are written to a global file system, such that one process
can read a file written by another process.

A list of sample index values from the source dataset are selected
by rank 0 and broadcast to all ranks.
Each process tokenizes a subset of samples and writes its output to a part file.
After all ranks have finished, rank 0 merges and deletes the part files.

To run:

mpiexec -np 320 python preprocess_dataset_mpi.py \
       --input openwebtext \
       --shuffle \
       --seed 100 \
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
import stat
import time

import numpy as np
import random

import torch
import torch.distributed as dist
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data.indexed_dataset import data_file_path, index_file_path, make_builder

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

        self.tokenizer = build_tokenizer(self.args)

        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                self.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                self.splitter = splitter
        else:
            self.splitter = IdentitySplitter()

    def encode_text(self, text):
        ids = {}
        for key in self.args.columns:
            doc_ids = []
            for sentence in self.splitter.tokenize(text):
                sentence_ids = self.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0:
                if self.args.append_eod:
                    doc_ids[-1].append(self.tokenizer.eod)
                ids[key] = doc_ids
        return ids, len(text)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Dataset name')
    group.add_argument('--split', type=str, default='train',
                       help='Dataset split to select.')
    group.add_argument('--columns', nargs='+', default=['text'],
                       help='Space separate listed of column names to extract from dataset')
    group.add_argument('--count', type=int, default=None,
                       help='Limit the number of samples to select.')
    group.add_argument('--shuffle', action='store_true',
                       help='Shuffle samples before writing output files.')
    group.add_argument('--seed', type=int, default=None,
                       help='Seed to pass to random.seed for shuffle operations.')
    group.add_argument('--download', action='store_true',
                       help='Enable dataset download if not already cached.')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

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

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--mpi4py', action='store_true',
                       help='Assume script has been launched as an MPI job, and use MPI for communication.')
    group.add_argument('--torch-backend', type=str, default='gloo', choices = ['gloo', 'mpi'],
                       help='Select torch.distributed backend.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='Local rank of calling process on its node (from torch.distributed.launch).')
    group.add_argument('--log-interval', type=int, default=30,
                       help='Seconds between progress updates (0 to disable)')

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    # use mpi4py instead of torch.distributed if requested
    args.use_mpi = False
    if args.mpi4py:
        try:
            from mpi4py import MPI
            args.MPI = MPI
            args.use_mpi = True
        except:
            print(f"ERROR: mpi4py requested, but failed to import, falling back to torch.distributed.")

    return args

def init_distributed(args):
    """Determine which distributed runtime to use and connect up processes"""
    # select our distributed runtime (MPI or torch.distributed)
    # lookup our process rank and the group size
    # some functions like build_tokenizer use args.rank to filter stdout messages
    if args.use_mpi:
        args.mpi_comm = args.MPI.COMM_WORLD
        args.rank = args.mpi_comm.Get_rank()
        args.numranks = args.mpi_comm.Get_size()
    else:
        dist.init_process_group(args.torch_backend, init_method="env://")
        args.rank = dist.get_rank()
        args.numranks = dist.get_world_size()

def barrier(args):
    """Globally synchronize all processes."""
    if args.use_mpi:
        args.mpi_comm.barrier()
    else:
        dist.barrier()

def bcast(args, vals, root=0):
    """Broadcast list of vals from root to all ranks, returns newly allocated list"""
    if args.use_mpi:
        vals = args.mpi_comm.bcast(vals, root=root)
        return vals
    else:
        # broadcast length of vals list
        length = [len(vals)]
        dist.broadcast_object_list(length, src=root)

        # allocate a tensor of appropriate size
        # initialize tensor with list values on root
        if args.rank == root:
            tvals = torch.tensor(vals, dtype=torch.int64)
        else:
            tvals = torch.zeros(length[0], dtype=torch.int64)

        # broadcast tensor from root, and return as a new list
        dist.broadcast(tvals, src=root)
        return tvals.tolist()

def all_sum_(args, vals):
    """Sums values in vals element-wise and updates vals with final result on all ranks"""
    if args.use_mpi:
        outval = np.zeros_like(vals)
        args.mpi_comm.Allreduce(vals, outval, op=args.MPI.SUM)
        vals[:] = outval
    else:
        tensor = torch.from_numpy(vals)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

def all_true(args, val):
    """Returns True if all procs input True, False otherwise"""
    if args.use_mpi:
        inval = np.array([val], dtype=np.bool_)
        outval = np.zeros_like(inval)
        args.mpi_comm.Allreduce(inval, outval, op=args.MPI.LAND)
        return bool(outval[0])
    else:
        tensor = torch.tensor([int(val)], dtype=torch.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.BAND)
        return bool(tensor[0])

def load_dset(args):
    # Avoid downloading datasets unless explicitly requested.
    # To disable downloads, we set $HF_DATASETS_OFFLINE=1.
    # The HF_DATASETS_OFFLINE environment variable is processed when datasets is imported,
    # so we must set it before the import statement.
    # We allow the user to override default behavior if they explictly set $HF_DATASETS_OFFLINE.
    if not args.download and 'HF_DATASETS_OFFLINE' not in os.environ:
        # sets HF_DATASETS_OFFLINE within the environment of this script
        os.environ['HF_DATASETS_OFFLINE'] = "1"

        # Alternatively, one could set datasets.config.HF_DATASETS_OFFLINE.
        # This seems to work even after the import statement,
        # but it feels like a bigger hack.
        #datasets.config.HF_DATASETS_OFFLINE = 1

    # import datasets after potentially setting environment variables
    from datasets import load_dataset, logging
    from datasets.utils.file_utils import OfflineModeIsEnabled

    # silence info messages from all procs except rank 0 
    if args.rank != 0:
        logging.set_verbosity(logging.ERROR)

    # Load the specified HuggingFace dataset.
    # Give rank 0 a head start in case the dataset is not already cached.
    success = True
    dsetname = args.input
    if args.rank == 0:
        print(f"Opening dataset {dsetname}")
        try:
            dset = load_dataset(dsetname, split=args.split, keep_in_memory=None)
        except OfflineModeIsEnabled:
            print(f"ERROR: Cannot download {dsetname} since running in offline mode, force with --download")
            success = False
        except:
            print("ERROR: Unexpected error:", sys.exc_info()[0])
            success = False

    # determine whether rank 0 succeeded in loading the dataset
    success = all_true(args, success)
    if not success:
        return None

    # Rank 0 succeeded, attempt to load dataset on all other ranks.
    # This should load from cache now.
    if args.rank != 0:
        try:
            dset = load_dataset(dsetname, split=args.split, keep_in_memory=None)
        except:
            # this print might be noisy, but better than nothing
            print("ERROR: Unexpected error:", sys.exc_info()[0])
            success = False

    # verify that all ranks loaded the dataset
    success = all_true(args, success)
    if not success:
        if args.rank == 0:
            print(f"ERROR: At least one process failed to load {dsetname}")
        return None

    return dset

def select_sample_list(args, dset_size):
    """Given the total number of samples, select a list of sample index values"""
    # create sample index list on rank 0,
    # optionally shuffle the list,
    # and optionally limit the sample count
    idx = []
    if args.rank == 0:
        # generate a list of all index values
        idx = list(range(dset_size))

        # optionally shuffle
        if args.shuffle:
            if args.seed is not None:
                random.seed(args.seed)
            random.shuffle(idx)

        # optionally limit the sample count
        if args.count is not None:
            idx = idx[:args.count]

    # broadcast sample index values from rank 0 to all procs
    idx = bcast(args, idx, root=0)
    return idx

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
    remainder = num % num_ranks
    if rank < remainder:
        start = (num_per_rank + 1) * rank;
        end = start + (num_per_rank + 1)
    else:
        start = (num_per_rank + 1) * remainder + num_per_rank * (rank - remainder);
        end = start + num_per_rank
    return start, end

def rank_files_write(args, dset, idx, encoder):
    tokenize_start = time.time()

    # we'll total up the number of docs, sentences, and bytes
    # processed across all ranks
    dset_stats = np.zeros(3, dtype=np.int64) # docs, sentences, bytes

    # we'll set this to false on any problem
    success = True

    try:
        # create data file for each rank
        if args.rank == 0:
            print(f"Vocab size: {args.vocab_size}")
            print(f"Output prefix: {args.output_prefix}")
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in args.columns:
            filebase = f"{args.output_prefix}_{key}_{args.level}_{args.rank}"
            output_bin_files[key] = data_file_path(filebase)
            output_idx_files[key] = index_file_path(filebase)
            builders[key] = make_builder(output_bin_files[key],
                                         impl=args.dataset_impl,
                                         vocab_size=args.vocab_size)

        # divide index list evenly among ranks
        idx_start, idx_end = get_start_end(len(idx), args.rank, args.numranks)

        # each rank tokenizes its samples and writes its own file
        progress_next = time.time() + float(args.log_interval)
        for i in idx[idx_start:idx_end]:
            for key in args.columns:
                # tokenize text for the given sample index
                text = dset[i][key]
                doc, bytes_processed = encoder.encode_text(text)

                # add tokenized sequence to our data file
                for key, sentences in doc.items():
                    for sentence in sentences:
                        builders[key].add_item(torch.IntTensor(sentence))
                    builders[key].end_document()
                    dset_stats[0] += 1
                    dset_stats[1] += len(sentences)
                dset_stats[2] += bytes_processed

            if args.rank == 0 and args.log_interval > 0 and time.time() > progress_next:
                current = time.time()
                progress_next = current + float(args.log_interval)

                elapsed = current - tokenize_start
                timestamp = time.strftime("%Y/%m/%dT%H:%M:%S")
                docs = dset_stats[0] * args.numranks
                percent = docs / len(idx) * 100.0
                docrate = docs / elapsed if elapsed > 0.0 else 0.0
                mbs = dset_stats[2] * args.numranks / elapsed / 1024 / 1024 if elapsed > 0.0 else 0.0
                secs_left = int((len(idx) - docs) / docrate if docrate > 0.0 else 0.0)
                print(f"{timestamp}: Processed (estimated) {docs} of {len(idx)} docs ({percent:0.2f}%),",
                      f"{docrate:0.3f} docs/s, {mbs:0.3f} MiB/s,",
                      f"{secs_left} secs left ...",
                      flush=True)

        # finalize file of each rank
        for key in args.columns:
            builders[key].finalize(output_idx_files[key])
            del builders[key] # file closed in __del__
    except:
        # caught an exception, assume our file is invalid
        success = False

    # wait for all ranks to finish their files
    barrier(args)
    tokenize_end = time.time()

    # compute total stats across all processes
    all_sum_(args, dset_stats)
    if args.rank == 0:
        secs = tokenize_end - tokenize_start
        docrate = dset_stats[0] / secs if secs > 0.0 else 0.0
        sentrate = dset_stats[1] / secs if secs > 0.0 else 0.0
        byterate = dset_stats[2] / secs if secs > 0.0 else 0.0
        print("Seconds to tokenize:", secs)
        print("Documents=", dset_stats[0], "docs/sec=", docrate)
        print("Sentences=", dset_stats[1], "sent/sec=", sentrate)
        print("Bytes=", dset_stats[2], "bytes/sec=", byterate)

    # allreduce to check whether all ranks wrote their part successfully
    success = all_true(args, success)
    return success

def rank_files_merge(args):
    # rank 0 merges all per-rank files
    if args.rank == 0:
        print("Merging rank files ...", flush=True)
        merge_start = time.time()
        numbytes = 0

        # define name of single file
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in args.columns:
            filebase = f"{args.output_prefix}_{key}_{args.level}"
            output_bin_files[key] = data_file_path(filebase)
            output_idx_files[key] = index_file_path(filebase)
            builders[key] = make_builder(output_bin_files[key],
                                         impl=args.dataset_impl,
                                         vocab_size=args.vocab_size)

        # merge all ranks into one file
        for rank in range(args.numranks):
            for key in args.columns:
                infile = f"{args.output_prefix}_{key}_{args.level}_{rank}"
                print(f"Merging file {infile}", flush=True)
                builders[key].merge_file_(infile)

                # sum up the number of merged bytes
                binfile = data_file_path(infile)
                filesize = os.stat(binfile)[stat.ST_SIZE]
                numbytes += filesize

        # finalize the merged file
        print("Finalizing merged file ...", flush=True)
        for key in args.columns:
            builders[key].finalize(output_idx_files[key])
            del builders[key] # file closed in __del__

        merge_end = time.time()
        secs = merge_end - merge_start
        byterate = numbytes / secs if secs > 0.0 else 0.0
        print("Seconds to merge:", merge_end - merge_start)
        print(f"Merged {args.numranks} files into {args.output_prefix}")
        print(f"Bytes=", numbytes, "bytes/sec=", byterate)

    # hold everyone until rank 0 is done
    barrier(args)

def rank_files_delete(args):
    # delete per-rank files
    if args.rank == 0:
        print("Deleting rank files ...", flush=True)

    for key in args.columns:
        filebase = f"{args.output_prefix}_{key}_{args.level}_{args.rank}"

        binfile = data_file_path(filebase)
        if os.path.exists(binfile):
            os.remove(binfile)

        idxfile = index_file_path(filebase)
        if os.path.exists(idxfile):
            os.remove(idxfile)

    # hold everyone until all are done
    barrier(args)

def main():
    args = get_args()
    startup_start = time.time()

    # connect processes and cache our rank and number of procs in args
    init_distributed(args)

    # load the dataset
    dset = load_dset(args)
    if dset is None:
        return
    if args.rank == 0:
        print(dset)

    # create sample index list,
    # optionally shuffle the list,
    # and optionally limit the sample count
    idx = select_sample_list(args, len(dset))
    
    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    encoder = Encoder(args)
    args.vocab_size = encoder.tokenizer.vocab_size

    args.level = "document"
    if args.split_sentences:
        args.level = "sentence"

    # wait for all ranks before stopping timer
    barrier(args)
    startup_end = time.time()
    if args.rank == 0:
        print("Seconds to startup:", startup_end - startup_start)

    # have each rank write its file, returns False if any rank had a problem
    success = rank_files_write(args, dset, idx, encoder)

    # merge files if all ranks were successful writing their file
    if success:
        rank_files_merge(args)
    elif args.rank == 0:
        # If any process fails, we skip the merge since the resulting file would be invalid.
        # We still delete files to clean up, since those might be invalid anyway.
        print(f"ERROR: At least one process failed to write its file, skipping merge and cleaning up")

    # delete per-rank files, do this even on error
    rank_files_delete(args)

if __name__ == '__main__':
    main()
