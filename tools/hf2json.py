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
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from datasets import config, logging, load_dataset
from datasets.utils.file_utils import OfflineModeIsEnabled

import json

from megatron.data.distdata import DistData

def msg(msg, flush=False):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"{timestamp}: {msg}", flush=flush)

def msgerr(msg, flush=False):
    print(f"ERROR: {msg}", flush=flush)

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

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--torch-backend', type=str, default='gloo', choices=['gloo', 'mpi'],
                       help='Select torch.distributed backend.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='Local rank of calling process on its node (from torch.distributed.launch).')
    group.add_argument('--merge', type=str, default='parallel', choices=['parallel', 'serial', 'both'],
                       help=('Method to merge intermediate per-rank files into the final data files.  '
                             'With "parallel", each rank writes directly to the final files, '
                             'while rank 0 copies data from all per-rank files with "serial".  '
                             'A parallel merge can be faster, but for correctness, it requires the underlying file system '
                             'to support parallel write operations to a file that is shared among multiple processes.  '
                             'One can choose "both" for testing purposes, in which case the final files written '
                             'by the parallel method are given an additional ".par" extension.'))
    group.add_argument('--scratch', type=str, default=None,
                       help=('Path to local storage on compute nodes to write per-rank files before merging, like /dev/shm.  '
                             'One can only use this option with a parallel merge.'))
    group.add_argument('--log-interval', type=int, default=30,
                       help='Seconds between progress updates (0 to disable)')

    args = parser.parse_args()

    # initialize our distributed environment
    args.distctx = DistData(backend=args.torch_backend)

    # some functions like build_tokenizer use args.rank to filter stdout messages
    args.rank = args.distctx.rank
    args.numranks = args.distctx.numranks

    # TODO: perhaps more user friendly to disable scratch and print a warning?
    # check that serial merge is not attempted with scratch
    if args.scratch is not None and args.merge != 'parallel':
        raise  ValueError("The --scratch option is only valid with --merge=parallel")

    return args

def format_byterate(byterate):
    mbps = byterate / (1024.0 * 1024.0)
    return f"{mbps:0.3f} MB/s"

def load_dset(args):
    # Avoid downloading datasets unless explicitly requested.
    # We allow the user to override this behavior if they set $HF_DATASETS_OFFLINE.
    if 'HF_DATASETS_OFFLINE' not in os.environ:
        # To disable downloads, we could set $HF_DATASETS_OFFLINE=1.
        # However, the HF_DATASETS_OFFLINE environment variable is processed
        # when the datasets module is imported, so it must be set before the import statement.
        # sets HF_DATASETS_OFFLINE within the environment of this script
        #os.environ['HF_DATASETS_OFFLINE'] = "1"

        # Alternatively, one can set datasets.config.HF_DATASETS_OFFLINE=1.
        # That seems to work even after the import statement,
        # though this usage is not documented.
        config.HF_DATASETS_OFFLINE = 1

    # silence info messages from all procs except rank 0 
    if args.rank != 0:
        logging.set_verbosity(logging.ERROR)

    time_start = time.time()

    # Load the specified HuggingFace dataset.
    # Give rank 0 a head start in case the dataset is not already cached.
    err = None
    dsetname = args.input
    if args.rank == 0:
        msg(f"Opening dataset {dsetname}")
        try:
            dset = load_dataset(dsetname, split=args.split, keep_in_memory=None)
        except OfflineModeIsEnabled as e:
            msgerr(f"Cannot download '{dsetname}' since running in offline mode.")
            msgerr(f"If the dataset is large, it may be more efficient to download with a single process:")
            msgerr(f"    from datasets import load_dataset")
            msgerr(f"    dset = load_dataset('{dsetname}')")
            msgerr(f"Alternatively, one can force this script to download by setting $HF_DATASETS_OFFLINE=0", flush=True)
            err = e
        except Exception as e:
            msgerr(f"Unexpected error: {sys.exc_info()[0]}", flush=True)
            err = e

    # determine whether rank 0 succeeded in loading the dataset
    args.distctx.allraise_if(err)

    # Rank 0 succeeded, attempt to load dataset on all other ranks.
    # This should load from cache now.
    if args.rank != 0:
        try:
            dset = load_dataset(dsetname, split=args.split, keep_in_memory=None)
        except Exception as e:
            # this print might be noisy, but better than nothing
            msgerr(f"Unexpected error: {sys.exc_info()[0]}", flush=True)
            err = e

    # verify that all ranks loaded the dataset
    args.distctx.allraise_if(err)

    time_end = time.time()
    if args.rank == 0:
        msg(f"Seconds to load dataset: {time_end - time_start}", flush=True)

    return dset

def get_num_samples(args, dset_size):
    """Given a dataset size and optional count argument, return number of samples to process."""
    num_samples = dset_size
    if args.count is not None and args.count < dset_size:
        num_samples = args.count
    return num_samples

def select_sample_list(args, dset_size):
    """Given the total number of samples, select a list of sample index values"""
    # determine total number of samples that we'll read
    num_samples = get_num_samples(args, dset_size)

    # create sample index list on rank 0,
    # optionally shuffle the list,
    # and optionally limit the sample count
    time_select = time.time()
    idxlist = None
    if args.rank == 0:
        # generate a list of all index values
        idxlist = np.arange(dset_size, dtype=np.int64)

        # optionally shuffle
        if args.shuffle:
            # args.seed may be an int (to seed) or None (to not)
            rng = np.random.default_rng(args.seed)
            rng.shuffle(idxlist)

        # optionally limit the sample count
        if args.count is not None:
            idxlist = idxlist[:num_samples]

    # get a list of the number of elements each rank will hold
    counts = get_proc_counts(num_samples, args.numranks)

    # scatter sample index values from rank 0 to all procs
    # based on distribution defined in counts list
    time_bcast = time.time()
    idx = args.distctx.scatterv_(idxlist, counts, root=0)

    args.distctx.barrier()
    time_end = time.time()
    if args.rank == 0:
        msg(f"Select index stats:")
        msg(f"    Shuffle: {args.shuffle}")
        msg(f"    Seconds to select: {time_bcast - time_select}")
        msg(f"    Seconds to broadcast: {time_end - time_bcast}")
        msg(f"    Seconds total: {time_end - time_select}", flush=True)

    return idx

def get_proc_counts(num, num_ranks):
    num_per_rank, remainder = divmod(num, num_ranks)
    return [num_per_rank + 1 if rank < remainder else num_per_rank for rank in range(num_ranks)]

def get_filename(args, rank=None):
    pathname = args.output_prefix

    # redirect per-rank file to scratch dir if defined
    if args.scratch is not None and rank is not None:
        basename = os.path.basename(pathname)
        pathname = os.path.join(args.scratch, basename)

    if rank is not None:
        filename = f"{pathname}_{rank}"
    else:
        filename = f"{pathname}"

    return filename

def rank_files_write(args, dset, idx):
    time_start = time.time()

    # compute total number of samples we'e processing
    num_samples = get_num_samples(args, len(dset))

    # we'll total up the number of bytes
    # processed across all ranks
    dset_stats = np.zeros(2, dtype=np.int64) # samples, bytes

    # we'll set this to false on any problem
    err = None
    times = np.zeros(3, dtype=np.float32) # read, tokenize, write
    try:
        filename = get_filename(args, args.rank)
        with open(filename, "w") as fout:
            progress_next = time.time() + float(args.log_interval)
            for i in idx:
                sample_id = int(i)

                start_read = time.time()
                sample = dset[sample_id]
                sample['id'] = sample_id

                start_encode = time.time()
                jsonline = json.dumps(sample)

                start_write = time.time()
                fout.write(jsonline + '\n')

                dset_stats[0] += 1
                dset_stats[1] += len(jsonline) + 1

                if args.rank == 0 and args.log_interval > 0 and time.time() > progress_next:
                    current = time.time()
                    progress_next = current + float(args.log_interval)

                    elapsed = current - time_start
                    docs = dset_stats[0] * args.numranks
                    percent = docs / num_samples * 100.0
                    docrate = docs / elapsed if elapsed > 0.0 else 0.0
                    mbs = dset_stats[1] * args.numranks / elapsed / 1024 / 1024 if elapsed > 0.0 else 0.0
                    secs_left = int((num_samples - docs) / docrate if docrate > 0.0 else 0.0)
                    msg(f"Processed (est) {docs} of {num_samples} docs ({percent:0.2f}%) in {int(elapsed)} secs, "
                        f"{docrate:0.3f} docs/s, {mbs:0.3f} MB/s, "
                        f"{secs_left} secs left ...",
                        flush=True)

    except Exception as e:
        # caught an exception, assume our file is invalid
        err = e

    # In case rank 0 finishes early and stops printing progress messages,
    # inform user that it's waiting for other ranks to finish.
    if args.rank == 0 and args.log_interval > 0:
        msg(f"Waiting for ranks to finalize files ...", flush=True)

    # wait for all ranks to finish their files
    args.distctx.barrier()
    time_end = time.time()

    # compute total stats across all processes
    args.distctx.all_sum_(times)
    args.distctx.all_sum_(dset_stats)
    if args.rank == 0:
        secs = time_end - time_start
        docrate = dset_stats[0] / secs if secs > 0.0 else 0.0
        byterate = dset_stats[1] / secs if secs > 0.0 else 0.0
        secs_read_per_sample = times[0] / dset_stats[0] if dset_stats[0] > 0 else 0.0
        secs_encode_per_sample = times[1] / dset_stats[0] if dset_stats[0] > 0 else 0.0
        secs_write_per_sample = times[2] / dset_stats[0] if dset_stats[0] > 0 else 0.0
        msg("Process stats:")
        msg(f"    Seconds to process: {secs}")
        msg(f"    {dset_stats[0]} docs {docrate} docs/sec")
        msg(f"    {dset_stats[1]} bytes {format_byterate(byterate)}")
        msg(f"    Total read seconds {times[0]}, {secs_read_per_sample} sec/sample")
        msg(f"    Total encode seconds {times[1]}, {secs_encode_per_sample} sec/sample")
        msg(f"    Total write seconds {times[2]}, {secs_write_per_sample} sec/sample")

    # check whether all ranks wrote their part successfully
    args.distctx.allraise_if(err)

def rank_files_merge_parallel(args):
    """Each process directly writes its portion of the data from its per-rank file into the final file."""
    merge_start = time.time()
    numbytes = np.zeros(1, dtype=np.int64)

    # merge the per-rank file from each process into a single shared file
    filemain = get_filename(args)
    filerank = get_filename(args, args.rank)
    numbytes[0] = args.distctx.gather_files_concat(filemain, [filerank])

    # If user want to use both a parallel and serial merge (for testing),
    # rename the parallel output files so that the serial merge does not clobber them.
    if args.merge == 'both' and args.rank == 0:
        os.rename(filemain, filemain + ".par")

    # Total up number of bytes read across all ranks,
    # and wait on all ranks before stopping the timer.
    args.distctx.all_sum_(numbytes)
    merge_end = time.time()
    if args.rank == 0:
        secs = merge_end - merge_start
        byterate = numbytes[0] / secs if secs > 0.0 else 0.0
        msg("Parallel merge stats:")
        msg(f"    Scratch: {args.scratch}")
        msg(f"    Seconds to merge: {secs}")
        msg(f"    {int(numbytes)} bytes {format_byterate(byterate)}")

def rank_files_merge(args):
    # use parallel merge if asked
    if args.merge in ['parallel', 'both']:
        rank_files_merge_parallel(args)

    # if using node-local storage, skip sequential merge
    if args.scratch is not None:
        return

def rank_files_delete(args):
    # delete per-rank files
    if args.rank == 0:
        msg("Deleting rank files ...", flush=True)

    filename = get_filename(args, args.rank)
    if os.path.exists(filename):
        os.remove(filename)

    # hold everyone until all are done
    args.distctx.barrier()

def main():
    args = get_args()
    startup_start = time.time()

    # load the dataset
    dset = load_dset(args)
    if args.rank == 0:
        print(dset)
        msg(f"Processing features: {args.columns}")

    # create sample index list,
    # optionally shuffle the list,
    # and optionally limit the sample count
    idx = select_sample_list(args, len(dset))

    # wait for all ranks before stopping timer
    args.distctx.barrier()
    startup_end = time.time()
    if args.rank == 0:
        msg(f"Seconds to startup: {startup_end - startup_start}")

    # have each rank write its file,
    # all ranks should raise an exception if any rank has a problem
    try:
        rank_files_write(args, dset, idx)
    except Exception as e:
        # If any process fails, we skip the merge since the resulting file would be invalid.
        # We still delete files to clean up, since those might be invalid anyway.
        if args.rank == 0:
            msgerr(f"At least one process failed to write its file, skipping merge and cleaning up", flush=True)

        # delete per-rank files, do this even on error
        rank_files_delete(args)

        # re-raise exception caught during write phase
        raise e

    # all ranks were successful writing their file, merge them into one
    rank_files_merge(args)

    # delete per-rank files
    rank_files_delete(args)

    end_time = time.time()
    if args.rank == 0:
        msg(f"Runtime: {end_time - startup_start} secs", flush=True)
        msg(f"Done")

if __name__ == '__main__':
    main()
