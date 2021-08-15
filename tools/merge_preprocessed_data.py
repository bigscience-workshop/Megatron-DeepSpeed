import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

import argparse
import time

from megatron import print_rank_0
from megatron.data import indexed_dataset
from megatron.data.indexed_dataset import infer_dataset_impl, MMapIndexedDataset, data_file_path, index_file_path, merge_files_dist
from megatron.data.distdata import DistData


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--datasets', nargs='+', default=None,
                       help='Paths to one or more input datasets to merge')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--merge', type=str, default='parallel', choices = ['parallel', 'serial'],
                       help='Whether to use a distributed parallal merge or a non-distributed serial merge.')
    group.add_argument('--torch-backend', type=str, default='gloo', choices = ['gloo', 'mpi'],
                       help='Select torch.distributed backend.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='Local rank of calling process on its node (from torch.distributed.launch).')

    args = parser.parse_args()

    # initialize distributed environment if parallel merge requested
    if args.merge == 'parallel':
        args.distctx = DistData(backend=args.torch_backend)

    return args

def main():
    """
    Allows merging multiple types of datasets generated through preprocess_data script
    """
    args = get_args()
    startup_start = time.time()

    print_rank_0(f"Merging {args.datasets}")
    print_rank_0(f"Output prefix: {args.output_prefix}")

    if args.merge == 'parallel':
        merge_files_dist(args.output_prefix, args.datasets, args.distctx)
    else:
        # We use the first dataset to infer the dataset implementation common to all datasets.
        dataset_impl = infer_dataset_impl(args.datasets[0])
        assert dataset_impl is not None

        first_dataset = indexed_dataset.make_dataset(args.datasets[0], dataset_impl)
        # We use the first dataset to infer the dtype common to all datasets.
        dtype = first_dataset.dtype if isinstance(first_dataset, MMapIndexedDataset) else None

        output_filename = args.output_prefix
        output_bin_file = data_file_path(output_filename)
        output_idx_file = index_file_path(output_filename)
        builder = indexed_dataset.make_builder(output_bin_file,
                                               impl=dataset_impl,
                                               dtype=dtype)
        for dataset in args.datasets:
            builder.merge_file_(dataset)

        builder.finalize(output_idx_file)

    startup_end = time.time()
    print_rank_0(f"Time to merge: {startup_end - startup_start")

    print_rank_0(f"Merged {len(args.datasets)} datasets to {args.output_prefix}")

if __name__ == "__main__":
    main()
