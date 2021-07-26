import argparse
import time

from megatron.data import indexed_dataset
from megatron.data.indexed_dataset import infer_dataset_impl, MMapIndexedDataset, data_file_path, index_file_path


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--datasets', nargs='+', default=None,
                       help='Paths to one or more input datasets to merge')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    return parser.parse_args()

def main():
    """
    Allows merging multiple types of datasets generated through preprocess_data script
    """
    args = get_args()
    startup_start = time.time()

    print("Merging", args.datasets)

    dataset_impl = infer_dataset_impl(args.datasets[0])
    assert dataset_impl is not None

    first_dataset = indexed_dataset.make_dataset(args.datasets[0], dataset_impl)
    dtype = first_dataset.dtype if isinstance(first_dataset, MMapIndexedDataset) else None

    print(f"Output prefix: {args.output_prefix}")

    output_filename = args.output_prefix
    output_bin_file = data_file_path(output_filename)
    output_idx_file = index_file_path(output_filename)
    builder = indexed_dataset.make_builder(output_bin_file,
                                           impl=infer_dataset_impl,
                                           dtype=dtype)
    for dataset in args.datasets:
        builder.merge_file_(dataset)

    startup_end = time.time()
    print("Time to merge:", startup_end - startup_start)

    builder.finalize(output_idx_file)

    print(f"Merged {len(args.datasets)} datasets to {args.output_prefix}")

if __name__ == "__main__":
    main()