import argparse
import time

from megatron.data import indexed_dataset
from megatron.data.indexed_dataset import infer_dataset_impl, MMapIndexedDataset


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--datasets', nargs='+', default=None,
                       help='Paths to one or more input datasets to merge')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    args = parser.parse_args()

    if args.datasets is not None and len(args.json_keys) > 1:
      raise RuntimeError("Merging datasets are performed only for one key at a time.")

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


    level = "document"
    if args.split_sentences:
        level = "sentence"

    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)

        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                     impl=infer_dataset_impl,
                                                     dtype=dtype)
        for dataset in args.datasets:
            builders[key].merge_file_(dataset)

    startup_end = time.time()
    print("Time to merge:", startup_end - startup_start)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])

    print(f"Merged {len(args.datasets)} datasets to {args.output_prefix}")

if __name__ == "__main__":
    main()