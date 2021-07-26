import argparse
import time

from megatron.data import indexed_dataset
from megatron.tokenizer import build_tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--datasets', nargs='+', default=None,
                       help='Paths to one or more input datasets to merge')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

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

    # TODO: Remove once we find a way to get vocab_size without loading the entire
    tokenizer = build_tokenizer(args)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    print(f"Vocab size: {tokenizer.vocab_size}")
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
                                                     impl=args.dataset_impl,
                                                     vocab_size=tokenizer.vocab_size)
        for dataset in args.datasets:
            builders[key].merge_file_(dataset)

    startup_end = time.time()
    print("Time to merge:", startup_end - startup_start)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])

    print(f"Merged {len(args.datasets)} datasets to {args.output_prefix}")

if __name__ == "__main__":
    main()