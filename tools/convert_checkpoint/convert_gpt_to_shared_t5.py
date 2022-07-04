import argparse
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-shared-t5-path", type=Path, required=True)
    parser.add_argument("--only-weights", type=bool, action="store_true")
    parser.add_argument("--num-proc", type=int, default=1)
    return parser.parse_args()

def get_shared_t5_file(gpt_path_file: Path, output_shared_t5_path: Path) -> Path:
    """"Given a GPT checkpoint file path, get the equivalent shared T5 checkpoint path"""
    raise NotImplementedError()

def get_shared_t5_weight_name(gpt_weight_name: str) -> str:
    """Given a GPT checkpoint weight name, get the equivalent shated T5 checkpoint weight name"""
    raise NotImplementedError()

def map_gpt_weights_to_shared_t5_weights(filename: Path, output_shared_t5_path: Path):
    gpt_weights = torch.load(filename)

    shared_t5_filename = get_shared_t5_file(filename, output_shared_t5_path=output_shared_t5_path)
    shared_t5_weights = {}
    for name, weight in gpt_weights.items():
        shared_t5_weight_name = get_shared_t5_weight_name(name)
        shared_t5_weights[shared_t5_weight_name] = weight

    torch.save(shared_t5_weights, shared_t5_filename)

IS_WEIGHT_REGEX=re.compile(r"layer_[\d]{2}-model_[\d]{2}-model_states.pt")
def is_weight_file(filename: Path) -> bool:
    if filename.is_dir():
        return False

    basename = filename.name
    return IS_WEIGHT_REGEX.match(basename) is not None

def main():
    args = get_args()

    weight_files = [filename for filename in args.gpt_checkpoint_path.iterdir() if is_weight_file(filename)]
    if args.num_proc == 1:
        for weight_file in weight_files:
            map_gpt_weights_to_shared_t5_weights(weight_file, output_shared_t5_path=args.output_shared_t5_path)
    else:
        with Pool(args.num_proc) as pool:
            pool.map(
                partial(map_gpt_weights_to_shared_t5_weights, output_shared_t5_path=args.output_shared_t5_path),
                weight_files
            )

if __name__ == "__main__":
    main()