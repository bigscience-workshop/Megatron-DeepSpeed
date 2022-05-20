#!/usr/bin/env python

import argparse
import os
import torch
from collections import OrderedDict
import sys
from pathlib import Path
from pprint import pprint

# insert megatron's root dir into sys.path
root_repo_path = str(Path(__file__).resolve().parents[2])
if root_repo_path not in sys.path:
    sys.path.insert(0, root_repo_path)


from deepspeed.checkpoint import DeepSpeedCheckpoint

MODEL_KEY = 'model'
ARGS_KEY = 'args'
LANGUGAGE_MODEL_KEY = 'language_model'
EMBEDDING_KEY = 'embedding'
ENCODER_KEY = 'encoder'
WORD_EMBEDDINGS_FOR_HEAD_KEY = 'word_embeddings_for_head'
WORD_EMBEDDINGS_KEY = 'word_embeddings'
FINAL_LAYER_NORM_KEY = 'final_layernorm'
CHECKPOINT_VERSION_KEY = 'checkpoint_version'
CHECKPOINT_VERSION_VALUE = 3.0
ITERATION_KEY = 'iteration'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder',
                        default=None,
                        type=str,
                        help='Input DeepSpeed Checkpoint folder')
    parser.add_argument('--output_folder',
                        default=None,
                        type=str,
                        help='Output Megatron checkpoint folder')
    parser.add_argument('--target_tp',
                        default=1,
                        type=int,
                        help='Target TP degree')
    parser.add_argument('--target_pp',
                        default=1,
                        type=int,
                        help='Target PP degree')
    parser.add_argument(
        '--for_release',
        action='store_true',
        help='Convert for release purpose, reset some (progress) counters.')
    args = parser.parse_args()
    print(f'args = {args}')
    return args


def _convert_ds_transformer_state(sd_list):
    new_sd = OrderedDict()
    for i, sd in enumerate(sd_list):
        for key, value in sd.items():
            new_key = f'layers.{i}.{key}'
            new_sd[new_key] = value

    return new_sd


def _create_checkpoint_paths(base_folder, iteration, tp_degree, pp_degree):
    path_list = []
    iter_folder = f'iter_{iteration:07d}'
    for i in range(0, tp_degree):
        path_list.append([])
        for j in range(0, pp_degree):
            rank_folder = f'mp_rank_{i:02d}' if pp_degree == 1 else f'mp_rank_{i:02d}_{j:03d}'
            ckpt_path = os.path.join(rank_folder, 'model_optim_rng.pt')
            path_list[i].append(
                os.path.join(base_folder, iter_folder, ckpt_path))

    return path_list


def _create_megatron_dict():
    language_model_dict = {EMBEDDING_KEY: {}, ENCODER_KEY: {}}
    megatron_dict = {
        MODEL_KEY: {
            LANGUGAGE_MODEL_KEY: language_model_dict
        },
        CHECKPOINT_VERSION_KEY: CHECKPOINT_VERSION_VALUE
    }
    return megatron_dict


def _save_checkpoint(file_path, chkpt_sd):
    dir, _ = os.path.split(file_path)
    os.makedirs(dir, exist_ok=True)
    torch.save(chkpt_sd, file_path)



def _create_latest_file(base_folder, iteration):
    file_path = os.path.join(base_folder, 'latest_checkpointed_iteration.txt')
    os.makedirs(base_folder, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(str(iteration))


def save_params_universal(dir, param_shapes):
    for name, shape in param_shapes.items():
        param_base_path = os.path.join(dir, name)
        os.makedirs(param_base_path, exist_ok=True)
        print(f"{name}: {shape} => {param_base_path}")
        for state in ("fp32", "exp_avg", "exp_avg_sq"):
            path = os.path.join(param_base_path, state)
            param = torch.Tensor(shape)
            _save_checkpoint(path, param)


def main():
    print(f'Convert DeepSpeed Checkpoint to Universal Checkpoint')

    args = parse_arguments()
    print(
        f'Converting DeepSpeed checkpoint in {args.input_folder} to Universal checkpoint in {args.output_folder}'
    )

    ds_checkpoint = DeepSpeedCheckpoint(args.input_folder, args.target_tp,
                                        args.target_pp)
    iteration = ds_checkpoint.get_iteration()
    _create_latest_file(args.output_folder, iteration)
    checkpoint_paths = _create_checkpoint_paths(args.output_folder, iteration,
                                                ds_checkpoint.tp_degree,
                                                ds_checkpoint.pp_degree)

    sd = torch.load(ds_checkpoint.mp_rank_files[0], map_location=torch.device('cpu'))

    param_shapes = sd["param_shapes"]
    # fix back to normal dict
    param_shapes = dict((k,v) for d in param_shapes for k,v in d.items() )
    pprint(param_shapes)

    save_params_universal(args.output_folder, param_shapes)

    # for i in range(0, ds_checkpoint.tp_degree):
    #     for j in range(0, ds_checkpoint.pp_degree):
    #         sd = _create_rank_checkpoint(ds_checkpoint, i, j, args.for_release)
    #         _save_checkpoint(checkpoint_paths[i][j], sd)


if __name__ == "__main__":
    main()
