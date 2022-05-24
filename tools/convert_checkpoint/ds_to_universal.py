#!/usr/bin/env python

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from pprint import pprint
import argparse
import glob
import logging
import os
import sys
import torch

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

# XXX: this is a temp hack that creates fake params but with the right shapes
def save_params_universal(dir, param_shapes):
    for name, shape in param_shapes.items():
        param_base_path = os.path.join(dir, name)
        os.makedirs(param_base_path, exist_ok=True)
        print(f"{name}: {shape} => {param_base_path}")
        for state in ("fp32", "exp_avg", "exp_avg_sq"):
            path = os.path.join(param_base_path, f"{state}.pt")
            param = torch.Tensor(shape)
            _save_checkpoint(path, param)


def extract_zero_fragments(dir, param_shapes, ds_checkpoint, dp_index, pp_index, tp_index):
    sd = ds_checkpoint.get_zero_checkpoint_state(
        pp_index=pp_index,
        tp_index=tp_index,
        dp_index=dp_index)

    pprint(f"Processing {dp_index=} {pp_index=}, {tp_index=}")

    optim_sd = sd["optimizer_state_dict"]
    param_slice_mappings = optim_sd["param_slice_mappings"]

    # dict
    state_groups = optim_sd["base_optimizer_state"]["state"]
    # list
    fp32_groups = optim_sd["single_partition_of_fp32_groups"]
    param_groups_cnt = len(state_groups)

    for param_group_id in range(param_groups_cnt):

        flat_state = dict(
            exp_avg=state_groups[param_group_id]["exp_avg"],
            exp_avg_sq=state_groups[param_group_id]["exp_avg_sq"],
            fp32=fp32_groups[param_group_id],
        )

        for k,v in param_slice_mappings[param_group_id].items():
            print(f"{param_group_id} {k} => {v.start}:{v.numel}")

            for state_key in flat_state.keys():
                dump_param_fragment(dir, state_key, flat_state[state_key], k, v.start, v.numel)




cnt = 0
def dump_param_fragment(dir, state_name, state_flat_tensor, param_name, offset, numel):

    global cnt # temp hack

    param_base_path = os.path.join(dir, param_name)
    os.makedirs(param_base_path, exist_ok=True)

    cnt += 1
    counter = f"{cnt:0>10d}"

    path = os.path.join(param_base_path, f"{state_name}.{counter}")

    print(f"{param_name}: {offset}: {numel} => {path}")

    t = state_flat_tensor.narrow(0, offset, numel)
    _save_checkpoint(path, t)



def merge_zero_fragments(dir, param_shapes):

    for name, shape in param_shapes.items():
        param_base_path = os.path.join(dir, name)
        print(f"\n{name}: {shape} => {param_base_path}")

        # XXX: shouldn't be in the states
        if "position_embeddings" in name:
            continue


        for state in ("fp32", "exp_avg", "exp_avg_sq"):
            final_path = os.path.join(param_base_path, f"{state}.pt")
            prefix_path = os.path.join(param_base_path, f"{state}")
            paths = sorted(list(glob.glob(f"{prefix_path}.0*")))
            orig_paths = deepcopy(paths)


            # XXX: tmp hack - need to deal with tied vars here
            if "word_embeddings.weight" in name and len(paths)>1:
                paths = paths[:1]


            print(paths)

            fragments = [torch.load(p) for p in paths]

            print(f"Expected shape: {shape}")
            print(f"Fragment sizes:", list(frag.shape for frag in fragments))

            # merge
            param = torch.cat(fragments, dim=0)
            param = param.reshape(shape)
            print(f"Final shape: {param.shape}")
            _save_checkpoint(final_path, param)
            for p in orig_paths:
                os.unlink(p)

            # XXX: probably not needed since torch.reshape would have failed if the inputs size was wrong
            if param.shape != shape:
                logging.error(f"✘ {name}: expected {shape} but got {param.shape}")


def main():
    print(f'Convert DeepSpeed Checkpoint to Universal Checkpoint')

    args = parse_arguments()
    print(
        f'Converting DeepSpeed checkpoint in {args.input_folder} to Universal checkpoint in {args.output_folder}'
    )

    ds_checkpoint = DeepSpeedCheckpoint(args.input_folder)#, 1, 2) # args.target_tp, args.target_pp)
    iteration = ds_checkpoint.get_iteration()
    _create_latest_file(args.output_folder, iteration)
    checkpoint_paths = _create_checkpoint_paths(args.output_folder, iteration,
                                                ds_checkpoint.tp_degree,
                                                ds_checkpoint.pp_degree)

    mp_sd = torch.load(ds_checkpoint.mp_rank_files[0], map_location=torch.device('cpu'))

    param_shapes = mp_sd["param_shapes"]
    # fix back to normal flat dict
    param_shapes = dict((k,v) for d in param_shapes for k,v in d.items() )

    # make fake params
    # save_params_universal(args.output_folder, param_shapes)

    for i in range(ds_checkpoint.dp_degree):
        for j in range(ds_checkpoint.pp_degree):
            for k in range(ds_checkpoint.tp_degree):
                print(f"{i=}, {j=}, {k=}")
                extract_zero_fragments(args.output_folder, param_shapes, ds_checkpoint, i, j, k)

    merge_zero_fragments(args.output_folder, param_shapes)



if __name__ == "__main__":
    main()
