"""
Many-to-one reshaping for checkpoints trained without ZeRO

Usage example:
python tools/convert_checkpoint/deepspeed_to_deepspeed_nozero.py --input_folder ../global_step156000_old --output_folder ../global_step156000_tp2pp2 --target_tp 2 --target_pp 2

Snippet for manual testing in a python environment:
from tools.convert_checkpoint.deepspeed_to_deepspeed_nozero import DeepSpeedCheckpointNoZeRO
ds_checkpoint = DeepSpeedCheckpointNoZeRO('../global_step156000_old',1,4)
sd = ds_checkpoint.get_embedding_state(0)
sd['word_embeddings.weight'].shape

Notes:
- You need changes from https://github.com/microsoft/DeepSpeed/pull/1953/ for this script to work
- There is a bug in PP layers in the above PR, so you may unexpectedly loose layers if reshaping to PP>1, see https://github.com/microsoft/DeepSpeed/pull/1953/files#r904886186
- Only tested on FP16 Checkpoints
- Data Parallelism is irrelevant here, as it does not influence non-ZeRO ckpts.
"""

import argparse
import os
from pathlib import Path
import sys

import torch

# Insert megatron's root dir into sys.path
root_repo_path = str(Path(__file__).resolve().parents[2])
if root_repo_path not in sys.path:
    sys.path.insert(0, root_repo_path)


from megatron.tokenizer.tokenizer import _vocab_size_with_padding

from deepspeed.checkpoint import (
    DeepSpeedCheckpoint,
    get_model_ckpt_name_for_rank,
    get_layer_ckpt_name_for_rank,
)
from deepspeed.checkpoint.deepspeed_checkpoint import (
    ARGS_KEY,
    CHECKPOINT_INFO_KEY,
    EMBEDDING_LAYER_INDEX,
    FINAL_LAYER_NORM_INDEX,
    SEQUENTIAL_LAYERS,
    LAYER_CONCAT_DIM
)
from deepspeed.checkpoint.reshape_meg_2d import reshape_meg_2d_parallel, meg_2d_parallel_map
from deepspeed.checkpoint.reshape_utils import (get_files, get_files_with_prefix)
from deepspeed.checkpoint.constants import (LAYER_FILE_PREFIX, MODEL_FILE_PREFIX)


# Add layers that should not be concatted
# The below are just copies across tp parallel files, thus we do not need to merge them
SEQUENTIAL_LAYERS.append('word_embeddings.norm.weight')
SEQUENTIAL_LAYERS.append('word_embeddings.norm.bias')

class DeepSpeedCheckpointNoZeRO(DeepSpeedCheckpoint):
    def __init__(self, dir, tp_degree=None, pp_degree=None, dp_degree=None):
        self.dir = dir
        self._validate_folder(dir)

        self.file_list = get_files(dir)
        self.zero_files = [] #get_files_with_prefix(self.file_list, ZERO_FILE_PREFIX)
        self.layer_files = get_files_with_prefix(self.file_list, LAYER_FILE_PREFIX)
        self.mp_rank_files = get_files_with_prefix(self.file_list, MODEL_FILE_PREFIX)

        self.layer_keys = self._get_layer_keys()
        self.layer_count = len(self.layer_keys)
        self.original_tp_degree = len(
            get_files_with_prefix(self.layer_files,
                                  f'{LAYER_FILE_PREFIX}01'))
        self.original_pp_degree = len(self.mp_rank_files) // self.original_tp_degree
        self.original_dp_degree = max(
            1,
            len(self.zero_files) // (self.original_pp_degree * self.original_tp_degree))

        self.tp_degree = self.original_tp_degree if tp_degree is None else tp_degree
        self.pp_degree = self.original_pp_degree if pp_degree is None else pp_degree
        self.dp_degree = self.original_dp_degree if dp_degree is None else dp_degree

        self.original_world_size = self.original_tp_degree * self.original_pp_degree * self.original_dp_degree
        self.world_size = self.tp_degree * self.pp_degree # * self.dp_degree

        self.old_2d_map = meg_2d_parallel_map(self.original_pp_degree,
                                              self.original_tp_degree)
        self.old_2d_map.simple_init()
        self.new_2d_map = reshape_meg_2d_parallel(old_pp_degree=self.original_pp_degree,
                                                  old_tp_degree=self.original_tp_degree,
                                                  new_pp_degree=self.pp_degree,
                                                  new_tp_degree=self.tp_degree)

        # No ZeRO Checkpoint
        # self.zero_checkpoint = ZeROCheckpoint(dir)
        # if self.is_change_pp_degree() or self.is_change_tp_degree(
        # ) or self.is_change_dp_degree():
        #     self.zero_checkpoint.reshape(
        #         model_3d_desc(self.pp_degree,
        #                       self.tp_degree,
        #                       self.dp_degree))

        self.global_state = {}

        self._sanity_check()
        self.pp_to_transformer_map = self._build_pp_transformer_map()
        self.transformer_file_map = self._build_transformer_file_map()
        self.tp_to_embedding_map = self._build_tp_other_layer_map(EMBEDDING_LAYER_INDEX)
        self.tp_to_final_norm_map = self._build_tp_other_layer_map(
            FINAL_LAYER_NORM_INDEX)
        self._build_global_state()

    # Overwrite _merge_state_dicts to include additional SEQUENTIAL_LAYERS
    def _merge_state_dicts(self, sd_list):
        merged_sd = {}
        for key in sd_list[0].keys():
            if not key in SEQUENTIAL_LAYERS:
                cat_dim = LAYER_CONCAT_DIM.get(key, 0)
                merged_sd[key] = torch.cat([sd[key] for sd in sd_list], dim=cat_dim)
            else:
                merged_sd[key] = sd_list[0][key]

        return merged_sd

CHECKPOINT_FILE_SUFFIX = '_model_states.pt'
MP_WORLD_SIZE ='mp_world_size'
WORD_EMBEDDINGS_KEY = 'word_embeddings.weight'
ORIGINAL_VOCAB_SIZE = 'original_vocab_size'
PADDED_VOCAB_SIZE = 'padded_vocab_size'

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
                        default=None,
                        type=int,
                        help='Target TP degree')
    parser.add_argument('--target_pp',
                        default=None,
                        type=int,
                        help='Target PP degree')
    parser.add_argument('--original_vocab_size',
                        default=250680,
                        type=int,
                        help='Vocab Size prior to padding; Default is for BLOOM models for which after padding it is commonly 250880')
    args = parser.parse_args()
    print(f'args = {args}')
    return args


def _save_checkpoint(file_path, chkpt_sd):
    dir, _ = os.path.split(file_path)
    os.makedirs(dir, exist_ok=True)
    torch.save(chkpt_sd, file_path)


def _create_transformer_layer_checkpoint(ds_checkpoint, base_folder, tp_index, pp_index):
    sd_list = ds_checkpoint.get_transformer_state(tp_index, pp_index)
    layer_id_list = ds_checkpoint.get_pp_transformer_map(pp_index)
    assert len(sd_list) == len(layer_id_list)
    for sd, layer_id in zip(sd_list, layer_id_list):
        ckpt_path = get_layer_ckpt_name_for_rank(
            base_folder=base_folder,
            layer_id=layer_id,
            tp_rank=tp_index)
        _save_checkpoint(ckpt_path, sd)


def _strip_vocab_padding(ds_checkpoint, padded_vocab_tensor, original_vocab_size=250680):
    target_args = ds_checkpoint.get_args()
    checkpoint_info = None
    if hasattr(ds_checkpoint, "get_checkpoint_info"):
        checkpoint_info = ds_checkpoint.get_checkpoint_info()
        if checkpoint_info is not None and ORIGINAL_VOCAB_SIZE in checkpoint_info:
            original_vocab_size = checkpoint_info[ORIGINAL_VOCAB_SIZE]
    target_args.tensor_model_parallel_size = ds_checkpoint.tp_degree
    target_args.padded_vocab_size = _vocab_size_with_padding(original_vocab_size, target_args)
    assert target_args.padded_vocab_size <= padded_vocab_tensor.numel()
    if checkpoint_info is not None:
        checkpoint_info[PADDED_VOCAB_SIZE] = target_args.padded_vocab_size
    # Need to divide by ds_checkpoint.tp_degree to allow many-to-many reshaping e.g. from TP=4 -> TP=2
    # This is because the vocab tensor will be split across tp dimensions
    unpadded_vocab_tensor = torch.narrow(padded_vocab_tensor, 0, 0, target_args.padded_vocab_size // ds_checkpoint.tp_degree)
    return unpadded_vocab_tensor.clone()


def _create_embedding_layer_checkpoint(ds_checkpoint, base_folder, tp_index, original_vocab_size=250680):
    sd = ds_checkpoint.get_embedding_state(tp_index)
    if ds_checkpoint.is_change_tp_degree():
        sd[WORD_EMBEDDINGS_KEY] = _strip_vocab_padding(ds_checkpoint, sd[WORD_EMBEDDINGS_KEY], 
            original_vocab_size=original_vocab_size)
    layer_id = ds_checkpoint.get_embedding_layer_id()
    ckpt_path = get_layer_ckpt_name_for_rank(
        base_folder=base_folder,
        tp_rank=tp_index,
        layer_id=layer_id)
    _save_checkpoint(ckpt_path, sd)


def _create_final_norm_layer_checkpoint(ds_checkpoint, base_folder, tp_index):
    sd = ds_checkpoint.get_final_norm_state(tp_index)
    layer_id = ds_checkpoint.get_final_norm_layer_id()
    ckpt_path = get_layer_ckpt_name_for_rank(
        base_folder=base_folder,
        tp_rank=tp_index,
        layer_id=layer_id)
    _save_checkpoint(ckpt_path, sd)


def _create_2d_parallel_checkpoint(ds_checkpoint, base_folder, tp_index,
                                   pp_index):
    sd = ds_checkpoint.get_2d_parallel_state(tp_index=tp_index,
                                             pp_index=pp_index)

    # The above merged all tensors including random states tensors
    # We just choose the first one as we cannot reuse them all
    # This is the same way it is done for NumPy random states which are ignored in the above
    fname_list = ds_checkpoint.get_2d_parallel_files(tp_index=tp_index, pp_index=pp_index)
    first_sd = torch.load(fname_list[0], map_location=torch.device('cpu'))
    sd['cuda_rng_state'] = first_sd['cuda_rng_state']
    sd['torch_rng_state'] = first_sd['torch_rng_state']
    sd['rng_tracker_states']['model-parallel-rng'] = first_sd['rng_tracker_states']['model-parallel-rng']

    # DeepSpeed sets the MP_WORLD_SIZE to the size of all non-data-parallel gpus
    sd[MP_WORLD_SIZE] = ds_checkpoint.tp_degree * ds_checkpoint.pp_degree
    file_id = pp_index * ds_checkpoint.tp_degree + tp_index
    ckpt_path = get_model_ckpt_name_for_rank(base_folder, f'{file_id:02d}')

    # Adjust specific fields
    sd[ARGS_KEY] = ds_checkpoint.get_args()
    sd[ARGS_KEY].tensor_model_parallel_size = ds_checkpoint.tp_degree
    sd[ARGS_KEY].pipeline_model_parallel_size = ds_checkpoint.pp_degree
    if CHECKPOINT_INFO_KEY in sd:
        sd[CHECKPOINT_INFO_KEY][PADDED_VOCAB_SIZE] = sd[ARGS_KEY].padded_vocab_size
    _save_checkpoint(ckpt_path, sd)

def _create_latest_file(base_folder, file_name, latest_tag):
    file_path = os.path.join(base_folder, file_name)
    os.makedirs(base_folder, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(str(latest_tag))

def main():
    args = parse_arguments()
    print(
        f'Converting DeepSpeed checkpoint in {args.input_folder} to DeepSpeed checkpoint in {args.output_folder}'
    )

    ds_checkpoint = DeepSpeedCheckpointNoZeRO(
        args.input_folder,
        args.target_tp,
        args.target_pp)
    iteration = ds_checkpoint.get_iteration()
    latest_tag = f'global_step{iteration}'
    _create_latest_file(args.output_folder,
                        'latest_checkpointed_iteration.txt', iteration)
    _create_latest_file(args.output_folder, 'latest', latest_tag)
    base_folder = os.path.join(args.output_folder, latest_tag)

    for i in range(ds_checkpoint.tp_degree):
        _create_embedding_layer_checkpoint(ds_checkpoint, base_folder, i, original_vocab_size=args.original_vocab_size)
        _create_final_norm_layer_checkpoint(ds_checkpoint, base_folder, i)

        for j in range(ds_checkpoint.pp_degree):
            _create_transformer_layer_checkpoint(ds_checkpoint, base_folder, i, j)
            _create_2d_parallel_checkpoint(ds_checkpoint, base_folder, i, j)

if __name__ == "__main__":
    main()

