#!/usr/bin/env python

# Usage:
# python tools/convert_checkpoint/deepspeed_to_transformers.py --input_folder path --output_folder path

import os
import torch

from megatron.checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
from tools.convert_checkpoint.deepspeed_to_megatron import _create_rank_checkpoint, parse_arguments

# the import was tested to work with this version
# https://github.com/huggingface/transformers/commit/0af901e83 if it diverges we may consider
# copying that version here instead
from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint import (
    convert_megatron_checkpoint,
)
from transformers import AutoTokenizer, GPT2Config


def main():
    # this first part comes mainly from deepspeed_to_megatron.main
    args = parse_arguments()
    print(
        f"Converting DeepSpeed checkpoint in {args.input_folder} to HF Transformers checkpoint in {args.output_folder}"
    )

    ds_checkpoint = DeepSpeedCheckpoint(
        args.input_folder, args.target_tp, args.target_pp
    )
    ds_args = ds_checkpoint.get_args()
    input_state_dict = _create_rank_checkpoint(ds_checkpoint, 0, 0, args.for_release)

    # the 2nd part comes from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint.main
    # Spell out all parameters in case the defaults change.
    if ds_args.bias_gelu_fusion:
        activation_function = "gelu_fast"
    elif ds_args.openai_gelu:
        activation_function = "gelu_new"
    else:
        activation_function = "gelu"

    config = GPT2Config(
        architectures=["GPT2LMHeadModel"],
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=1024,
        n_layer=24,
        n_head=16,
        n_inner=4096,
        activation_function=activation_function,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        gradient_checkpointing=False,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
    )

    # Convert.
    print("Converting to HF Checkpoint")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)

    basename = args.output_folder
    os.makedirs(basename, exist_ok=True)

    # Print the structure of converted state dict.
    # if args.print_checkpoint_structure:
    #    recursive_print(None, output_state_dict)

    # Add tokenizer class info to config
    # see https://github.com/huggingface/transformers/issues/13906)
    tokenizer_type = ds_args.tokenizer_type
    if tokenizer_type == "GPT2BPETokenizer":
        tokenizer_model_name = "gpt2"
    elif tokenizer_type == "PretrainedFromHF":
        tokenizer_model_name = ds_args.tokenizer_name_or_path
    else:
        raise ValueError(f"Unrecognized tokenizer_type {tokenizer_type}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    tokenizer_class = type(tokenizer).__name__
    config.tokenizer_class = tokenizer_class

    # Store the config to file.
    print("Saving config")
    config.save_pretrained(basename)

    # Save tokenizer based on args
    print(f"Adding {tokenizer_class} tokenizer files")
    tokenizer.save_pretrained(basename)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)



if __name__ == "__main__":
    main()
