#!/usr/bin/env python

# at the moment this is very much a quick hack to replace half-precision weights with fp32 weights in the existing HF transformers checkpoint seeded from the universal checkpoint

# 1. create a normal Meg-DS checkpoint
#
# 2. convert to universal
#
# python tools/convert_checkpoint/ds_to_universal.py --input_folder checkpoints/gpt2/global_step3 --output_folder checkpoints/gpt2/global_step3_universal
#
# # 3. convert to hf checkpoint or clone an existing one
#
# python ../transformers-master/src/transformers/models/bloom/convert_bloom_original_checkpoint_to_pytorch.py --bloom_checkpoint_path checkpoints/gpt2/global_step3 --pytorch_dump_folder_path checkpoints/gpt2/global_step3_hf  --pretraining_tp 1
#
#
# # needed to hack - or need to come up with a json config file
#         config = BloomConfig()
#     else:
#         config = BloomConfig.from_json_file(bloom_config_file)

#     config.hidden_size = 8
#     config.n_head = 2
#     config.n_layers = 4
#     config.vocab_size = 50304
#     print(config)
#
# 4. replace half-precision weights with fp32 weights
# python tools/convert_checkpoint/universal_to_fp32_checkpoint.py --universal_path checkpoints/gpt2/global_step3_universal --hf_half_path checkpoints/gpt2/global_step3_hf --hf_fp32_path checkpoints/gpt2/global_step3_hf_fp32


from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import glob
import os
import re
import shutil
import torch

parser = ArgumentParser()
parser.add_argument("--hf_half_path", required=True, type=str, help="path to the HF half path checkpoint")
parser.add_argument("--universal_path", required=True, type=str, help="path to the universal checkpoint")
parser.add_argument("--hf_fp32_path", required=True, type=str, help="path to the fp32 version output")
args = parser.parse_args()

hf_half_path = args.hf_half_path
universal_path = args.universal_path
hf_fp32_path = args.hf_fp32_path

# adapted from the conversion script
def layer_name_mapping(key):
    """ map Megatron-DeepSpeed weights to transformers """
    # Handle first and last layers
    layer_rename_map = {
        "tied_modules.embed.word_embeddings.weight":      "word_embeddings.weight",
        "tied_modules.embed.word_embeddings.norm.weight": "word_embeddings_layernorm.weight",
        "tied_modules.embed.word_embeddings.norm.bias":   "word_embeddings_layernorm.bias",
        "tied_modules.embed.position_embeddings.weight":   "word_embeddings_layernorm.bias",
        "weight": "ln_f.weight",
        "bias":   "ln_f.bias",
    }

    if key in layer_rename_map:
        return layer_rename_map[key]

    layer_rename_map2 = {
        "weight": "ln_f.weight",
        "bias":   "ln_f.bias",
    }

    segments = re.split("\.", key)
    if len(segments) == 2:
        return layer_rename_map2[segments[1]]

    # Handle transformer blocks
    try:
        layer_number, *rest = re.split("\.", key)
        layer_number = str(int(layer_number) - 3)
        return ".".join(["h", layer_number] + rest)
    except:
        return key

# universal checkpoint name remap
ds_layer_names = sorted(os.listdir(f"{universal_path}/zero"))
#ds_layer_names = glob.glob(f"{universal_path}/zero/*", root_dir=f"{universal_path}/zero")
pprint(ds_layer_names)

key_map = {layer_name_mapping(key):key for key in ds_layer_names}
pprint(key_map)

# copy non-weight files
Path(hf_fp32_path).mkdir(parents=True, exist_ok=True)
hf_files = [x for x in os.listdir(hf_half_path) if not x.endswith("bin") and os.path.isfile(x)]
pprint(hf_files)
for f in hf_files:
    shutil.copy2(f"{hf_half_path}/{f}", f"{hf_fp32_path}/{f}")

# replace half precision with fp32 weights
hf_checkpoint_files = glob.glob(f"{hf_half_path}/*bin")
pprint(hf_checkpoint_files)
for f in hf_checkpoint_files:
    sd = torch.load(f, map_location="cpu")
    for k in sd.keys():
        fp32_path = f"{universal_path}/zero/{key_map[k]}/fp32.pt"
        print(k, fp32_path)
        new_value = torch.load(fp32_path, map_location="cpu")
        sd[k] = new_value
    f = f.replace(hf_half_path, hf_fp32_path)
    torch.save(sd, f)



# tokenizer = AutoTokenizer.from_pretrained(mname)
# tokenizer.save_pretrained(hf_fp32_path)

config = AutoConfig.from_pretrained(hf_half_path)
# replicate the existing tiny model but we need longer max_position_embeddings
config.update(dict(
    torch_dtype="float32"
))
config.save_pretrained(hf_fp32_path)


# tiny_model = AutoForConditionalGeneration(config)
# print(f"num of params {tiny_model.num_parameters()}")

# # Test
# model_inputs = tokenizer("Making tiny model", return_tensors="pt")
# gen_tokens = tiny_model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("fr"))
# print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
# #

# # Save
# mname_tiny = "tiny-m2m_100"
# tiny_model.half() # makes it smaller
# tiny_model.save_pretrained(mname_tiny)
# tokenizer.save_pretrained(mname_tiny)

# print(f"Generated {mname_tiny}")
