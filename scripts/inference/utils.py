import io
import json
import subprocess
import traceback
from argparse import Namespace
from typing import Any, List, Tuple

import torch
import torch.distributed as dist
from transformers import AutoConfig


def gpu_status():
    try:
        info = subprocess.check_output(["nvidia-smi"])
        info = info.decode("utf8")
    except Exception as e:
        info = "Executing nvidia-smi failed: " + str(e)
    return info


def about(log_file: str) -> str:
    if (log_file):
        description = "Please don't send any personal information to this endpoint. We are logging your data.\n\n"
    else:
        description = ""
    description += '''Usage:
A request object should look like:
{
    input_text: "Hello, I'm a model",
    "top_k": 5,
    "top_p": 0.9,
    "temperature": 0.7,
    "min_length": 1,
    "max_new_tokens": 40
}
Default values (use if not provided in request object):
top_k = 50
top_p = 1
temperature = 1
min_length = 1
max_new_tokens = 40
'''
    return description


### Model loading and instantiating on GPUs
def get_checkpoint_files(pretrained_model_name_or_path):
    # XXX: I just hacked this one together to automatically handle the fetching of the model file or
    # shards into cache and returning the cached entries - note that I removed most arguments

    from transformers.modeling_utils import get_checkpoint_shard_files
    from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, cached_path, hf_bucket_url, is_offline_mode
    from transformers.utils.hub import EntryNotFoundError

    cache_dir = None
    is_sharded = False

    # XXX: preparation for revision branches if needed
    revision = None
    #revision = "sharded"

    # this supports nodes with no network (so you need to pre-cache the model and the tokenizer with
    # python -c "from transformers import AutoModel; AutoModel.from_pretrained('bigscience/bloom')"
    if is_offline_mode():
        print("Offline mode: forcing local_files_only=True")
        local_files_only = True
    else:
        local_files_only = False

    filename = WEIGHTS_NAME
    archive_file = hf_bucket_url(
        pretrained_model_name_or_path, filename=filename, revision=revision)

    try:
        resolved_archive_file = cached_path(
            archive_file, cache_dir=cache_dir, local_files_only=local_files_only,)
        return [resolved_archive_file]

    except (EntryNotFoundError, FileNotFoundError):
        if filename == WEIGHTS_NAME:
            # Maybe the checkpoint is sharded, we try to grab the index name in this case.
            archive_file = hf_bucket_url(
                pretrained_model_name_or_path,
                filename=WEIGHTS_INDEX_NAME,
                revision=revision,
            )
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            is_sharded = True

    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            revision=revision
        )

        return resolved_archive_file


def get_stack_trace(e_stack_trace):
    trace_back = traceback.extract_tb(e_stack_trace)

    # Format stacktrace
    stack_trace = []
    for trace in trace_back:
        stack_trace.append("File : {}, Line : {}, Func.Name : {}, Message : {}".format(
            trace[0], trace[1], trace[2], trace[3]))

    return stack_trace


class MaxTokensError(Exception):
    def __init__(self, max_new_tokens: int, allowed_max_new_tokens: int) -> None:
        super().__init__("max_new_tokens = {} > {} is not supported.".format(
            max_new_tokens, allowed_max_new_tokens))


def run_rank_n(func: callable,
               kwargs: dict,
               barrier: bool = False,
               rank: int = 0) -> Any:
    if (dist.is_initialized()):
        if (dist.get_rank() == rank):
            output = func(**kwargs)
            if (barrier):
                dist.barrier()
            return output
        else:
            if (barrier):
                dist.barrier()
    else:
        return func(**kwargs)


def print_rank_n(*values, rank: int = 0) -> None:
    if (dist.is_initialized()):
        if (dist.get_rank() == rank):
            print(*values)
    else:
        print(*values)


def get_max_memory_per_gpu_dict(dtype, model_name):
    """ try to generate the memory map based on what we know about the model and the available hardware """

    # figure out the memory map - the minimum per gpu required to load the model
    n_gpus = torch.cuda.device_count()

    if model_name == "bigscience/bloom" and n_gpus == 8 and torch.cuda.get_device_properties(0).total_memory > 79*2**30:
        # hand crafted optimized memory map for 8x80 setup over BLOOM
        # this works with bs=40
        return {0: '0GIB', 1: '51GIB', 2: '51GIB', 3: '51GIB', 4: '51GIB', 5: '51GIB', 6: '51GIB', 7: '51GIB'}

    try:
        # model_params calculation, as we don't have a model yet to do:
        #model_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

        config = AutoConfig.from_pretrained(model_name)
        h = config.n_embed
        l = config.n_layer
        v = config.vocab_size
        # from https://github.com/bigscience-workshop/bigscience/tree/6917a3b5fefcf439d3485ca184b4d9f6ab605150/math#model-sizing
        model_params = l*(12*h**2 + 13*h) + v*h + 4*h
    except:
        print(
            f"The model {model_name} has a broken config file. Please notify the owner")
        raise

    bytes = torch.finfo(dtype).bits / 8
    param_memory_total_in_bytes = model_params * bytes
    # add 5% since weight sizes aren't the same and some GPU may need more memory
    param_memory_per_gpu_in_bytes = int(
        param_memory_total_in_bytes / n_gpus * 1.05)
    print(
        f"Estimating {param_memory_per_gpu_in_bytes/2**30:0.2f}GB per gpu for weights")

    # check the real available memory
    # load cuda kernels first and only measure the real free memory after loading (shorter by ~2GB)
    torch.ones(1).cuda()
    max_memory_per_gpu_in_bytes = torch.cuda.mem_get_info(0)[0]
    if max_memory_per_gpu_in_bytes < param_memory_per_gpu_in_bytes:
        raise ValueError(
            f"Unable to generate the memory map automatically as the needed estimated memory per gpu ({param_memory_per_gpu_in_bytes/2**30:0.2f}GB) is bigger than the available per gpu memory ({max_memory_per_gpu_in_bytes/2**30:0.2f}GB)")

    return {i: param_memory_per_gpu_in_bytes for i in range(torch.cuda.device_count())}


def write_checkponts_json(model_name: str, checkpoints_json: str) -> None:
    with io.open(checkpoints_json, 'w', encoding='utf-8') as f:
        #checkpoint_dir = "/gpfsscratch/rech/six/commun/uan68tv-model-conversion/bloom"
        #checkpoint_files = glob.glob(f"{checkpoint_dir}/*bin")
        checkpoint_files = get_checkpoint_files(model_name)

        #print("Checkpoint files:", checkpoint_files)

        data = {
            "type": "BLOOM-176B",
            "checkpoints": checkpoint_files,
            "version": 1.0
        }
        json.dump(data, f)


def generate_(inputs,
              model,
              tokenizer,
              generate_kwargs,
              input_device):
    """ returns a list of zipped inputs, outputs and number of new tokens """

    input_tokens = tokenizer.batch_encode_plus(
        inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(input_device)

    outputs = model.generate(**input_tokens, **generate_kwargs)

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o-i for i,
                        o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)


def get_torch_dtype(dtype_str: str):
    if (dtype_str == "bf16"):
        return torch.bfloat16
    elif (dtype_str == "fp16"):
        return torch.float16


def get_str_dtype(dtype_str: str):
    if (dtype_str == torch.bfloat16):
        return "bf16"
    elif (dtype_str == torch.float16):
        return "fp16"


def parse_input(json_obj: dict,
                args: Namespace) -> Tuple[List[str],
                                          int,
                                          float,
                                          float,
                                          int,
                                          int,
                                          str]:
    input_text = json_obj["input_text"]
    top_k = int(json_obj.get("top_k", args.top_k))
    top_p = float(json_obj.get("top_p", args.top_p))
    temperature = float(json_obj.get("temperature", args.temperature))
    min_length = int(json_obj.get("min_length", args.min_length))
    max_new_tokens = int(json_obj.get(
        "max_new_tokens", args.max_new_tokens))

    return (
        input_text,
        top_k,
        top_p,
        temperature,
        min_length,
        max_new_tokens
    )
