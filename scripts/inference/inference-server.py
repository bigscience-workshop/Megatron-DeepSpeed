import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from typing import Tuple

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import deepspeed

from flask import Flask, request
from waitress import serve


class MaxTokensError(Exception):
    def __init__(self, max_new_tokens: int, allowed_max_new_tokens: int) -> None:
        self.message = "max_new_tokens ({}) > {} is not supported.".format(
            max_new_tokens, allowed_max_new_tokens)


# TODO remove when bloom-inference is merged into main
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


def write_checkponts_json(args: argparse.Namespace):
    #checkpoint_dir = "/gpfsscratch/rech/six/commun/uan68tv-model-conversion/bloom"
    #checkpoint_files = glob.glob(f"{checkpoint_dir}/*bin")
    checkpoint_files = get_checkpoint_files(args.model_name)

    #print("Checkpoint files:", checkpoint_files)

    data = {
        "type": "BLOOM-176B",
        "checkpoints": checkpoint_files,
        "version": 1.0
    }
    json.dump(data, open(args.checkpoints_json, "w", encoding="utf-8"))


# TODO remove when bloom-inference is merged into main
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


def ParseArgs():
    parser = argparse.ArgumentParser(description="Text generation server")

    group = parser.add_argument_group(title="model")
    group.add_argument("--model_name", type=str,
                       required=True, help="model to use")
    group.add_argument("--log_file", type=str, help="log data")
    group.add_argument("--host", type=str, required=True, help="host address")
    group.add_argument("--port", type=int, required=True, help="port number")
    group.add_argument("--dtype", type=str, required=True,
                       choices=["bf16", "fp16"], help="dtype for model")
    group.add_argument("--inference_method", type=str, required=True,
                       choices=["hf_accelerate", "deepspeed"], help="inference method to use")

    group = parser.add_argument_group(title="limitting values")
    group.add_argument("--allowed_max_new_tokens", type=int,
                       default=100, help="max allowed tokens")

    group = parser.add_argument_group(title="default values")
    group.add_argument("--top_k", type=int, default=50, help="default top_k")
    group.add_argument("--top_p", type=float, default=1, help="default top_p")
    group.add_argument("--temperature", type=float,
                       default=1, help="default temperature")
    group.add_argument("--min_length", type=int, default=1, help="min length")
    group.add_argument("--max_new_tokens", type=int,
                       default=40, help="max new tokens")
    group.add_argument("--return_type", type=str, default="both_input_output",
                       choices=["both_input_output", "output_only"], help="return type")

    args = parser.parse_args()

    if (args.dtype == "bf16"):
        args.dtype = torch.bfloat16
    elif (args.dtype == "fp16"):
        args.dtype = torch.float16

    if (args.inference_method == "deepspeed"):
        args.checkpoints_json = "checkpoints.json"

    return args


def GetStackTrace(e_stack_trace):
    trace_back = traceback.extract_tb(e_stack_trace)

    # Format stacktrace
    stack_trace = []
    for trace in trace_back:
        stack_trace.append("File : {}, Line : {}, Func.Name : {}, Message : {}".format(
            trace[0], trace[1], trace[2], trace[3]))

    return stack_trace


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        print("Loading model...")

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        if (args.inference_method == "hf_accelerate"):
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map="auto",
                max_memory=get_max_memory_per_gpu_dict(
                    args.dtype, args.model_name),
                torch_dtype=args.dtype
            )
        elif (args.inference_method == "deepspeed"):
            self.config = AutoConfig.from_pretrained(args.model_name)

            with deepspeed.OnDevice(dtype=args.dtype, device="meta"):
                self.model = AutoModelForCausalLM.from_config(
                    self.config, torch_dtype=args.dtype)

        self.model.eval()

        if (args.inference_method == "deepspeed"):
            world_size = int(os.getenv("WORLD_SIZE", "1"))

            if (args.dtype == torch.float16):
                self.model = deepspeed.init_inference(
                    self.model,
                    mp_size=world_size,
                    dtype=args.dtype,
                    checkpoint=args.checkpoints_json,
                    replace_with_kernel_inject=True
                )
            elif (args.dtype == torch.bfloat16):
                raise NotImplementedError("This is not yet finished")
                # self.model = deepspeed.init_inference(
                #     self.model,
                #     mp_size=world_size,
                #     dtype=args.dtype,
                #     checkpoint=args.checkpoints_json,
                #     injection_policy={
                #         BloomBlock: (
                #             'self_attention.dense',
                #             'mlp.dense_4h_to_h'
                #         )
                #     }
                # )

            self.model = self.model.module

        if (args.inference_method == "hf_accelerate"):
            self.input_device = "cuda:0"
        elif (args.inference_method == "deepspeed"):
            self.input_device = torch.cuda.current_device()

        # optimize model by generating once (optimization happens on the first run)
        self.Generate("Hi, I'm a model", 5, 0.9, 0.7, 1, 40)

        print("Model loaded")

    def Generate(self,
                 text: str,
                 top_k: int,
                 top_p: float,
                 temperature: float,
                 min_length: int,
                 max_new_tokens: int,
                 remove_input_from_output: bool = False) -> Tuple[str, int]:
        x = self.tokenizer([text])

        input_ids = torch.tensor(x["input_ids"]).to(self.input_device)
        attention_mask = torch.tensor(
            x["attention_mask"]).to(self.input_device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                max_new_tokens=max_new_tokens
            )

        output_tokens = output[0]

        if (remove_input_from_output):
            output_tokens = output_tokens[len(input_ids[0]):]
            num_output_tokens = len(output_tokens)
        else:
            num_output_tokens = len(output_tokens) - len(input_ids[0])

        output_text = self.tokenizer.decode(output_tokens)

        return output_text, num_output_tokens


####################################################################################
args = ParseArgs()
app = Flask(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename=args.log_file
)
logger = logging.getLogger(__name__)

if (args.inference_method == "deepspeed"):
    deepspeed.init_distributed("nccl")
    args.rank = dist.get_rank()
    if (args.rank == 0):
        write_checkponts_json(args)
    dist.barrier()

model = Model(args)
####################################################################################


@app.route("/gpu/", methods=["GET"])
def gpu() -> str:
    try:
        info = subprocess.check_output(["nvidia-smi"])
        info = info.decode("utf8")
    except Exception as e:
        info = "Executing nvidia-smi failed: " + str(e)
    return info


@app.route("/about/", methods=["GET"])
def about() -> str:
    if (args.log_file):
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
    "max_new_tokens": 40,
    "return_type": "output_only",
}

Dafault values (use if not provided in request object):
top_k = 50
top_p = 1
temperature = 1
min_length = 1
max_new_tokens = 40
return_type = "both_input_output"
'''
    return description


@app.route("/generate/", methods=["POST"])
def generate() -> str:
    try:
        start_time = time.time()
        json_obj = request.get_json()

        input_text = str(json_obj["input_text"])
        top_k = int(json_obj.get("top_k", args.top_k))
        top_p = float(json_obj.get("top_p", args.top_p))
        temperature = float(json_obj.get("temperature", args.temperature))
        min_length = int(json_obj.get("min_length", args.min_length))
        max_new_tokens = int(json_obj.get(
            "max_new_tokens", args.max_new_tokens))
        return_type = str(json_obj.get("return_type", args.return_type))

        if (max_new_tokens > args.allowed_max_new_tokens):
            raise MaxTokensError(max_new_tokens, args.allowed_max_new_tokens)

        output_text, num_output_tokens = model.Generate(
            input_text,
            top_k,
            top_p,
            temperature,
            min_length,
            max_new_tokens,
            remove_input_from_output=(return_type == "output_only")
        )

        total_time_taken = time.time() - start_time
        output = {
            "output_text": output_text,
            "num_output_tokens": num_output_tokens,
            "total_time_taken": "{:.3f} s".format(total_time_taken),
            "throughput": "{:.3f} tokens/s".format(num_output_tokens / total_time_taken)
        }
        if (args.log_file):
            logger.info(json_obj)
            logger.info(output)
    except Exception:
        e_type, e_message, e_stack_trace = sys.exc_info()
        output = {
            "error": {
                "error": str(e_type.__name__),
                "message": str(e_message),
                "stack_trace": GetStackTrace(e_stack_trace)
            },
            "time_taken": "{} s".format(str(time.time() - start_time))
        }
        if (args.log_file):
            logger.info(json_obj)
            logger.error(output)
        del output["error"]["stack_trace"]

    return output


def main():
    serve(app, host=args.host, port=args.port)


if (__name__ == "__main__"):
    main()
