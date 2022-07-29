import argparse
import logging
import subprocess
import sys
import time
import traceback
from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from flask import Flask, request
from waitress import serve


class MaxTokensError(Exception):
    def __init__(self, max_new_tokens: int, max_allowed_tokens: int) -> None:
        self.message = "max_new_tokens ({}) > {} is not supported.".format(
            max_new_tokens, max_allowed_tokens)


# TODO remove when bloom-oinference is merged into main
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
    group.add_argument("--dtype", type=str, required=True, help="port number")
    group.add_argument("--max_allowed_tokens", type=int,
                       default=100, help="max allowed tokens")

    args = parser.parse_args()

    if (args.dtype == "bf16"):
        args.dtype = torch.bfloat16
    elif (args.dtype == "fp16"):
        args.dtype = torch.float16
    elif (args.dtype == "fp32"):
        args.dtype = torch.float32

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
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            max_memory=get_max_memory_per_gpu_dict(
                args.dtype, args.model_name),
            torch_dtype=args.dtype
        )
        self.model.eval()

        print("Model loaded")

    def Generate(self,
                 text: str,
                 top_k: int,
                 top_p: float,
                 temperature: float,
                 min_length: int,
                 max_new_tokens: int) -> Tuple[str, int]:
        x = self.tokenizer([text])

        input_ids = torch.tensor(x["input_ids"])
        attention_mask = torch.tensor(x["attention_mask"])

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
        output_text = self.tokenizer.decode(output_tokens)

        return output_text, len(output_tokens)


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
model = Model(args)
####################################################################################


@app.route("/gpu/", methods=["GET"])
def gpu() -> str:
    try:
        info = subprocess.check_output(["nvidia-smi"])
        info = info.decode("utf8")
    except Exception as e:
        info = "Executing nvidia-smi failed: " + str(e)
    return info.strip()


@app.route("/generate/", methods=["POST"])
def generate() -> str:
    try:
        start_time = time.time()
        json_obj = request.get_json()

        input_text = str(json_obj["input_text"])
        top_k = int(json_obj["top_k"])
        top_p = float(json_obj["top_p"])
        temperature = float(json_obj["temperature"])
        min_length = int(json_obj["min_length"])
        max_new_tokens = int(json_obj["max_new_tokens"])

        if (max_new_tokens > args.max_allowed_tokens):
            raise MaxTokensError(max_new_tokens, args.max_allowed_tokens)

        output_text, num_output_tokens = model.Generate(
            input_text,
            top_k,
            top_p,
            temperature,
            min_length,
            max_new_tokens
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
