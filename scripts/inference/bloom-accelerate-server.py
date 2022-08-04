import argparse
import logging
import sys
import time
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils
from flask import Flask, request
from utils import MaxTokensError, get_max_memory_per_gpu_dict, get_stack_trace
from waitress import serve


def ParseArgs():
    parser = argparse.ArgumentParser(description="Text generation server")

    group = parser.add_argument_group(title="model")
    group.add_argument("--model_name", type=str,
                       required=True, help="model to use")
    group.add_argument("--dtype", type=str, required=True,
                       choices=["bf16", "fp16"], help="dtype for model")

    group = parser.add_argument_group(title="launch config")
    group.add_argument("--log_file", type=str, help="log data")
    group.add_argument("--host", type=str, required=True, help="host address")
    group.add_argument("--port", type=int, required=True, help="port number")

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

    return args


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        print("Loading model...")

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            max_memory=get_max_memory_per_gpu_dict(
                args.dtype, args.model_name),
            torch_dtype=args.dtype
        )

        self.model.eval()
        self.input_device = "cuda:0"

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

model = Model(args)
query_id = 0
####################################################################################


@app.route("/gpu/", methods=["GET"])
def gpu() -> str:
    utils.gpu()


@app.route("/about/", methods=["GET"])
def about() -> str:
    utils.about(args.log_file)


@app.route("/generate/", methods=["POST"])
def generate() -> str:
    # needs to be global since it is updated
    global query_id

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
        json_obj["query_id"] = query_id

        total_time_taken = time.time() - start_time
        output = {
            "output_text": output_text,
            "num_output_tokens": num_output_tokens,
            "total_time_taken": "{:.3f} s".format(total_time_taken),
            "throughput": "{:.3f} tokens/s".format(num_output_tokens / total_time_taken),
            "query_id": query_id
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
                "stack_trace": get_stack_trace(e_stack_trace)
            },
            "time_taken": "{} s".format(str(time.time() - start_time)),
            "query_id": query_id
        }
        if (args.log_file):
            logger.info(json_obj)
            logger.error(output)
        del output["error"]["stack_trace"]

    query_id += 1

    return output


def main():
    serve(app, host=args.host, port=args.port)


if (__name__ == "__main__"):
    main()
