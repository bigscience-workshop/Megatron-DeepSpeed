import argparse
import logging
import os
import sys
import traceback

import constants
import utils
from ds_inference import DSInferenceGRPCServer
from fastapi import FastAPI, HTTPException
from hf_accelerate import HFAccelerateModel
from pydantic import BaseModel
from utils import (
    GenerateRequest,
    GenerateResponse,
    TokenizeRequest,
    TokenizeResponse,
    get_argument_parser,
    get_num_tokens_to_generate,
    run_and_log_time
)
from uvicorn import run


class QueryID(BaseModel):
    generate_query_id: int = 0
    tokenize_query_id: int = 0


def get_args() -> argparse.Namespace:
    parser = get_argument_parser()

    group = parser.add_argument_group(title="launch config")
    group.add_argument(
        "--deployment_framework",
        type=str,
        choices=[
            constants.HF_ACCELERATE,
            constants.DS_INFERENCE,
        ],
        default=constants.HF_ACCELERATE
    )
    group.add_argument("--save_mp_checkpoint_path", required=False,
                       type=str, help="MP checkpoints path for DS inference")
    group.add_argument("--host", type=str, required=True, help="host address")
    group.add_argument("--port", type=int, required=True, help="port number")
    group.add_argument("--workers", type=int, default=1,
                       help="number of http workers")
    group.add_argument("--allowed_max_new_tokens", type=int,
                       default=100, help="max allowed tokens")
    group.add_argument("--debug", action="store_true",
                       help="launch in debug mode")

    args = utils.get_args(parser)

    if (args.save_mp_checkpoint_path):
        assert args.deployment_framework == constants.DS_INFERENCE, "save_mp_checkpoint_path only works with DS inference"

    return args


####################################################################################
args = get_args()
app = FastAPI()

logger = logging.getLogger(__name__)

if (args.deployment_framework == constants.HF_ACCELERATE):
    model = HFAccelerateModel(args)
elif (args.deployment_framework == constants.DS_INFERENCE):
    model = DSInferenceGRPCServer(args)
else:
    raise ValueError(
        f"Unknown deployment framework {args.deployment_framework}")

query_ids = QueryID()
####################################################################################


def get_exception_response(query_id: int, method: str):
    e_type, e_message, e_stack_trace = sys.exc_info()
    response = {
        "error": str(e_type.__name__),
        "message": str(e_message),
        "query_id": query_id,
        "method": method
    }

    if (args.debug):
        trace_back = traceback.extract_tb(e_stack_trace)

        # Format stacktrace
        stack_trace = []
        for trace in trace_back:
            stack_trace.append("File : {}, Line : {}, Func.Name : {}, Message : {}".format(
                trace[0], trace[1], trace[2], trace[3]))

        response["stack_trace"] = stack_trace

    return response


@app.post("/generate/")
def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        request.max_new_tokens = get_num_tokens_to_generate(
            request.max_new_tokens, args.allowed_max_new_tokens)

        response, total_time_taken = run_and_log_time(
            (model.generate, {"request": request})
        )

        response.query_id = query_ids.generate_query_id
        query_ids.generate_query_id += 1
        response.total_time_taken = "{:.2f} secs".format(total_time_taken)

        return response
    except Exception:
        response = get_exception_response(
            query_ids.generate_query_id, request.method)
        query_ids.generate_query_id += 1
        raise HTTPException(500, response)


@app.post("/tokenize/")
def tokenize(request: TokenizeRequest) -> TokenizeResponse:
    try:
        response, total_time_taken = run_and_log_time(
            (model.tokenize, {"request": request})
        )

        response.query_id = query_ids.tokenize_query_id
        query_ids.tokenize_query_id += 1
        response.total_time_taken = "{:.2f} msecs".format(
            total_time_taken * 1000)

        return response
    except Exception:
        response = get_exception_response(
            query_ids.tokenize_query_id, request.method)
        query_ids.tokenize_query_id += 1
        raise HTTPException(500, response)


@app.get("/query_id/")
def query_id() -> QueryID:
    return query_ids


try:
    run(app, host=args.host, port=args.port, workers=args.workers)
except KeyboardInterrupt:
    model.shutdown()
