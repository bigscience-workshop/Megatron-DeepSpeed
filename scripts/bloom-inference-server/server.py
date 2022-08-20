import argparse
import logging
import sys
import time

import constants
import utils
from ds_inference import DSInferenceGRPCServer
from fastapi import FastAPI, HTTPException
from hf_accelerate import HFAccelerateModel
from utils import GenerateRequest, get_argument_parser, get_num_tokens_to_generate, get_stack_trace
from uvicorn import run


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

query_id = 0
####################################################################################


@app.post("/generate/")
def generate(request: GenerateRequest) -> dict:
    # needs to be global since it is updated
    global query_id

    try:
        start_time = time.time()

        request.max_new_tokens = get_num_tokens_to_generate(
            request.max_new_tokens, args.allowed_max_new_tokens)

        response = model.generate(request)
        response.query_id = query_id
        response.total_time_taken = time.time() - start_time

        query_id += 1
        return response
    except Exception:
        e_type, e_message, e_stack_trace = sys.exc_info()
        response = {
            "error": str(e_type.__name__),
            "message": str(e_message),
            "query_id": query_id
        }

        if (args.debug):
            response["stack_trace"] = get_stack_trace(e_stack_trace)

        query_id += 1
        raise HTTPException(500, response)


run(app, host=args.host, port=args.port, workers=args.workers)
