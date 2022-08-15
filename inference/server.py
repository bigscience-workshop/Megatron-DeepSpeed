import argparse
import logging
import time
from typing import List, Union

import constants
import utils
from ds_inference import DSInferenceGRPCServer
from fastapi import FastAPI, HTTPException
from hf_accelerate import HFAccelerateModel
from pydantic import BaseModel
from utils import MaxTokensError, get_argument_parser, parse_generate_kwargs
from uvicorn import run


class GenerateRequest(BaseModel):
    text: Union[List[str], str]
    generate_kwargs: dict


class GenerateResponse(BaseModel):
    text: Union[List[str], str]
    num_generated_tokens: Union[List[int], int]
    query_id: int
    total_time_taken: float


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
    group.add_argument("--log_file", type=str, help="log data")
    group.add_argument("--host", type=str, required=True, help="host address")
    group.add_argument("--port", type=int, required=True, help="port number")
    group.add_argument("--workers", type=int, default=1,
                       help="number of http workers")
    group.add_argument("--allowed_max_new_tokens", type=int,
                       default=100, help="max allowed tokens")

    args = utils.get_args(parser)

    return args


####################################################################################
args = get_args()
app = FastAPI()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename=args.log_file
)
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

        text = request.text
        generate_kwargs = args.generate_kwargs
        remove_input_from_output = False
        if (request.generate_kwargs):
            generate_kwargs, remove_input_from_output = parse_generate_kwargs(
                request.generate_kwargs)

        if (generate_kwargs["max_new_tokens"] > args.allowed_max_new_tokens):
            raise MaxTokensError(
                generate_kwargs["max_new_tokens"], args.allowed_max_new_tokens)

        output_text, num_generated_tokens = model.generate(
            text,
            generate_kwargs,
            remove_input_from_output=remove_input_from_output
        )

        total_time_taken = time.time() - start_time

        output = GenerateResponse(
            text=output_text,
            num_generated_tokens=num_generated_tokens,
            query_id=query_id,
            total_time_taken=total_time_taken
        )
        query_id += 1
        return output
    except Exception as e:
        query_id += 1
        raise HTTPException(500, {"error": str(e)})


run(app, host=args.host, port=args.port, workers=args.workers)
