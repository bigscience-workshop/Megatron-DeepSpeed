import argparse
import sys
import traceback
from functools import partial

import utils
from ds_inference import DSInferenceGRPCServer
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from hf_accelerate import HFAccelerateModel
from pydantic import BaseModel
from utils import (
    DS_INFERENCE,
    HF_ACCELERATE,
    SERVER,
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
    group.add_argument("--host", type=str, required=True, help="host address")
    group.add_argument("--port", type=int, required=True, help="port number")
    group.add_argument("--workers", type=int, default=1,
                       help="number of http workers")
    group.add_argument("--allowed_max_new_tokens", type=int,
                       default=100, help="max allowed tokens")
    group.add_argument("--debug", action="store_true",
                       help="launch in debug mode")

    args = utils.get_args(parser, SERVER)

    return args


class Server:
    def __init__(self, args: argparse.Namespace):
        self.host = args.host
        self.port = args.port
        self.workers = args.workers
        self.debug = args.debug

        self.allowed_max_new_tokens = args.allowed_max_new_tokens
        self.query_ids = QueryID()

        if (args.deployment_framework == HF_ACCELERATE):
            self.model = HFAccelerateModel(args)
        elif (args.deployment_framework == DS_INFERENCE):
            self.model = DSInferenceGRPCServer(args)
        else:
            raise ValueError(
                f"Unknown deployment framework {args.deployment_framework}")

        self.app = FastAPI(
            routes=[
                APIRoute(
                    "/generate/",
                    self.generate,
                    methods=["POST"],
                ),
                APIRoute(
                    "/tokenize/",
                    self.tokenize,
                    methods=["POST"],
                ),
                APIRoute(
                    "/query_id/",
                    self.query_id,
                    methods=["GET"],
                )
            ],
            timeout=600,
        )

    def get_exception_response(self, query_id: int, method: str):
        e_type, e_message, e_stack_trace = sys.exc_info()
        response = {
            "error": str(e_type.__name__),
            "message": str(e_message),
            "query_id": query_id,
            "method": method
        }

        if (self.debug):
            trace_back = traceback.extract_tb(e_stack_trace)

            # Format stacktrace
            stack_trace = []
            for trace in trace_back:
                stack_trace.append("File : {}, Line : {}, Func.Name : {}, Message : {}".format(
                    trace[0], trace[1], trace[2], trace[3]))

            response["stack_trace"] = stack_trace

        return response

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        try:
            request.preprocess()

            request.max_new_tokens = get_num_tokens_to_generate(
                request.max_new_tokens, self.allowed_max_new_tokens)

            response, total_time_taken = run_and_log_time(
                partial(self.model.generate, request=request)
            )

            response.query_id = self.query_ids.generate_query_id
            self.query_ids.generate_query_id += 1
            response.total_time_taken = "{:.2f} secs".format(total_time_taken)

            return response
        except Exception:
            response = self.get_exception_response(
                self.query_ids.generate_query_id, request.method)
            self.query_ids.generate_query_id += 1
            raise HTTPException(500, response)

    def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        try:
            response, total_time_taken = run_and_log_time(
                partial(self.model.tokenize, request=request)
            )

            response.query_id = self.query_ids.tokenize_query_id
            self.query_ids.tokenize_query_id += 1
            response.total_time_taken = "{:.2f} msecs".format(
                total_time_taken * 1000)

            return response
        except Exception:
            response = self.get_exception_response(
                self.query_ids.tokenize_query_id, request.method)
            self.query_ids.tokenize_query_id += 1
            raise HTTPException(500, response)

    def query_id(self) -> QueryID:
        return self.query_ids

    def run(self):
        run(
            self.app,
            host=self.host,
            port=self.port,
            workers=self.workers
        )

    def shutdown(self):
        self.model.shutdown()


def main() -> None:
    args = get_args()
    server = Server(args)
    try:
        server.run()
    except KeyboardInterrupt:
        server.shutdown()


if (__name__ == "__main__"):
    main()
