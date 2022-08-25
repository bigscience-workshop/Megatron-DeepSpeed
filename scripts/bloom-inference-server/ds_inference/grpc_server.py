import argparse
import json
import os

import torch
from transformers import AutoTokenizer

import mii
from utils import GenerateRequest, GenerateResponse, Model, get_filter_dict, get_str_dtype, print_rank_n


class DSInferenceGRPCServer(Model):
    def __init__(self, args: argparse.Namespace) -> None:
        self.deployment_name = "ds_inference_grpc_server"

        files = os.listdir(args.save_mp_checkpoint_path)
        for file in files:
            if (file.endswith(".json")):
                checkpoints_json = json.load(
                    open(os.path.join(args.save_mp_checkpoint_path, file), "r"))
                break

        if ("base_dir" in checkpoints_json):
            del checkpoints_json["base_dir"]

        if (args.dtype == torch.float16):
            mii.deploy(
                task="text-generation",
                model=args.model_name,
                deployment_name=self.deployment_name,
                mii_config={
                    "dtype": get_str_dtype(args.dtype),
                    "tensor_parallel": 8,
                    "port_number": 50950,
                    "checkpoint_dict": checkpoints_json
                },
                model_path=args.save_mp_checkpoint_path
            )
        else:
            raise NotImplementedError("This is not yet supported")

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.pad = self.tokenizer.pad_token_id
        self.model = mii.mii_query_handle(self.deployment_name)

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        text = request.text

        return_type = type(text)
        if (return_type == str):
            text = [text]

        output_text = self.model.query(
            {"query": text},
            **get_filter_dict(request)
        ).response

        output_text = [_ for _ in output_text]

        # Remove input from output
        input_tokens = self.tokenizer(text).input_ids
        output_tokens = self.tokenizer(output_text).input_ids

        input_token_lengths = [len(x) for x in input_tokens]
        output_token_lengths = [len(x) for x in output_tokens]
        num_generated_tokens = [
            o - i for i, o in zip(input_token_lengths, output_token_lengths)]

        if (request.remove_input_from_output):
            output_tokens = [x[-i:]
                             for x, i in zip(output_tokens, num_generated_tokens)]
            output_text = self.tokenizer.batch_decode(
                output_tokens, skip_special_tokens=True)

        if (return_type == str):
            output_text = output_text[0]
            num_generated_tokens = num_generated_tokens[0]

        return GenerateResponse(
            text=output_text,
            num_generated_tokens=num_generated_tokens
        )

    def shutdown(self) -> None:
        print_rank_n("shutting down")
        # MII is buggy and sometimes spits out an error in terminate
        try:
            mii.terminate(self.deployment_name)
        except Exception:
            exit()
