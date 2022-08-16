import argparse
import json
import os

import torch
from transformers import AutoTokenizer

import mii
from utils import GenerateRequest, GenerateResponse, Model, get_str_dtype


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
        self.model = mii.mii_query_handle(self.deployment_name)

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        text = request.text

        return_type = type(text)
        if (return_type == str):
            text = [text]

        output_text = self.model.query(
            {"query": text},
            min_length=request.min_length,
            do_sample=request.do_sample,
            early_stopping=request.early_stopping,
            num_beams=request.num_beams,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            typical_p=request.typical_p,
            repitition_penalty=request.repitition_penalty,
            bos_token_id=request.bos_token_id,
            pad_token_id=request.pad_token_id,
            eos_token_id=request.eos_token_id,
            length_penalty=request.length_penalty,
            no_repeat_ngram_size=request.no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=request.encoder_no_repeat_ngram_size,
            num_return_sequences=request.num_return_sequences,
            max_time=request.max_time,
            max_new_tokens=request.max_new_tokens,
            decoder_start_token_id=request.decoder_start_token_id,
            num_beam_groups=request.num_beam_groups,
            diversity_penalty=request.diversity_penalty,
            forced_bos_token_id=request.forced_bos_token_id,
            forced_eos_token_id=request.forced_eos_token_id,
            exponential_decay_length_penalty=request.exponential_decay_length_penalty,
            bad_words_ids=request.bad_words_ids,
            force_words_ids=request.force_words_ids
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
        mii.terminate(self.deployment_name)
        exit()
