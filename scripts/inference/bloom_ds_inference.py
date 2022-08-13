import io
import json
import os
import shutil
from argparse import Namespace
from typing import List, Union

import deepspeed
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import get_checkpoint_shard_files
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, cached_path, hf_bucket_url, is_offline_mode
from transformers.utils.hub import EntryNotFoundError

import utils
from utils import Model, benchmark_end_to_end, get_argument_parser, print_rank_n, run_rank_n


class DSInferenceModel(Model):
    def __init__(self, args: Namespace) -> None:
        print_rank_n("Loading model...")

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        use_hf_checkpoints = not args.save_mp_checkpoint_path
        cache_ds_checkpoints = args.save_mp_checkpoint_path and not os.path.isdir(
            args.save_mp_checkpoint_path)

        world_size = int(os.getenv("WORLD_SIZE", "1"))

        # Load model
        with deepspeed.OnDevice(dtype=args.dtype, device="meta"):
            self.model = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(args.model_name),
                torch_dtype=torch.bfloat16
            )
        self.model = self.model.eval()

        # Write checkpoints.json
        tmp_directory = "tmp"
        run_rank_n(
            os.makedirs,
            {
                "name": tmp_directory,
                "exist_ok": True
            }
        )
        checkpoints_json = os.path.join(tmp_directory, "checkpoints.json")
        run_rank_n(
            write_checkponts_json,
            {
                "checkpoints_json": checkpoints_json,
                "model_name": args.model_name
            },
            barrier=True
        )

        # init inference
        if (use_hf_checkpoints):
            print_rank_n("Loading HF checkpoints")

            if (args.dtype == torch.float16):
                self.model = deepspeed.init_inference(
                    self.model,
                    mp_size=world_size,
                    dtype=args.dtype,
                    checkpoint=checkpoints_json,
                    replace_with_kernel_inject=True
                )
            else:
                raise NotImplementedError("bfloat16 is not yet supported")
        elif (cache_ds_checkpoints):
            print_rank_n("Caching DS checkpoints and loading model")

            run_rank_n(
                os.makedirs,
                {
                    "name": args.save_mp_checkpoint_path,
                    "exist_ok": True
                },
                barrier=True
            )

            if (args.dtype == torch.float16):
                self.model = deepspeed.init_inference(
                    self.model,
                    mp_size=world_size,
                    dtype=args.dtype,
                    checkpoint=checkpoints_json,
                    replace_with_kernel_inject=True,
                    save_mp_checkpoint_path=args.save_mp_checkpoint_path
                )
            else:
                raise NotImplementedError("bfloat16 is not yet supported")
        else:
            print_rank_n("Loading DS cached checkpoints")

            checkpoints_json = os.path.join(
                args.save_mp_checkpoint_path, "BLOOM-176B_ds-inference_config.json")

            if (args.dtype == torch.float16):
                self.model = deepspeed.init_inference(
                    self.model,
                    mp_size=world_size,
                    dtype=args.dtype,
                    checkpoint=checkpoints_json,
                    replace_with_kernel_inject=True
                )
            else:
                raise NotImplementedError("bfloat16 is not yet supported")

        run_rank_n(shutil.rmtree, {"path": tmp_directory})

        self.model = self.model.module
        self.input_device = torch.cuda.current_device()

        print_rank_n("Model loaded")

    def generate(self,
                 text: Union[str, List[str]],
                 generate_kwargs: dict,
                 remove_input_from_output: bool = False) -> Union[str, List[str]]:
        if (type(text) == str):
            text = [text]

        input_tokens = self.tokenizer(text, return_tensors="pt", padding=True)

        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.input_device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **input_tokens,
                **generate_kwargs
            )

        input_token_lengths = [x.shape[0] for x in input_tokens.input_ids]
        output_token_lengths = [x.shape[0] for x in output_tokens]
        generated_tokens = [
            o - i for i, o in zip(input_token_lengths, output_token_lengths)]

        if (remove_input_from_output):
            output_tokens = [x[-i:]
                             for x, i in zip(output_tokens, generated_tokens)]

        output_text = self.tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True)

        return output_text, generated_tokens


def get_args():
    parser = get_argument_parser()

    group = parser.add_argument_group(title="launch config")
    group.add_argument("--benchmark_cycles", type=int,
                       default=0, help="additionally run benchmark")
    group.add_argument("--local_rank", required=False,
                       type=int, help="used by dist launchers")
    group.add_argument("--save_mp_checkpoint_path", required=False,
                       type=str, help="MP checkpoints path")

    args = utils.get_args(parser)

    return args


def get_checkpoint_files(pretrained_model_name_or_path):
    # XXX: I just hacked this one together to automatically handle the fetching of the model file or
    # shards into cache and returning the cached entries - note that I removed most arguments
    cache_dir = None
    is_sharded = False

    # XXX: preparation for revision branches if needed
    revision = None
    #revision = "sharded"

    # this supports nodes with no network (so you need to pre-cache the model and the tokenizer with
    # python -c "from transformers import AutoModel; AutoModel.from_pretrained('bigscience/bloom')"
    if (is_offline_mode()):
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

    if (is_sharded):
        # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            revision=revision
        )

        return resolved_archive_file


def write_checkponts_json(checkpoints_json: str, model_name: str) -> None:
    with io.open(checkpoints_json, 'w', encoding='utf-8') as f:
        #checkpoint_dir = "/gpfsscratch/rech/six/commun/uan68tv-model-conversion/bloom"
        #checkpoint_files = glob.glob(f"{checkpoint_dir}/*bin")
        checkpoint_files = get_checkpoint_files(model_name)
        data = {
            "type": "BLOOM-176B",
            "checkpoints": checkpoint_files,
            "version": 1.0
        }
        json.dump(data, f)


if (__name__ == "__main__"):
    deepspeed.init_distributed('nccl')
    benchmark_end_to_end(get_args(), DSInferenceModel)