import argparse
import json

import deepspeed

import constants
import utils
from ds_inference import DSInferenceGRPCServer
from hf_accelerate import HFAccelerateModel
from utils import get_argument_parser, parse_generate_kwargs, print_rank_n


def get_args() -> argparse.Namespace:
    parser = get_argument_parser()

    group = parser.add_argument_group(title="launch config")
    group.add_argument(
        "--deployment_framework",
        type=str,
        choices=[
            constants.HF_ACCELERATE,
            constants.DS_INFERENCE
        ],
        default=constants.HF_ACCELERATE
    )
    group.add_argument("--save_mp_checkpoint_path", required=False,
                       type=str, help="MP checkpoints path for DS inference")
    group.add_argument("--shutdown_command", required=False,
                       type=str, default="__shutdown__", help="This string will exit the script")

    args = utils.get_args(parser)

    if (args.save_mp_checkpoint_path):
        assert args.deployment_framework == constants.DS_INFERENCE, "save_mp_checkpoint_path only works with DS inference"

    return args


def main() -> None:
    args = get_args()

    if (args.deployment_framework == constants.HF_ACCELERATE):
        model = HFAccelerateModel(args)
    elif (args.deployment_framework == constants.DS_INFERENCE):
        model = DSInferenceGRPCServer(args)
    else:
        raise ValueError(
            f"Unknown deployment framework {args.deployment_framework}")

    generate_kwargs = args.generate_kwargs

    while (True):
        # currently only 1 process is running so its
        # fine but might need to run_rank_n for this
        # if running a deployment_framework with
        # multiple processes
        input_text = input("Input text: ")

        if (input_text == args.shutdown_command):
            model.shutdown()

        if (input("change generate_kwargs? [y/n] ") == "y"):
            generate_kwargs = input("Generate kwargs: ")
            generate_kwargs = json.loads(generate_kwargs)
            generate_kwargs = parse_generate_kwargs(generate_kwargs)
        print_rank_n("generate_kwargs:", generate_kwargs)

        output_text, num_generated_tokens = model.generate(
            input_text,
            generate_kwargs,
            remove_input_from_output=True
        )

        print_rank_n("Output text:", output_text)
        print_rank_n("Generated tokens:", num_generated_tokens)


if (__name__ == "__main__"):
    main()
