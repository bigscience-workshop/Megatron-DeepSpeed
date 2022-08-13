import deepspeed

import utils
import values
from ds_inference import DSInferenceModel
from ds_zero import DSZeROModel
from hf_accelerate import HFAccelerateModel
from utils import benchmark_end_to_end, get_argument_parser


def get_args():
    parser = get_argument_parser()

    group = parser.add_argument_group(title="launch config")
    group.add_argument(
        "--deployment_framework",
        type=str,
        choices=[
            values.HF_ACCELERATE,
            values.DS_INFERENCE,
            values.DS_ZERO
        ],
        default=values.HF_ACCELERATE
    )
    group.add_argument("--benchmark_cycles", type=int,
                       default=0, help="additionally run benchmark")
    group.add_argument("--local_rank", required=False,
                       type=int, help="used by dist launchers")
    group.add_argument("--save_mp_checkpoint_path", required=False,
                       type=str, help="MP checkpoints path for DS inference")
    group.add_argument("--cpu_offload", action="store_true",
                       help="whether to activate CPU offload for DS ZeRO")

    args = utils.get_args(parser)

    launched_with_deepspeed = args.deployment_framework in [
        values.DS_INFERENCE, values.DS_ZERO]

    if (not launched_with_deepspeed):
        assert args.local_rank == None, "local_rank must be None if not launched with DeepSpeed"

    if (args.save_mp_checkpoint_path):
        assert args.deployment_framework == values.DS_INFERENCE, "save_mp_checkpoint_path only works with DS inference"

    if (args.cpu_offload):
        assert args.deployment_framework == values.DS_ZERO, "cpu_offload only works with DS_ZeRO"

    return args


if (__name__ == "__main__"):
    args = get_args()

    if (args.deployment_framework == values.HF_ACCELERATE):
        benchmark_end_to_end(get_args(), HFAccelerateModel)
    elif (args.deployment_framework == values.DS_INFERENCE):
        deepspeed.init_distributed('nccl')
        benchmark_end_to_end(get_args(), DSInferenceModel)
    else:
        benchmark_end_to_end(get_args(), DSZeROModel, zero_activated=True)
