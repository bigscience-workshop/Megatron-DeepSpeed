import argparse

import utils
from ds_inference import cache_ds_checkpoints
from utils import get_argument_parser


def get_args() -> argparse.Namespace:
    parser = get_argument_parser()

    group = parser.add_argument_group(title="launch config")
    group.add_argument("--local_rank", required=False,
                       type=int, help="used by dist launchers")
    group.add_argument("--save_mp_checkpoint_path", required=True,
                       type=str, help="MP checkpoints path for DS inference")

    args = utils.get_args(parser)

    return args


def main() -> None:
    cache_ds_checkpoints(get_args())


if (__name__ == "__main__"):
    main()
