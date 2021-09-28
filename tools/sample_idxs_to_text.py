from megatron import get_args
from megatron import initialize_megatron
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.training import update_train_iters
import numpy as np

def _add_network_size_args(parser):
    group = parser.add_argument_group(title='Get text from sample idxs.')
    group.add_argument('--sample-id-range', type=int, nargs='+', required=True,
                        help='The number of samples consumed. ex) --sample-id-range 1024 2048')

    return parser


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=_add_network_size_args)
    args = get_args()
    update_train_iters(args)
    # prepare data iterators
    eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [args.train_samples,
                                  eval_iters * args.global_batch_size,
                                  test_iters * args.global_batch_size]
    train_dataset, _, _ = build_train_valid_test_datasets(data_prefix=args.data_path,
                                                          data_impl=args.data_impl,
                                                          splits_string=args.split,
                                                          train_valid_test_num_samples=train_val_test_num_samples,
                                                          seq_length=args.seq_length,
                                                          seed=args.seed,
                                                          skip_warmup=(not args.mmap_warmup))


    np_orig_opts = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)

    for i in range(args.sample_id_range[0], args.sample_id_range[1]):
        print(f"[{i}/{len(train_dataset)}]-th sample: ")
        print(train_dataset[i]["text"])

    np.set_printoptions(**np_orig_opts)  # restore
