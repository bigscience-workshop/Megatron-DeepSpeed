from megatron import get_args, get_tokenizer
from megatron import initialize_megatron
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.training import update_train_iters
import torch
from itertools import chain

def _add_network_size_args(parser):
    group = parser.add_argument_group(title='Get text from sample idxs.')
    group.add_argument('--sample-id-range', type=int, nargs='+', required=True,
                        help='The number of samples consumed. ex) --sample-id-range 1024 2048')
    group.add_argument('--all_tokens', action='store_true', help='Whether to dump all tokens per record')
    group.add_argument('--print_tokens', action='store_true', help='Whether to print tokens')
    group.add_argument('--print_text', action='store_true', help='Whether to print text')

    return parser


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=_add_network_size_args)

    args = get_args()
    tokenizer = get_tokenizer()
    update_train_iters(args)

    if not (args.print_tokens or args.print_text):
        raise ValueError("Need to specify either --print_tokens or --print_text or both")

    if args.all_tokens and not args.print_tokens:
        raise ValueError("--all_tokens requires --print_tokens")

    # prepare data iterators
    eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [args.train_samples,
                                  eval_iters * args.global_batch_size,
                                  test_iters * args.global_batch_size]
    train_ds, _, _ = build_train_valid_test_datasets(data_prefix=args.data_path,
                                                          data_impl=args.data_impl,
                                                          splits_string=args.split,
                                                          train_valid_test_num_samples=train_val_test_num_samples,
                                                          seq_length=args.seq_length,
                                                          seed=args.seed,
                                                          skip_warmup=(not args.mmap_warmup))

    # fast forward to where we want to start sampling
    train_dataloader = build_pretraining_data_loader(train_ds, args.sample_id_range[0])
    data_iterator = iter(train_dataloader)

    if args.all_tokens:
        torch.set_printoptions(threshold=2**20)


    for i in range(args.sample_id_range[0], args.sample_id_range[1]):

        tokens = next(data_iterator)["text"]

        if args.print_tokens:
            print(f"{i} {tokens}")

        if args.print_text:
            tokens = list(chain(*tokens.tolist()))
            trim_decode_tokens = tokenizer.detokenize(tokens)
            print(f"{i} {trim_decode_tokens}")
