"""
A script which prints the data according to the given sample index.
Below is an example bash script to print the data in sample index range 29040~29050:
```
source $six_ALL_CCFRWORK/code/tr1-13B/bigscience/train/tr1-13B-base/start-tr1-13B
MEGATRON_DEEPSPEED_REPO=$six_ALL_CCFRWORK/code/tr1-13B/Megatron-DeepSpeed-tr1-13B/

cd $MEGATRON_DEEPSPEED_REPO

VOCAB_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-vocab.json
MERGE_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/oscar-en/meg-gpt2_text_document

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024

NLAYERS=40
NHIDDEN=5120
NHEADS=32
SEQ_LEN=2048
VOCAB_SIZE=50257

python ./tools/sample_idxs_to_text.py \
    --seed 42 \
    --sample-id-range 29040 29050 \
    --print_text \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples 300_000_000 \
    --eval-interval 1000 \
    --eval-iters 5 \
    \
    `# Dummy params` \
    --num-layers 1 \
    --hidden-size 1 \
    --num-attention-heads 1
```
"""
from itertools import chain

import torch

from megatron import get_args
from megatron import get_tokenizer
from megatron import initialize_megatron
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.training import update_train_iters


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
