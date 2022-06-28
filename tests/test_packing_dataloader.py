from functools import partial
import sys
from pathlib import Path
import os

import torch

# Insert megatron's root dir into sys.path
root_repo_path = str(Path(__file__).resolve().parents[1])
if root_repo_path not in sys.path:
    sys.path.insert(0, root_repo_path)

from megatron.initialize import initialize_megatron
from megatron.data.data_samplers import MegatronPackedRandomSampler, pack_samples
from megatron.data.non_causal_mtf_dataset import build_train_valid_test_datasets
from megatron.utils import get_packed_attention_mask

"""
To preprocess data before testing

TOKENIZER_PATH="gpt2"
DATA_PATH="tests/data/t0/ag_news_classify_question_first.json"
OUTPUT="tests/data/t0/ag_news_prompt"

python tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key inputs \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --append-eod \
    --workers 8

python tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key targets \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --append-eod \
    --workers 8
"""


"""
Define Environment variables if necessary
"""
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "jean-zay-pp2" # $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
os.environ["MASTER_PORT"] = "6002"
os.environ["LOCAL_RANK"] = "0"


seq_length = 256



# Initialize Megatron with dummy variables
initialize_megatron(
    extra_args_provider=None,
    allow_no_cuda=True,
    args_defaults={
        "micro_batch_size": 4,
        "num_layers": 4,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "seq_length": seq_length,
        "max_position_embeddings": seq_length,
        "distributed_backend": "nccl",
        "tokenizer_type": "PretrainedFromHF",
        "tokenizer_name_or_path": "gpt2",
    }
)

train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
    data_prefix=[{
        "input_tokens": "tests/data/t0/ag_news_prompt_inputs_document",
        "target_tokens": "tests/data/t0/ag_news_prompt_targets_document"
    }],
    data_impl="mmap",
    splits_string="90,5,5",
    train_valid_test_num_samples=[50,0,0],
    seq_length=seq_length,
    seed=124,
    skip_warmup=True
)

print("Test show dataset")
for idx in range(0,4):
    line = train_ds[idx]
    print(len(line))
    print(line)


batch_sampler = MegatronPackedRandomSampler(
    sequence_length=seq_length,
    dataset=train_ds,
    total_samples=len(train_ds),
    consumed_samples=0,
    micro_batch_size=4,
    data_parallel_rank=0,
    data_parallel_size=1
)

dl = torch.utils.data.DataLoader(
    train_ds,
    batch_sampler=batch_sampler,
    num_workers=4,
    pin_memory=True,
    collate_fn=partial(pack_samples, max_seq_len=256),
)

for i, items in enumerate(dl):

    micro_batch_size, seq_length = items['decoder_target_tokens'].shape
    causal_mask = torch.tril(
        torch.ones(
            (micro_batch_size, seq_length, seq_length))
    ).view(
        micro_batch_size, 1, seq_length, seq_length
    )

    mask = get_packed_attention_mask(
        causal_mask=causal_mask,
        tokens=torch.tensor(items['decoder_target_tokens']),
        decoder_causal_attention=torch.tensor(items['decoder_causal_attention']),
        segment_ids=torch.tensor(items['decoder_segment_ids']),
    )

    assert mask.shape == (micro_batch_size, 1, seq_length, seq_length)

