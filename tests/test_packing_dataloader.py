import os
import torch.distributed as dist

from megatron.initialize import initialize_megatron
# from megatron.data.data_samplers import MegatronPackedRandomSampler
# from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.data.non_causal_mtf_dataset import build_train_valid_test_datasets

from datasets import load_dataset

import torch.distributed as dist

# To preprocess data before testing
# python tools/preprocess_data.py \
#     --input tests/data/t0/ag_news_classify_question_first.json \
#     --output-prefix tests/data/t0/ag_news_prompt \
#     --dataset-impl mmap \
#     --tokenizer-type PretrainedFromHF \
#     --tokenizer-name-or-path gpt2 \
#     --append-eod \
#     --workers 8

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# initialize the process group
dist.init_process_group("nccl", rank=0, world_size=1)

#Initialize Megatron with dummy variables
initialize_megatron(
    extra_args_provider=None,
    args_defaults={
        "micro_batch_size": 4,
        "num_layers": 4,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "seq_length": 256,
        "max_position_embeddings": 256,
        "distributed_backend": "nccl",
        "tokenizer_type": "PretrainedFromHF",
        "tokenizer_name_or_path": "gpt2",
        }
    )

train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
    data_prefix=["tests/data/t0/ag_news_prompt_text_document"],
    data_impl="mmap",
    splits_string="90,5,5",
    train_valid_test_num_samples=[100,100,100],
    seq_length=1024,
    seed=124,
    skip_warmup=True
    )

print("Test show dataset")
for idx in range(0,4):
    line = train_ds[idx]
    print(len(line))
    print(line)


# dl = torch.utils.data.DataLoader(
#     train_ds,
#     batch_size=4,
#     # batch_sampler=batch_sampler,
#     num_workers=4,
#     pin_memory=True
#     )
