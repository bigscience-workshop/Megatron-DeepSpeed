import os
import torch.distributed as dist

from megatron.initialize import initialize_megatron
# from megatron.data.data_samplers import MegatronPackedRandomSampler
from megatron.data.gpt_dataset import build_train_valid_test_datasets, build_dataset_group

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
        }
    )

train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
    data_prefix=["tests/data/gpt2/meg-gpt2-openwebtext_text_document"],
    data_impl="mmap",
    splits_string="90,5,5",
    train_valid_test_num_samples=[100,100,100],
    seq_length=1024,
    seed=124,
    skip_warmup=True
    )

dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=4,
    # batch_sampler=batch_sampler,
    num_workers=4,
    pin_memory=True
    )
