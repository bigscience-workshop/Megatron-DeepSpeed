import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import mpu
from megatron.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name
from megatron.data.bert_dataset import get_indexed_dataset_
from megatron.data.ict_dataset import InverseClozeDataset
from megatron.data.samplers import DistributedBatchSampler
from megatron.initialize import initialize_megatron
from megatron.training import get_model
from pretrain_bert_ict import get_batch, model_provider


def main():
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    args = get_args()
    model = load_checkpoint()
    model.eval()
    dataset = get_dataset()
    data_iter = iter(get_dataloader(dataset))

    all_input_tokens = []
    all_input_logits = []
    all_doc_tokens = []
    all_doc_logits = []

    for i in range(100):
        input_tokens, input_types, input_pad_mask, doc_tokens, doc_token_types, doc_pad_mask = get_batch(data_iter)
        input_logits, doc_logits, _ = model.module.module.forward(
            input_tokens, input_types, input_pad_mask, doc_tokens, doc_pad_mask, doc_token_types, return_logits=True)

        all_input_tokens.append(input_tokens.detach().cpu().numpy())
        all_input_logits.append(input_logits.detach().cpu().numpy())
        all_doc_tokens.append(doc_tokens.detach().cpu().numpy())
        all_doc_logits.append(doc_logits.detach().cpu().numpy())

    all_inputs_tokens = np.array(all_input_tokens).reshape(-1, args.seq_length)
    all_inputs_logits = np.array(all_input_logits).reshape(-1, 128)
    all_doc_tokens = np.array(all_doc_tokens).reshape(-1, args.seq_length)
    all_doc_logits = np.array(all_doc_logits).reshape(-1, 128)
    np.save('input_tokens.npy', all_input_tokens)
    np.save('input_logits.npy', all_input_logits)
    np.save('doc_tokens.npy', all_doc_tokens)
    np.save('doc_logits.npy', all_doc_logits)


def load_checkpoint():
    args = get_args()
    model = get_model(model_provider)

    if isinstance(model, torchDDP):
        model = model.module
    tracker_filename = get_checkpoint_tracker_filename(args.load)
    with open(tracker_filename, 'r') as f:
        iteration = int(f.read().strip())

    assert iteration > 0
    checkpoint_name = get_checkpoint_name(args.load, iteration, False)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    state_dict = torch.load(checkpoint_name, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    torch.distributed.barrier()

    if mpu.get_data_parallel_rank() == 0:
        print(' successfully loaded {}'.format(checkpoint_name))

    return model


def get_dataset():
    args = get_args()
    indexed_dataset = get_indexed_dataset_(args.data_path, 'mmap', True)

    doc_idx_ptr = indexed_dataset.get_doc_idx()
    total_num_documents = indexed_dataset.doc_idx.shape[0] - 1
    indexed_dataset.set_doc_idx(doc_idx_ptr[0:total_num_documents])
    kwargs = dict(
        name='full',
        indexed_dataset=indexed_dataset,
        data_prefix=args.data_path,
        num_epochs=None,
        max_num_samples=total_num_documents,
        max_seq_length=288,  # doesn't matter
        short_seq_prob=0.0001,  # doesn't matter
        seed=1
    )
    dataset = InverseClozeDataset(**kwargs)
    return dataset


def get_dataloader(dataset):
    args = get_args()

    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)

    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)


if __name__ == "__main__":
    main()
