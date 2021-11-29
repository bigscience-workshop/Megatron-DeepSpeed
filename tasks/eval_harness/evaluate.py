
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))

# Downloads the taks in the evaluation harness
# This is particularly useful when running in environments where the GPU nodes 
# do not have internet access. This way we can pre-download them and use the cached data-set during evaluation.

from lm_eval.models.gpt2 import GPT2LM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from tqdm import tqdm
import torch.nn.functional as F 

from lm_eval.tasks import ALL_TASKS
from pretrain_gpt import model_provider

import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.training import setup_model_and_optimizer
from megatron.mpu.mappings import gather_from_tensor_model_parallel_region

from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
import pickle
import json

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.module import Float16Module

class EvalHarnessAdaptor(GPT2LM):
    def __init__(self, model, tokenizer):
        args = get_args()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.encode = self.tokenizer.tokenize
        self.VOCAB_SIZE = tokenizer.vocab_size
        self.EOT_TOKEN_ID = tokenizer.eod

        self.max_length = args.max_position_embeddings
        self.max_gen_toks = 128
        self.batch_size = args.micro_batch_size
        self.cache_hook = CacheHook(None)
        self.is_main = args.rank == 0
        self.is_local_main = args.local_rank == 0
        self.device = torch.cuda.current_device()
        self.is_model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.is_pipe_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        self.is_data_parallel = mpu.get_data_parallel_world_size() > 1
        
        if self.is_data_parallel:
            raise NotImplementedError("Data parallelism is currently not supported for evaluation")

        self.is_last_stage = True if not self.is_pipe_parallel else mpu.is_pipeline_last_stage()  # only the last stage of the pipeline model will receive the logits

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):  
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        self.model.eval()
        with torch.no_grad():
            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps, contlens, inplens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen

                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))
                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))

                res_len += len(chunk)
                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1)  # [batch, seq, vocab]
                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens,
                                                                                 contlens):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0).to(multi_logits.device)
                        max_equal = (greedy_tokens == cont_toks).all()
                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))

                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                        res.append(answer)

        # broadcast results to all ranks
        if self.is_pipe_parallel:
            src_rank = mpu.get_pipeline_model_parallel_last_rank()

            if res:
                logits_sums, max_equals = list(zip(*res))
                logits_sums = torch.FloatTensor(logits_sums).cuda()
                max_equals = torch.LongTensor(max_equals).cuda()
            else:
                logits_sums = torch.zeros(res_len, dtype=torch.float32).cuda()
                max_equals = torch.zeros(res_len, dtype=torch.int64).cuda()
            torch.distributed.broadcast(tensor=logits_sums, src=src_rank)
            torch.distributed.broadcast(tensor=max_equals, src=src_rank)
            max_equals = [bool(i) for i in max_equals.tolist()]
            logits_sums = logits_sums.tolist()
            res = list(zip(logits_sums, max_equals))

        return reord.get_original(res)


    def _model_call(self, inps):
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            inps,
            self.tokenizer.eod,
            self.args.reset_position_ids,
            self.args.reset_attention_mask,
            self.args.eod_mask_loss,
            prefix_indices=None,
            loss_on_targets_only=self.args.loss_on_targets_only
        )
        
        # Since the shape of the micro-batch will change
        # We need set the correct shapes here 
        # So that latter pipeline stages knows which shapes to expect.
        # Otherwise we will deadlock. 
        args = get_args()
        args.micro_batch_size = len(inps)
        args.seq_length = len(inps[0])

        input_tensor = recv_forward()

        # Forward pass through the model.
        unwrapped_model = unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.set_input_tensor(input_tensor)
        output = self.model(inps, position_ids, attention_mask)
        
        send_forward(output)
        if mpu.is_pipeline_last_stage():
            return gather_from_tensor_model_parallel_region(output)
        else:
            return None

from megatron.initialize import initialize_megatron

def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='eval harness')
    group.add_argument('--task_list', type=str, default = "all", help='Either "all" or comma separated list of tasks.')
    group.add_argument('--task_load_path', type=str, default = "./task_cache.pickle", help='Path to where the downloaded tasks are stored, or None if download is possible.')
    group.add_argument('--results_path', type=str, default = "./results.json", help='Path to where the results will be stored.')
    return parser

def main():
    
    initialize_megatron(extra_args_provider=get_tasks_args)

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    #load eval harness task dict
    if args.task_load_path != 'None':
        with open(args.task_load_path, 'rb') as file:
            task_dict = pickle.load(file)
        
        if args.task_list != 'all':
            task_list = args.task_list.split(',')
            task_dict = dict((k,task_dict[k]) for k in task_list)
            
    else:
        task_list = ALL_TASKS if args.task_list == 'all' else args.task_list.split(',')
        task_dict = tasks.get_task_dict(task_list)

    # Set up model and load checkpoint.
    model, _, _  = setup_model_and_optimizer(model_provider)
    
    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    tokenizer = get_tokenizer()
    adaptor = EvalHarnessAdaptor(model, tokenizer) 
    results = evaluator.evaluate(adaptor, task_dict, False, 0, None)
    
    print_rank_0(json.dumps(results, indent=2))
    if args.rank==0:
        with open(args.results_path, 'w') as outfile:
            json.dump(results, outfile, indent = 4)


if __name__ == '__main__':
    main()