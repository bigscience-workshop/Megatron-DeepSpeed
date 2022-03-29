import os
import unittest
from random import randint
from unittest.mock import patch

import deepspeed
import torch
import logging
import numpy as np

import pytest
from parameterized import parameterized

from megatron import initialize_megatron, get_args, get_tokenizer, global_vars
from megatron.testing_utils import TestCasePlus, mockenv_context, flatten_arguments, require_deepspeed, \
    require_torch_multi_gpu, torch_assert_equal, CaptureStdout, execute_subprocess_async
from megatron.training import setup_model_and_optimizer
from megatron.mpu.mappings import gather_from_tensor_model_parallel_region
from pretrain_gpt import model_provider as gpt_model_provider, get_batch_pipe as get_gpt_batch_pipe
from pretrain_prefix_lm import model_provider as prefix_lm_model_provider, get_batch_pipe as get_prefix_lm_batch_pipe
import multiprocessing as mp
from multiprocessing import Pool
from megatron.checkpointing import save_checkpoint

from megatron.utils import get_ltor_masks_and_position_ids

@require_deepspeed
@require_torch_multi_gpu
class MegDSTestTP(TestCasePlus):
    def get_default_args(self):
        """return a dictionary with key as argument name and value as additional arguments"""
        data_dir = f"{self.data_dir}/gpt2"
        return {
            # GPT_ARGS
            "--num-layers": "2",
            "--hidden-size": "128",
            "--num-attention-heads": "4",
            "--seq-length": "256",
            "--max-position-embeddings": "256",
            "--micro-batch-size": "4",
            "--global-batch-size": "8",
            "--lr-decay-iters": "320000",
            "--lr-decay-style": "cosine",
            "--lr": "0.00015",
            "--min-lr": "1.0e-5",
            "--train-iters": "5000",
            "--tokenizer-type": "GPT2BPETokenizer",
            "--merge-file": f"{data_dir}/gpt2-tiny-merges.txt",
            "--vocab-file": f"{data_dir}/gpt2-tiny-vocab.json",
            "--data-impl": "mmap",
            "--distributed-backend": "nccl",
            "--weight-decay": "1e-2",
            "--clip-grad": "1.0",
            "--lr-warmup-fraction": ".01",
            "--fp16": "",

            "--attention-dropout": "0",
            "--hidden-dropout": "0",
            

            # OUTPUT_ARGS
            "--log-interval": "10",
            "--save-interval": "10",
            "--eval-interval": "10",
            "--eval-iters": "5",
            "--checkpoint-activations": "",
            
            #ds args
            "--deepspeed": "",
            "--deepspeed_config": f"{self.test_file_dir_str}/ds_config.json",
            "--zero-stage": "1",
            "--deepspeed-activation-checkpointing": ""
            # DATA_ARGS
        }
        
    def setUp(self) -> None:
        super().setUp()

        # We reset all global variables
        global_vars._GLOBAL_ARGS = None
        global_vars._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
        global_vars._GLOBAL_TOKENIZER = None
        global_vars._GLOBAL_TENSORBOARD_WRITER = None
        global_vars._GLOBAL_ADLR_AUTORESUME = None
        global_vars._GLOBAL_TIMERS = None

    def infer_model(args):
        tp_index, tp_size, command_args, token_ids, save, load = args
        dist_env = dict(
            MASTER_ADDR="localhost", MASTER_PORT="9991", RANK=str(tp_index), LOCAL_RANK=str(tp_index), WORLD_SIZE=str(tp_size)
        )
        logging.getLogger().critical("Process: starting")
        
        #Hack
        import megatron.initialize as init
        init.git_ds_info = lambda: None

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**dist_env):
                
                def create_model_inputs(tokens):
                    args = get_args()

                    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                        tokens,
                        tokenizer.eod,
                        args.reset_position_ids,
                        args.reset_attention_mask,
                        args.eod_mask_loss,
                        prefix_indices=None,
                        loss_on_targets_only=False)

                    return (tokens, position_ids, attention_mask), (tokens, loss_mask)

                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()

                tokenizer = get_tokenizer()

                if load is not None:
                    # Hack (same as in eval_harness/evaluate.py)
                    # Loading pipelined models in deepspeed with different TP than it was originally trained on fails
                    # due to a sanity check, that makes sure that all state_dicts that we merge contains attention layers.
                    # This, however, is not true for pipelining when we will merge the state_dict for the embeddings which
                    # which does not contain these attention-specific keys.
                    #
                    # Deepspeed does however manage to load the model if we just turn off this sanity check.
                    deepspeed.runtime.state_dict_factory.MegatronSDLoader.sanity_check = lambda self, ckpt_file_name: None

                model, _, _ = setup_model_and_optimizer(gpt_model_provider)
                model = model[0]

                if load is not None:
                    zero_enabled = model._config.zero_enabled
                    model._config.zero_enabled = False
                    _, _ = model.load_checkpoint(load, load_optimizer_states=False, load_lr_scheduler_states=False, load_module_only=True)
                    model._config.zero_enabled = zero_enabled
                
                if token_ids is None:
                    token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))

                    # eod is a special token
                    token_ids[token_ids == tokenizer.eod] += 1
                    token_ids[token_ids == tokenizer.eod] %= args.padded_vocab_size
                else:
                    token_ids = torch.tensor(token_ids)
                
                model.micro_batches = 1
                model.set_batch_fn(create_model_inputs)
                # process batch
                input_batch = get_gpt_batch_pipe({"text": token_ids})[0]

                # get a modified version of the first batch, we change a specific index
                changed_index = randint(0, args.seq_length - 2)
                input_token_ids_changed = input_batch[0].clone()
                # We increment the token_id by one for that index in order to artificially change the sequence.
                input_token_ids_changed[:, changed_index] = \
                    (input_token_ids_changed[:,changed_index] + 1) % args.padded_vocab_size

                output = model.eval_batch(iter([token_ids]), compute_loss = False, reduce_output = None)[0]
                
                output = gather_from_tensor_model_parallel_region(output)

                if save != None:
                    args.save = save
                    save_checkpoint(0, [model], None, None)
                
                return (output[0].detach().cpu().numpy(), token_ids.detach().cpu().numpy())

    def test_alibi_tp(self):
        mp.set_start_method('spawn', force=True)
        cp_dir = self.get_auto_remove_tmp_dir()
        
        command_args = self.get_default_args()
        command_args["--pad-vocab-size-to"] = "5120" # This is equal to 128 * 40 which is above the len of gp2-tiny vocabulary
        command_args["--position-embedding-type"] = "alibi"
        command_args["--tensor-model-parallel-size"] = "1"
        
        pool = Pool(1)
        result = pool.map(MegDSTestTP.infer_model, [((0, 1, command_args, None, cp_dir, None))])
        pool.close()
        pool.join()
        
        output, tokens = result[0]
        logging.getLogger().info("First done!")

        command_args["--tensor-model-parallel-size"] = "2"

        pool = Pool(2)
        result = pool.map(MegDSTestTP.infer_model, [((0, 2, command_args, tokens, None, cp_dir)), ((1, 2, command_args, tokens, None, cp_dir))])
        pool.close()
        pool.join()
        
        output2, tokens = result[0]

        logging.getLogger().critical(output-output2)
        self.assertTrue(np.allclose(output, output2, atol=5e-3, rtol=0), "Different results when running with TP=1 and TP=2")



    def test_embedding_matrix_tp(self):
        mp.set_start_method('spawn', force=True)
        cp_dir = self.get_auto_remove_tmp_dir()
        
        command_args = self.get_default_args()
        command_args["--pad-vocab-size-to"] = "5120" # This is equal to 128 * 40 which is above the len of gp2-tiny vocabulary
        command_args["--seq-length"] = "4"
        command_args["--micro-batch-size"] = "2"
        tokens = [[5119, 0, 1, 5100],[0, 1, 5111, 5101]]

        command_args["--tensor-model-parallel-size"] = "1"
        
        pool = Pool(1)
        # tp_index, tp_size, command_args, token_ids, save, load
        result = pool.map(MegDSTestTP.infer_model, [((0, 1, command_args, tokens, cp_dir, None))])
        pool.close()
        pool.join()
        
        output, _ = result[0]
        logging.getLogger().info("First done!")

        command_args["--tensor-model-parallel-size"] = "2"

        pool = Pool(2)
        result = pool.map(MegDSTestTP.infer_model, [((0, 2, command_args, tokens, None, cp_dir)), ((1, 2, command_args, tokens, None, cp_dir))])
        pool.close()
        pool.join()
        
        output2, _ = result[0]

        logging.getLogger().critical(output-output2)
        self.assertTrue(np.allclose(output,output2, atol=5e-3, rtol=0), "Different results when running with TP=1 and TP=2")


    def test_embedding_matrix_tp_with_invalid_tokens_ids(self):
        mp.set_start_method('spawn', force=True)
        
        command_args = self.get_default_args()
        command_args["--pad-vocab-size-to"] = "5120" # This is equal to 128 * 40 which is above the len of gp2-tiny vocabulary
        command_args["--seq-length"] = "4"
        command_args["--micro-batch-size"] = "2"
        tokens = [[5120, 0, 1, 2],[0, 1, 3, 4]]

        command_args["--tensor-model-parallel-size"] = "1"

        pool = Pool(1)
        with pytest.raises(Exception) as exc_info: 
            _ = pool.map(MegDSTestTP.infer_model, [((0, 1, command_args, tokens, None, None))])
        pool.close()
        pool.join()

        self.assertIn("There is an input id in the input that is greater than the highest possible input id" , str(exc_info.value))
        
        logging.getLogger().info("First done!")

        command_args["--tensor-model-parallel-size"] = "2"

        pool = Pool(2)
        with pytest.raises(Exception) as exc_info: 
            _ = pool.map(MegDSTestTP.infer_model, [((0, 2, command_args, tokens, None, None)), ((1, 2, command_args, tokens, None, None))])
        pool.close()
        pool.join()

        self.assertIn("There is an input id in the input that is greater than the highest possible input id", str(exc_info.value))


    def test_tokenizer_vocab_size_multiple_of_tp_size(self):
        mp.set_start_method('spawn', force=True)
        
        command_args = self.get_default_args()
        command_args["--pad-vocab-size-to"] = "5121" # This is equal to 128 * 40 + 1 which is above the len of gp2-tiny vocabulary
        command_args["--micro-batch-size"] = "4"
        command_args["--tensor-model-parallel-size"] = "2"
        command_args["--make-vocab-size-divisible-by"] = "1"

        pool = Pool(2)
        with pytest.raises(Exception) as exc_info: 
            _ = pool.map(MegDSTestTP.infer_model, [((0, 2, command_args, None, None, None)), ((1, 2, command_args, None, None, None))])
        pool.close()
        pool.join()

        self.assertEqual(str(exc_info.value), "5121 is not divisible by 2")

    def test_tokenizer_raise_error_make_vocab_size_divisible_by(self):
        mp.set_start_method('spawn', force=True)
        
        command_args = self.get_default_args()
        command_args["--pad-vocab-size-to"] = "5121" # This is equal to 128 * 40 + 1 which is above the len of gp2-tiny vocabulary
        command_args["--micro-batch-size"] = "4"
        

        pool = Pool(2)
        with pytest.raises(Exception) as exc_info: 
            _ = pool.map(MegDSTestTP.infer_model, [((0, 2, command_args, None, None, None)), ((1, 2, command_args, None, None, None))])
        pool.close()
        pool.join()

        self.assertEqual(str(exc_info.value), "5121 is not divisible by 128")

    @parameterized.expand(["bf16", "fp16"])
    def test_layer_norm_consistent(self, variation):
        src_dir = self.src_dir
        output_dir = self.get_auto_remove_tmp_dir()
        num_gpus = 2
        seq_len = 128
        data_dir = f"{self.data_dir}/gpt2"

        command_args = self.get_default_args()
        command_args["--pad-vocab-size-to"] = "5120"  # This is equal to 128 * 40 which is above the len of gp2-tiny vocabulary
        command_args["--position-embedding-type"] = "alibi"
        command_args["--embed-layernorm"] = ""
        command_args["--tensor-model-parallel-size"] = "2"
        command_args["--save"] = f"{output_dir}/checkpoints"
        command_args["--load"] = f"{output_dir}/checkpoints"
        command_args["--data-path"] = f"{data_dir}/meg-gpt2-openwebtext_text_document"
        command_args["--train-samples"] = "200"
        command_args["--rampup-batch-size"] = "4 4 200"
        command_args["--seq-length"] = "128"
        command_args["--exit-interval"] = "20"
        del command_args["--train-iters"]
        del command_args["--lr-decay-iters"]
        command_args["--tensorboard-dir"] = f"{output_dir}/tensorboard"
        command_args["--lr"] = "1e-1"

        if variation == "bf16":
            command_args["--bf16"] = ""
            del command_args["--fp16"]
            command_args["--deepspeed_config"] = f"{self.test_file_dir_str}/ds_config_bf16.json"
            command_args["--zero-stage"] = "0"
        elif variation == "fp16":
            command_args["--fp16"] = ""
            command_args["--deepspeed_config"] = f"{self.test_file_dir_str}/ds_config.json"
            command_args["--zero-stage"] = "1"

        # args, ds_args, num_gpus = self.get_variation_config("base", output_dir, n_samples=200)

        script = [f"{src_dir}/pretrain_gpt.py"]
        launcher = f"deepspeed --num_nodes 1 --num_gpus {num_gpus}".split()
        cmd = launcher + script + [elt for elts in [f"{key} {value}".split() for key, value in command_args.items()] for elt in elts]
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # 1. test that the layer norm weights and biases are synchronized
        checkpoints = ["global_step10", "global_step20"]

        # Check transformer layer norm
        keys_to_compare = [
            "input_layernorm.weight",
            "input_layernorm.bias",
            "post_attention_layernorm.weight",
            "post_attention_layernorm.bias",
            "self_attention.dense.bias",
            "mlp.dense_4h_to_h.bias"
        ]
        files_to_compare = [[f"layer_{layer_id:02d}-model_{tp:02d}-model_states.pt" for tp in range(num_gpus)] for
                            layer_id in [3, 4]]
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(output_dir, "checkpoints", checkpoint)
            for key in keys_to_compare:
                for files in files_to_compare:
                    weights = [torch.load(os.path.join(checkpoint_path, file))[key] for file in files]
                    ref = weights[0]
                    for weight in weights[1:]:
                        torch_assert_equal(ref, weight, check_device=False)

        # Check embed layer norm
        keys_to_compare = [
            "word_embeddings.norm.weight",
            "word_embeddings.norm.bias"
        ]
        files_to_compare = [[f"layer_{layer_id:02d}-model_{tp:02d}-model_states.pt" for tp in range(num_gpus)] for
                            layer_id in [1]]
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(output_dir, "checkpoints", checkpoint)
            for key in keys_to_compare:
                for files in files_to_compare:
                    weights = [torch.load(os.path.join(checkpoint_path, file))[key] for file in files]
                    ref = weights[0]
                    for weight in weights[1:]:
                        torch_assert_equal(ref, weight, check_device=False)

        # Final layer norm
        keys_to_compare = [
            "weight",
            "bias"
        ]
        files_to_compare = [[f"layer_{layer_id:02d}-model_{tp:02d}-model_states.pt" for tp in range(num_gpus)]
                            for
                            layer_id in [6]]
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(output_dir, "checkpoints", checkpoint)
            for key in keys_to_compare:
                for files in files_to_compare:
                    weights = [torch.load(os.path.join(checkpoint_path, file))[key] for file in files]
                    ref = weights[0]
                    for weight in weights[1:]:
                        torch_assert_equal(ref, weight, check_device=False)

        keys_to_compare = ["torch_rng_state"]
        files_to_compare = [[f"mp_rank_{tp:02d}_model_states.pt" for tp in range(num_gpus)]]
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(output_dir, "checkpoints", checkpoint)
            for key in keys_to_compare:
                for files in files_to_compare:
                    weights = [torch.load(os.path.join(checkpoint_path, file))[key] for file in files]
                    ref = weights[0]
                    for weight in weights[1:]:
                        assert ref == weight, f"key: {key} ref: {ref}, weight: {weight}"

        # # 2. test training from checkpoint: resume
        # # now do it again, this time resuming from the checkpoint
        # with CaptureStdout() as cs:
        #     execute_subprocess_async(cmd, env=self.get_env())
        #
        # # test checkpoint loading
        # self.assertIn(f"successfully loaded checkpoint from {output_dir}/checkpoints", cs.out)
        #
        # # test reports
        # self.assertIn("consumed samples", cs.out)

        # 3. test that inference with changes TP works.
        mp.set_start_method('spawn', force=True)
        del command_args["--rampup-batch-size"]
        command_args["--tensor-model-parallel-size"] = "1"
        del command_args["--load"]
        del command_args["--save"]
        command_args["--force-sync-layer-norm-parameters"] = ""

        checkpoints_path = os.path.join(output_dir, "checkpoints")
        pool = Pool(1)
        result = pool.map(MegDSTestTP.infer_model, [((0, 1, command_args, None, None, checkpoints_path))])
        pool.close()
        pool.join()

        output, tokens = result[0]
        logging.getLogger().info("First done!")

        command_args["--tensor-model-parallel-size"] = "2"

        pool = Pool(2)
        result = pool.map(MegDSTestTP.infer_model,
                          [((0, 2, command_args, tokens, None, checkpoints_path)), ((1, 2, command_args, tokens, None, checkpoints_path))])
        pool.close()
        pool.join()

        output2, tokens = result[0]

        logging.getLogger().critical(output - output2)
        self.assertTrue(np.allclose(output, output2, atol=0, rtol=0),
                        "Different results when running with TP=1 and TP=2")

if __name__ == '__main__':
    unittest.main()
