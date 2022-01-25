# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
from pathlib import Path

from megatron.testing_utils import (
    CaptureStdout,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    require_deepspeed,
    require_torch_gpu,
    require_torch_multi_gpu,
    set_seed
)

set_seed(42)


def get_launcher(num_gpus):
    # 1. explicitly set --num_nodes=1 just in case these tests end up run on a multi-node setup
    # - it won't be able to handle that
    return f"deepspeed --num_nodes 1 --num_gpus {num_gpus}".split()

@require_deepspeed
@require_torch_gpu
class MegDSTestCheckpoints(TestCasePlus):
    """ """

    def setUp(self):
        super().setUp()

        # at times magatron fails to build kernels and doesn't remove the lock file, which makes
        # subsequent runs hang - so make sure there is no lock when starting the testing
        meg_lock_file_path = self.repo_root_dir_str + "/megatron/fused_kernels/build/lock"
        if os.path.exists(meg_lock_file_path):
            os.unlink(meg_lock_file_path)

    def get_config(self, output_dir, tp_size, pp_size, dp_size):
        data_dir = f"{self.data_dir}/gpt2"

        num_gpus = pp_size * tp_size * dp_size
        print(f"Using {num_gpus} GPUs")

        n_samples = 300 # about 56 iterations

        exit_interval = 20 # some samples in the first half and then some more in the 2nd half after resume
        seq_len = 128

        # common/shared configs

        ds_args = f"""
                --deepspeed
                --deepspeed_config {self.test_file_dir_str}/ds_config.json
                --zero-stage 1
                --deepspeed-activation-checkpointing
        """.split()

        args = f"""
                --tensor-model-parallel-size {tp_size}
                --pipeline-model-parallel-size {pp_size}
                --distributed-backend nccl

                --log-interval 1
                --save-interval 20
                --eval-interval 10
                --eval-iters 5
                --checkpoint-activations
                --partition-activations
                --exit-interval {exit_interval}

                --merge-file {data_dir}/gpt2-tiny-merges.txt
                --vocab-file {data_dir}/gpt2-tiny-vocab.json
                --save {output_dir}/checkpoints
                --load {output_dir}/checkpoints
                --data-path {data_dir}/meg-gpt2-openwebtext_text_document
                --tensorboard-dir {output_dir}/tensorboard
                --tensorboard-queue-size 5
                --log-timers-to-tensorboard
                --log-batch-size-to-tensorboard
                --log-validation-ppl-to-tensorboard

                --num-layers 2
                --hidden-size 64
                --num-attention-heads 2
                --seq-length {seq_len}
                --max-position-embeddings 1024
                --micro-batch-size 1
                --global-batch-size 16
                --rampup-batch-size 2 2 {n_samples}
                --train-samples {n_samples}

                --optimizer adam
                --adam-beta1 0.9
                --adam-beta2 0.95
                --adam-eps 1e-8
                --lr 1e-4
                --lr-warmup-samples 5
                --lr-decay-samples 6
                --clip-grad 1.0
                --weight-decay 1e-1
                --fp16

                --log-level debug
                --log-level-replica info
        """.split()

        # XXX: fails to handle:
        #--embed-layernorm
        #
# stderr: RuntimeError: Error(s) in loading state_dict for VocabParallelEmbedding:
# stderr:         size mismatch for norm.weight: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
# stderr:         size mismatch for norm.bias: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).

        return args, ds_args, num_gpus


    def train_checkpoint(self, output_dir, tp_size=1, pp_size=1, dp_size=1):
        src_dir = self.src_dir
        script = [f"{src_dir}/pretrain_gpt.py"]

        args, ds_args, num_gpus = self.get_config(output_dir, tp_size, pp_size, dp_size)
        launcher = get_launcher(num_gpus)
        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        # 1. test training from scratch (no checkpoint)
        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test deepspeed is running
        self.assertIn("DeepSpeed info", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test there should be no checkpoint this round
        self.assertIn(f"Unable to find latest file at {output_dir}/checkpoints/latest", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)


    def resume_from_checkpoint(self, output_dir, tp_size=1, pp_size=1, dp_size=1):
        src_dir = self.src_dir
        script = [f"{src_dir}/pretrain_gpt.py"]

        args, ds_args, num_gpus = self.get_config(output_dir, tp_size, pp_size, dp_size)
        launcher = get_launcher(num_gpus)
        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test checkpoint loading
        self.assertIn(f"successfully loaded checkpoint from {output_dir}/checkpoints", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)


    def reshape_checkpoint(self, input_dir, output_dir, target_tp_size, target_pp_size):
        cmd = f"""
            python tools/convert_checkpoint/deepspeed_to_deepspeed.py
            --input_folder   {input_dir}/checkpoints/global_step20
            --output_folder {output_dir}/checkpoints
            --target_tp {target_tp_size} --target_pp {target_pp_size}
        """.split()

        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        self.assertIn("Convert DeepSpeed Checkpoint to DeepSpeed Checkpoint", cs.out)



    @require_torch_multi_gpu
    def test_checkpoint_reshaping_tp2_pp1_dp1(self):
        # this test requires at least 2 gpus - will use only 2 gpus for now - XXX: extend to more gpus

        output_dir1 = self.get_auto_remove_tmp_dir() # "./xxx1", after=False)
        output_dir2 = self.get_auto_remove_tmp_dir() # "./xxx2", after=False)

        # 1. train with TP=2 / PP=1
        self.train_checkpoint(output_dir1, tp_size=2, pp_size=1, dp_size=1)

        # 2. convert checkpoint to TP=1 / PP=1
        self.reshape_checkpoint(input_dir=output_dir1, output_dir=output_dir2, target_tp_size=1, target_pp_size=1)

        # 3. check we can resume training from a reshaped checkpoint with TP=1 / PP=1
        self.resume_from_checkpoint(output_dir2, tp_size=1, pp_size=1, dp_size=1)


    @require_torch_multi_gpu
    def test_checkpoint_reshaping_tp2_pp2_dp1(self):
        # this test requires at least 4 gpus - will use only 2 gpus for now - XXX: extend to more gpus

        output_dir1 = self.get_auto_remove_tmp_dir() # "./xxx1", after=False)
        output_dir2 = self.get_auto_remove_tmp_dir() # "./xxx2", after=False)

        # 1. train with TP=2 / PP=2
        self.train_checkpoint(output_dir1, tp_size=2, pp_size=2, dp_size=1)

        # 2. convert checkpoint to TP=1 / PP=1
        self.reshape_checkpoint(input_dir=output_dir1, output_dir=output_dir2, target_tp_size=1, target_pp_size=1)

        # 3. check we can resume training from a reshaped checkpoint with TP=1 / PP=1
        self.resume_from_checkpoint(output_dir2, tp_size=1, pp_size=1, dp_size=1)


    @require_torch_multi_gpu
    def test_checkpoint_reshaping_tp1_pp2_dp1(self):
        # this test requires at least 2 gpus - will use only 2 gpus for now - XXX: extend to more gpus

        output_dir1 = self.get_auto_remove_tmp_dir() # "./xxx1", after=False)
        output_dir2 = self.get_auto_remove_tmp_dir() # "./xxx2", after=False)

        # 1. train with TP=1 / PP=2
        self.train_checkpoint(output_dir1, tp_size=1, pp_size=2, dp_size=1)

        # 2. convert checkpoint to TP=1 / PP=1
        with self.assertRaises(AssertionError) as context:
            self.reshape_checkpoint(input_dir=output_dir1, output_dir=output_dir2, target_tp_size=1, target_pp_size=1)

        # 3. check we can resume training from a reshaped checkpoint with TP=1 / PP=1
        self.resume_from_checkpoint(output_dir2, tp_size=1, pp_size=1, dp_size=1)


    @require_torch_multi_gpu
    def test_checkpoint_reshaping_empty_dir(self):
        # this test requires at least 2 gpus - will use only 2 gpus for now - XXX: extend to more gpus

        output_dir1 = self.get_auto_remove_tmp_dir() # "./xxx1", after=False)
        output_dir2 = self.get_auto_remove_tmp_dir() # "./xxx2", after=False)
        with self.assertRaises(RuntimeError) as context:
            self.reshape_checkpoint(input_dir=output_dir1+"/xyz", output_dir=output_dir2, target_tp_size=1, target_pp_size=1)
