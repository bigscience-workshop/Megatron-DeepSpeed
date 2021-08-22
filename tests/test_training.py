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

import dataclasses
import io
import json
import os
import glob
import unittest
from pathlib import Path

from megatron.testing_utils import (
    CaptureStd,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    require_deepspeed,
    require_torch_gpu,
    set_seed
)

set_seed(42)


def get_launcher(distributed=True):
    # 1. explicitly set --num_nodes=1 just in case these tests end up run on a multi-node setup
    # - it won't be able to handle that
    # 2. for now testing with just 2 gpus max (since some quality tests may give different
    # results with mode gpus because we use very little data)
    num_gpus = min(2, get_gpu_count()) if distributed else 1
    return f"deepspeed --num_nodes 1 --num_gpus {num_gpus}".split()


@require_deepspeed
@require_torch_gpu
class MegDSTestTraining(TestCasePlus):
    """ """

    def setUp(self):
        super().setUp()


    def test_training_all(self):
        # all in one test
        src_dir = self.src_dir
        data_dir = f"{self.data_dir}/gpt2"
        output_dir = self.get_auto_remove_tmp_dir() # "./xxx", after=False)

        pp_size = max(2, get_gpu_count()) # trying to test pp=2 or pp=1
        n_samples = 200 # about 37 iterations
        exit_interval = 20 # some samples in the first half and then some more in the 2nd half after resume
        args = f"""
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size {pp_size}
            --distributed-backend nccl

            --num-layers 2
            --hidden-size 64
            --num-attention-heads 2
            --seq-length 128
            --max-position-embeddings 1024
            --micro-batch-size 1
            --rampup-batch-size 2 2 {n_samples}
            --global-batch-size 16
            --train-samples {n_samples}

            --optimizer adam
            --adam-beta1 0.9
            --adam-beta2 0.95
            --adam-eps 1e-8
            --lr 1e-4
            --lr-warmup-samples 5
            --clip-grad 1.0
            --weight-decay 1e-1
            --fp16

            --log-interval 5
            --save-interval 10
            --eval-interval 10
            --eval-iters 5
            --checkpoint-activations
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
            --glu-activation geglu
        """.split()

        ds_args = f"""
            --deepspeed
            --deepspeed_config {self.test_file_dir_str}/ds_config.json
            --zero-stage 1
            --deepspeed-activation-checkpointing
        """.split()

        script = [f"{src_dir}/pretrain_gpt.py"]
        launcher = get_launcher()

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        # 1. test training from scratch (no checkpoint)
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test deepspeed is running
        self.assertIn("DeepSpeed info", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test there should be no checkpoint this round
        self.assertIn(f"Unable to find latest file at {output_dir}/checkpoints/latest", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)

        # test tensorboard
        tensorboard_files = glob.glob(f"{output_dir}/tensorboard/events*")
        self.assertEqual(len(tensorboard_files), 1, "tensorboard files")

        # 2. test training from checkpoint: resume
        # now do it again, this time resuming from the checkpoint
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test checkpoint loading
        self.assertIn(f"successfully loaded checkpoint from {output_dir}/checkpoints", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)

        # test tensorboard (1 file from the first run, plus 1 now)
        tensorboard_files = glob.glob(f"{output_dir}/tensorboard/events*")
        self.assertEqual(len(tensorboard_files), 2, "tensorboard files")
