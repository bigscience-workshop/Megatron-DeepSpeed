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
import json
import re
import os
from pathlib import Path

from megatron.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    set_seed
)

set_seed(42)


def write_jsonl(path, lines_num=1000, line_length=1024):
    def get_text_line(line_length):
        # XXX: fix to generate line_length
        return "It's a wonderful world. I'm just walking on air. Talk of heaven on earth. I've got more than my share. Haven't got a care. Happy all day through. It's a wonderful world. Loving wonderful you!"

    with io.open(path, "w", encoding="utf-8") as f:

        for i in range(lines_num):
            rec = dict(text=get_text_line(line_length))
            x = json.dumps(rec, indent=0, ensure_ascii=False)
            x = re.sub(r'\n', ' ', x, 0, re.M)
            f.write(x + "\n")

class MegDSTestPreprocessing(TestCasePlus):
    """ """

    def setUp(self):
        super().setUp()


    def test_preprocess_data(self):
        src_dir = self.src_dir
        data_dir = f"{self.data_dir}/gpt2"
        output_dir = self.get_auto_remove_tmp_dir() # "./xxx", after=False)

        # autogenerate "input.jsonl"
        input_path = f"{output_dir}/input.jsonl"
        write_jsonl(input_path)

        output_prefix =f"{output_dir}/test-ds"

        cmd = f"""
        python {src_dir}/tools/preprocess_data.py
            --input {input_path}
            --output-prefix {output_prefix}
            --dataset-impl mmap
            --tokenizer-type GPT2BPETokenizer
            --merge-file {data_dir}/gpt2-tiny-merges.txt
            --vocab {data_dir}/gpt2-tiny-vocab.json
            --append-eod
            --workers 2
        """.split()

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        for ext in ["bin", "idx"]:
            tgt_path = f"{output_prefix}_text_document.{ext}"
            self.assertTrue(Path(tgt_path).exists(), )
