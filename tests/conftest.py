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

# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import sys
import warnings
import deepspeed
from os.path import abspath, dirname, join

# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__))))
sys.path.insert(1, git_repo_path)

# silence FutureWarning warnings in tests since often we can't act on them until
# they become normal warnings - i.e. the tests still need to test the current functionality
warnings.simplefilter(action="ignore", category=FutureWarning)


def pytest_sessionstart(session):
    # since some of us use use deepspeed@master for other uses make sure we test against the correct
    # deepspeed branch. eventually this won't be needed by for now we use a frozen branch
    if deepspeed.__git_branch__ != "big-science":
        raise ValueError(f"detected deepspeed.__git_branch__={deepspeed.__git_branch__ }, but needing branch='big-science'.\n"
                         "You can install it with: pip install git+https://github.com/microsoft/DeepSpeed.git@big-science")


def pytest_sessionfinish(session, exitstatus):
    # If no tests are collected, pytest exists with code 5, which makes the CI fail.
    if exitstatus == 5:
        session.exitstatus = 0
