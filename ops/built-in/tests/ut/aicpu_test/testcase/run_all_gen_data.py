"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#set tensorflow log error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import sys
import numpy as np
from data_generation import *

data_files = os.listdir("./data_generation")
modules = []
for data_file in data_files:
    split_item = os.path.splitext(data_file)
    if split_item[1] == ".py":
        if split_item[0][-9:] == "_gen_data":
            modules.append(split_item[0])

def run():
    for module in modules:
        print("===run", module, "to generate data begin===")
        eval(module).run()

if __name__ == "__main__":
    run()
