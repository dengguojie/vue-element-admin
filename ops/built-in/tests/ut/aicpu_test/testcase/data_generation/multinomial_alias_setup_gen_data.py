"""
Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

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
import sys
import logging
import torch
import numpy as np

# prama1: file_name: the file which store the data
# param2: data: data which will be stored
# param3: fmt: format
def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')

# prama1: file_name: the file which store the data
# param2: dtype: data type
# param3: delim: delimiter which is used to split data
def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()

# prama1: data_file: the file which store the generation data
# param2: shape: data shape
# param3: dtype: data type
# param4: rand_type: the method of generate data, select from "randint, uniform"
# param5: data lower limit
# param6: data upper limit
def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    if dtype == np.float16:
        data = np.array(rand_data, dtype=np.float32)
        write_file_txt(data_file, data, fmt="%s")
        data = np.array(data, dtype=np.float16)
    else:
        data = np.array(rand_data, dtype=dtype)
        write_file_txt(data_file, data, fmt="%s")
    return data

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config


def gen_random_data_float():
    data_files=["multinomial_alias_setup/data/multinomial_alias_setup_data_input1_1.txt",
                "multinomial_alias_setup/data/multinomial_alias_setup_data_output1_1.txt",
                "multinomial_alias_setup/data/multinomial_alias_setup_data_output2_1.txt"]
    np.random.seed(23457)
    shape_x1 = [1024]
    prot = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 0.01)
    prot = torch.tensor(prot)
    j,q = torch._multinomial_alias_setup(prot)
    write_file_txt(data_files[1], j, fmt="%s")
    write_file_txt(data_files[2], q, fmt="%s")


def gen_random_data_double():
    data_files=["multinomial_alias_setup/data/multinomial_alias_setup_data_input1_2.txt",
                "multinomial_alias_setup/data/multinomial_alias_setup_data_output1_2.txt",
                "multinomial_alias_setup/data/multinomial_alias_setup_data_output2_2.txt"]
    np.random.seed(3442)
    shape_x1 = [512]
    prot = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 0.01)
    prot = torch.tensor(prot)
    j,q = torch._multinomial_alias_setup(prot)
    write_file_txt(data_files[1], j, fmt="%s")
    write_file_txt(data_files[2], q, fmt="%s")

def run():
    gen_random_data_float()
    gen_random_data_double()