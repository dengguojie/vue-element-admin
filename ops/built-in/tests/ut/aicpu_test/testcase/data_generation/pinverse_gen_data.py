"""
Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

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


def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')


def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()


def read_file_to_bool(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)


def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    else:
        rand_data = np.random.uniform(low, high, size=shape)

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
    data_files = ["pinverse/data/pinverse_data_input_float.txt",
                  "pinverse/data/pinverse_data_output_float.txt"]
    np.random.seed(12345)
    shape_x = [3, 5]
    a = gen_data_file(data_files[0], shape_x, np.float, "randint", -10, 10)

    x = torch.tensor(a)
    data = torch.pinverse(x)
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_double():
    data_files = ["pinverse/data/pinverse_data_input_double.txt",
                  "pinverse/data/pinverse_data_output_double.txt"]
    np.random.seed(12345)
    shape_x = [3, 3]
    a = gen_data_file(data_files[0], shape_x, np.double, "uniform", -100, 100)

    x = torch.tensor(a)
    data = torch.pinverse(x)
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_double_rcond():
    data_files = ["pinverse/data/pinverse_data_input_double_rcond.txt",
                  "pinverse/data/pinverse_data_output_double_rcond.txt"]
    np.random.seed(12345)
    shape_x = [3, 3]
    a = gen_data_file(data_files[0], shape_x, np.double, "uniform", -100, 100)

    x = torch.tensor(a)
    data = torch.pinverse(x, rcond=1e-8)
    write_file_txt(data_files[1], data, fmt="%s")


def run():
    gen_random_data_float()
    gen_random_data_double()
    gen_random_data_double_rcond()


run()
