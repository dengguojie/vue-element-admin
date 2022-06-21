"""
Copyright(c) Huawei Technologies Co., Ltd.2021-2021.All rights reserved.

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
    if(file_name is None):
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
    if dtype == np.float16:
        data = np.array(rand_data, dtype=np.float32)
        write_file_txt(data_file, data, fmt="%s")
        data = np.array(data, dtype=np.float16)
    else:
        data = np.array(rand_data, dtype=dtype)
        write_file_txt(data_file, data, fmt="%s")
    return data

def gen_random_data_int32():
    data_files=["triu/data/triu_data_input1_1.txt",
                "triu/data/triu_data_output1_1.txt"]
    np.random.seed(12345)
    shape_x = [3, 4]
    a = gen_data_file(data_files[0], shape_x, np.int32, "randint", 0, 10)

    x = torch.tensor(a)
    data = torch.triu(x)
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_int64():
    data_files=["triu/data/triu_data_input1_2.txt",
                "triu/data/triu_data_output1_2.txt"]
    np.random.seed(12345)
    shape_x = [3,5]
    a = gen_data_file(data_files[0], shape_x, np.int64, "randint", -10000, 10000)

    x = torch.tensor(a)
    data = torch.triu(x)
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_float():
    data_files=["triu/data/triu_data_input1_3.txt",
                "triu/data/triu_data_output1_3.txt"]
    np.random.seed(12345)
    shape_x = [6, 4]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)

    x = torch.tensor(a)
    data = torch.triu(x)
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_double():
    data_files=["triu/data/triu_data_input1_4.txt",
                "triu/data/triu_data_output1_4.txt"]
    np.random.seed(12345)
    shape_x = [64, 64, 64]
    a = gen_data_file(data_files[0], shape_x, np.float64, "uniform", -100, 100)

    x = torch.tensor(a)
    data = torch.triu(x)
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_int8():
    data_files=["triu/data/triu_data_input1_5.txt",
                "triu/data/triu_data_output1_5.txt"]
    np.random.seed(12345)
    shape_x = [12, 8]
    a = gen_data_file(data_files[0], shape_x, np.int8, "randint", -100, 100)

    x = torch.tensor(a)
    data = torch.triu(x)
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_uint8():
    data_files=["triu/data/triu_data_input1_6.txt",
                "triu/data/triu_data_output1_6.txt"]
    np.random.seed(12345)
    shape_x = [10, 20]
    a = gen_data_file(data_files[0], shape_x, np.uint8, "randint", 0, 200)

    x = torch.tensor(a)
    data = torch.triu(x)
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_int16():
    data_files=["triu/data/triu_data_input1_7.txt",
                "triu/data/triu_data_output1_7.txt"]
    np.random.seed(12345)
    shape_x = [9, 6]
    a = gen_data_file(data_files[0], shape_x, np.int16, "randint", 0, 6000)

    x = torch.tensor(a)
    data = torch.triu(x)
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_float16():
    data_files=["triu/data/triu_data_input1_8.txt",
                "triu/data/triu_data_output1_8.txt"]
    np.random.seed(12345)
    shape_x = [4, 5]
    a = gen_data_file(data_files[0], shape_x, np.float16, "uniform", -10, 10)

    x = torch.tensor(a)
    data = torch.triu(x)
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_bool():
    data_files=["triu/data/triu_data_input1_9.txt",
                "triu/data/triu_data_output1_9.txt"]
    np.random.seed(12345)
    shape_x = [6, 5]
    a = gen_data_file(data_files[0], shape_x, np.bool, "randint", 0, 2)

    x = torch.tensor(a)
    data = torch.triu(x)
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_diagonal_positive():
    data_files=["triu/data/triu_data_input1_10.txt",
                "triu/data/triu_data_output1_10.txt"]
    np.random.seed(12345)
    shape_x = [8, 10]
    a = gen_data_file(data_files[0], shape_x, np.float64, "uniform", -100, 100)

    x = torch.tensor(a)
    data = torch.triu(x, 7)
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_diagonal_negative():
    data_files=["triu/data/triu_data_input1_11.txt",
                "triu/data/triu_data_output1_11.txt"]
    np.random.seed(12345)
    shape_x = [12, 10]
    a = gen_data_file(data_files[0], shape_x, np.int32, "uniform", 0, 100)

    x = torch.tensor(a)
    data = torch.triu(x, -9)
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_batch_matrixs():
    data_files=["triu/data/triu_data_input1_12.txt",
                "triu/data/triu_data_output1_12.txt"]
    np.random.seed(12345)
    shape_x = [2, 3, 6, 8]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)

    x = torch.tensor(a)
    data = torch.triu(x)
    write_file_txt(data_files[1], data, fmt="%s")

def run():
    gen_random_data_int32()
    gen_random_data_int64()
    gen_random_data_float()
    gen_random_data_double()
    gen_random_data_int8()
    gen_random_data_uint8()
    gen_random_data_int16()
    gen_random_data_float16()
    gen_random_data_bool()
    gen_random_data_diagonal_positive()
    gen_random_data_diagonal_negative()
    gen_random_data_batch_matrixs()
