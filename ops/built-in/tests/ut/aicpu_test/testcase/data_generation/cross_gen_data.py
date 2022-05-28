"""
Copyright 2021 Huawei Technologies Co., Ltd

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


from tensorflow.python.framework import constant_op, dtypes


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


# prama1: file_name: the file which store the data
# param2: delim: delimiter which is used to split data
def read_file_txt_to_bool(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)


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
    data = np.array(rand_data, dtype=dtype)
    write_file_txt(data_file, data, fmt="%s")
    return data


def config(execute_type):
    if execute_type == 'cpu':
        session_config = torch.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config

def gen_random_x_int8_int8():
    data_files = ["cross/data/Cross_data_input1_4.txt",
                  "cross/data/Cross_data_input2_4.txt",
                  "cross/data/Cross_data_output1_4.txt"]
    np.random.seed(2369)
    shape_x1 = [3,2,3]
    shape_x2 = [3,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.int8, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.int8, "uniform", 0, 10)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,0)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_int32_int32():
    data_files = ["cross/data/Cross_data_input1_5.txt",
                  "cross/data/Cross_data_input2_5.txt",
                  "cross/data/Cross_data_output1_5.txt"]
    np.random.seed(2369)
    shape_x1 = [3,2,3]
    shape_x2 = [3,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.int32, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.int32, "uniform", 0, 10)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,2)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_int16_int16():
    data_files = ["cross/data/Cross_data_input1_1.txt",
                  "cross/data/Cross_data_input2_1.txt",
                  "cross/data/Cross_data_output1_1.txt"]
    np.random.seed(2369)
    shape_x1 = [2,3,3]
    shape_x2 = [2,3,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.int16, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.int16, "uniform", 0, 10)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,1)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_int64_int64():
    data_files = ["cross/data/Cross_data_input1_2.txt",
                  "cross/data/Cross_data_input2_2.txt",
                  "cross/data/Cross_data_output1_2.txt"]
    np.random.seed(2369)
    shape_x1 = [2,3,3]
    shape_x2 = [2,3,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.int64, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.int64, "uniform", 0, 10)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,2)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_int64_int64random():
    data_files = ["cross/data/Cross_data_input1_3.txt",
                  "cross/data/Cross_data_input2_3.txt",
                  "cross/data/Cross_data_output1_3.txt"]
    np.random.seed(2369)
    shape_x1 = [2,3,3,3]
    shape_x2 = [2,3,3,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.int64, "uniform", 0, 100)
    x2 = gen_data_file(data_files[1], shape_x2, np.int64, "uniform", 0, 100)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,1)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_uint8_uint8():
    data_files = ["cross/data/Cross_data_input1_6.txt",
                  "cross/data/Cross_data_input2_6.txt",
                  "cross/data/Cross_data_output1_6.txt"]
    np.random.seed(2369)
    shape_x1 = [1,2,3]
    shape_x2 = [1,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.uint8, "uniform", 0, 200)
    x2 = gen_data_file(data_files[1], shape_x2, np.uint8, "uniform", 0, 200)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,2)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_uint16_uint16():
    data_files = ["cross/data/Cross_data_input1_7.txt",
                  "cross/data/Cross_data_input2_7.txt",
                  "cross/data/Cross_data_output1_7.txt"]
    np.random.seed(2369)
    shape_x1 = [1,2,3]
    shape_x2 = [1,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.int64, "uniform", 0, 100)
    x2 = gen_data_file(data_files[1], shape_x2, np.int64, "uniform", 0, 100)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,2)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_uint32_uint32():
    data_files = ["cross/data/Cross_data_input1_8.txt",
                  "cross/data/Cross_data_input2_8.txt",
                  "cross/data/Cross_data_output1_8.txt"]
    np.random.seed(2369)
    shape_x1 = [1,2,3]
    shape_x2 = [1,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.int64, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.int64, "uniform", 0, 10)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,2)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_uint64_uint64():
    data_files = ["cross/data/Cross_data_input1_9.txt",
                  "cross/data/Cross_data_input2_9.txt",
                  "cross/data/Cross_data_output1_9.txt"]
    np.random.seed(2369)
    shape_x1 = [1,2,3]
    shape_x2 = [1,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.int64, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.int64, "uniform", 0, 10)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,2)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float64_float64():
    data_files = ["cross/data/Cross_data_input1_10.txt",
                  "cross/data/Cross_data_input2_10.txt",
                  "cross/data/Cross_data_output1_10.txt"]
    np.random.seed(2369)
    shape_x1 = [1,2,3]
    shape_x2 = [1,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", 0, 10)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,2)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float32_float32():
    data_files = ["cross/data/Cross_data_input1_11.txt",
                  "cross/data/Cross_data_input2_11.txt",
                  "cross/data/Cross_data_output1_11.txt"]
    np.random.seed(2369)
    shape_x1 = [1,2,3]
    shape_x2 = [1,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    x2 = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,2)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_int8_int8random():
    data_files = ["cross/data/Cross_data_input1_12.txt",
                  "cross/data/Cross_data_input2_12.txt",
                  "cross/data/Cross_data_output1_12.txt"]
    np.random.seed(2369)
    shape_x1 = [3,2,2,3]
    shape_x2 = [3,2,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.int8, "uniform", -100, 100)
    x2 = gen_data_file(data_files[1], shape_x2, np.int8, "uniform", -100, 100)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,-1)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_int16_int16random():
    data_files = ["cross/data/Cross_data_input1_13.txt",
                  "cross/data/Cross_data_input2_13.txt",
                  "cross/data/Cross_data_output1_13.txt"]
    np.random.seed(2369)
    shape_x1 = [3,2,3,3]
    shape_x2 = [3,2,3,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.int16, "uniform", -1000, 1000)
    x2 = gen_data_file(data_files[1], shape_x2, np.int16, "uniform", -1000, 1000)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,-2)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_int32_int32random():
    data_files = ["cross/data/Cross_data_input1_14.txt",
                  "cross/data/Cross_data_input2_14.txt",
                  "cross/data/Cross_data_output1_14.txt"]
    np.random.seed(2369)
    shape_x1 = [3,3,2,3]
    shape_x2 = [3,3,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.int32, "uniform", -1000, 1000)
    x2 = gen_data_file(data_files[1], shape_x2, np.int32, "uniform", -1000, 1000)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,-3)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_uint8_uint8random():
    data_files = ["cross/data/Cross_data_input1_15.txt",
                  "cross/data/Cross_data_input2_15.txt",
                  "cross/data/Cross_data_output1_15.txt"]
    np.random.seed(2369)
    shape_x1 = [3,2,2,3]
    shape_x2 = [3,2,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.uint8, "uniform", 0, 200)
    x2 = gen_data_file(data_files[1], shape_x2, np.uint8, "uniform", 0, 200)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,0)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_double_double():
    data_files = ["cross/data/Cross_data_input1_16.txt",
                  "cross/data/Cross_data_input2_16.txt",
                  "cross/data/Cross_data_output1_16.txt"]
    np.random.seed(2369)
    shape_x1 = [3,2,3]
    shape_x2 = [3,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 200)
    x2 = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", 0, 200)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,0)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float16_float16():
    data_files = ["cross/data/Cross_data_input1_19.txt",
                  "cross/data/Cross_data_input2_19.txt",
                  "cross/data/Cross_data_output1_19.txt"]
    np.random.seed(2369)
    shape_x1 = [1,2,3]
    shape_x2 = [1,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 2000)
    x2 = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 2000)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22,2)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_default_dim():
    data_files = ["cross/data/Cross_data_input1_20.txt",
                  "cross/data/Cross_data_input2_20.txt",
                  "cross/data/Cross_data_output1_20.txt"]
    np.random.seed(2369)
    shape_x1 = [2,2,3]
    shape_x2 = [2,2,3]
    print(data_files[0], data_files[1], data_files[2])
    x1 = gen_data_file(data_files[0], shape_x1, np.uint8, "uniform", 0, 200)
    x2 = gen_data_file(data_files[1], shape_x2, np.uint8, "uniform", 0, 200)
    x11 = torch.from_numpy(x1)
    x22 = torch.from_numpy(x2)
    result = torch.cross(x11,x22)

    write_file_txt(data_files[0], x11, fmt="%s")
    write_file_txt(data_files[1], x22, fmt="%s")
    write_file_txt(data_files[2], result, fmt="%s")

def run():
    gen_random_x_int16_int16()
    gen_random_x_int64_int64()
    gen_random_x_int64_int64random()
    gen_random_x_int8_int8()
    gen_random_x_int32_int32()
    gen_random_x_uint8_uint8()
    gen_random_x_uint16_uint16()
    gen_random_x_uint32_uint32()
    gen_random_x_uint64_uint64()
    gen_random_x_float64_float64()
    gen_random_x_float32_float32()
    gen_random_x_int8_int8random()
    gen_random_x_int16_int16random()
    gen_random_x_int32_int32random()
    gen_random_x_uint8_uint8random()
    gen_random_x_double_double()
    gen_random_x_float16_float16()
    gen_random_x_default_dim()
  
    



