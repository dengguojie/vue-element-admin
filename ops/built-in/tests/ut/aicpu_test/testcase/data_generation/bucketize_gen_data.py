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
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op, dtypes


def write_file_txt(file_name, data, fmt="%s"):
    # prama1: file_name: the file which store the data
    # param2: data: data which will be stored
    # param3: fmt: format
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')


def read_file_txt(file_name, dtype, delim=None):
    # prama1: file_name: the file which store the data
    # param2: dtype: data type
    # param3: delim: delimiter which is used to split data
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()


def read_file_txt_to_boll(file_name, delim=None):
    # prama1: file_name: the file which store the data
    # param2: delim: delimiter which is used to split data
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)


def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    # prama1: data_file: the file which store the generation data
    # param2: shape: data shape
    # param3: dtype: data type
    # param4: rand_type: the method of generate data, select from "randint, uniform"
    # param5: data lower limit
    # param6: data upper limit
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


def gen_random_data_int32():
    data_files = ["bucketize/data/bucketize_data_input1_1.txt",
                  "bucketize/data/bucketize_data_output1_1.txt"]
    np.random.seed(23457)
    shape_input = [2, 10]
    input_data = gen_data_file(data_files[0], shape_input, np.int32, "randint", 0, 100)
    bound = [0, 10, 50, 100]
    re = np.digitize(input_data, bound)
    write_file_txt(data_files[1], re, fmt="%s")


def gen_random_data_int64():
    data_files = ["bucketize/data/bucketize_data_input1_2.txt",
                  "bucketize/data/bucketize_data_output1_2.txt"]
    np.random.seed(23457)
    shape_input = [10, 3]
    input_data = gen_data_file(data_files[0], shape_input, np.int64, "randint", 0, 200)
    bound = [0, 10, 50, 100]
    re = np.digitize(input_data, bound)
    write_file_txt(data_files[1], re, fmt="%s")


def gen_random_data_float():
    data_files = ["bucketize/data/bucketize_data_input1_3.txt",
                  "bucketize/data/bucketize_data_output1_3.txt"]
    np.random.seed(23457)
    shape_input = [15, 5]
    input_data = gen_data_file(data_files[0], shape_input, np.float32, "uniform", 0, 150)
    bound = [0, 10, 50, 100]
    re = np.digitize(input_data, bound)
    write_file_txt(data_files[1], re, fmt="%s")


def gen_random_data_double():
    data_files = ["bucketize/data/bucketize_data_input1_4.txt",
                  "bucketize/data/bucketize_data_output1_4.txt"]
    np.random.seed(23457)
    shape_input = [5, 20]
    input_data = gen_data_file(data_files[0], shape_input, np.float64, "uniform", -50, 100)

    bound = [0, 10, 50, 100]
    re = np.digitize(input_data, bound)
    write_file_txt(data_files[1], re, fmt="%s")


def run():
    gen_random_data_int32()
    gen_random_data_int64()
    gen_random_data_float()
    gen_random_data_double()
