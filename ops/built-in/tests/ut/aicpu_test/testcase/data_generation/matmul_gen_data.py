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
import sys
import logging
import tensorflow as tf
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
    data_files=["matmul/data/matmul_data_input1_1.txt",
                "matmul/data/matmul_data_input2_1.txt",
                "matmul/data/matmul_data_output1_1.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3]
    shape_x2 = [3,2]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.matmul(x1, x2, transpose_a = False, transpose_b = False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float2():
    data_files=["matmul/data/matmul_data_input1_2.txt",
                "matmul/data/matmul_data_input2_2.txt",
                "matmul/data/matmul_data_output1_2.txt"]
    np.random.seed(23457)
    shape_x1 = [3,2]
    shape_x2 = [3,2]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.matmul(x1, x2, transpose_a = True)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_double():
    data_files=["matmul/data/matmul_data_input1_3.txt",
                "matmul/data/matmul_data_input2_3.txt",
                "matmul/data/matmul_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3]
    shape_x2 = [3,2]
    a = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float64, shape=shape_x2)
    re = tf.matmul(x1, x2, transpose_a = False, transpose_b = False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_int32():
    data_files=["matmul/data/matmul_data_input1_4.txt",
                "matmul/data/matmul_data_input2_4.txt",
                "matmul/data/matmul_data_output1_4.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3]
    shape_x2 = [3,2]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.int32, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    re = tf.matmul(x1, x2, transpose_a = False, transpose_b = False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float16():
    data_files=["matmul/data/matmul_data_input1_5.txt",
                "matmul/data/matmul_data_input2_5.txt",
                "matmul/data/matmul_data_output1_5.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3]
    shape_x2 = [3,2]
    a = gen_data_file(data_files[0], shape_x1, np.float16, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float16, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float16, shape=shape_x2)
    re = tf.matmul(x1, x2, transpose_a = False, transpose_b = False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def run():
    gen_random_data_float()
    gen_random_data_float2()
    gen_random_data_double()
    gen_random_data_int32()
    gen_random_data_float16()
