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
from binascii import unhexlify


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
def read_file_txt_to_boll(file_name, delim=None):
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


def gen_data_file_inf(data_file, dtype):
    data = []
    if dtype == np.float16:
        temp_data = unhexlify(bytes("007C" * 10, 'utf-8'))
        data = np.frombuffer(temp_data, dtype)
    elif dtype == np.float32:
        temp_data = unhexlify(bytes("0000807F" * 10, 'utf-8'))
        data = np.frombuffer(temp_data, dtype)
    else:
        temp_data = unhexlify(bytes("000000000000F07F" * 10, 'utf-8'))
        data = np.frombuffer(temp_data, dtype)

    write_file_txt(data_file, data, fmt="%s")
    return data


# def gen_data_file_inf(data_file, shape, dtype, rand_type, low, high):
#     if rand_type == "randint":
#         rand_data = np.random.randint(low, high, size=shape)
#     else:
#         rand_data = np.random.uniform(low, high, size=shape)
#     data = np.array(rand_data, dtype=dtype)

#     for i in range(0,shape[0]):
#         data[i] = float('inf').half()

#     write_file_txt(data_file, data, fmt="%s")
#     return data


def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    return session_config


def gen_random_data_float16(data_files):
    np.random.seed(23457)
    shape_x = [15, 12, 30]
    a = gen_data_file(data_files[0], shape_x, np.float16, "uniform", -100, 100)
    x = tf.compat.v1.placeholder(tf.float16, shape=shape_x)
    re = tf.math.is_inf(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")
    # read_data = write_file_txt(data_files[1], np.float16)


def gen_random_data_float(data_files):
    np.random.seed(23457)
    shape_x = [15, 12, 30]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", -100, 100)
    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    re = tf.math.is_inf(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")
    # read_data = write_file_txt(data_files[1], np.float32)


def gen_random_data_double(data_files):
    np.random.seed(23457)
    shape_x = [64, 32, 32]
    a = gen_data_file(data_files[0], shape_x, np.float64, "uniform", 0, 100)
    x = tf.compat.v1.placeholder(tf.double, shape=shape_x)
    re = tf.math.is_inf(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")
    # read_data = write_file_txt(data_files[1], np.double)


def gen_random_data_float16_inf(data_files):
    np.random.seed(23457)
    shape_x = [10]
    a = gen_data_file_inf(data_files[0], np.float16)
    x = tf.compat.v1.placeholder(tf.float16, shape=shape_x)
    re = tf.math.is_inf(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")
    # read_data = write_file_txt(data_files[1], np.double)


def gen_random_data_float_inf(data_files):
    np.random.seed(23457)
    shape_x = [10]
    a = gen_data_file_inf(data_files[0], np.float32)
    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    re = tf.math.is_inf(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")
    # read_data = write_file_txt(data_files[1], np.double)


def gen_random_data_double_inf(data_files):
    np.random.seed(23457)
    shape_x = [10]
    a = gen_data_file_inf(data_files[0], np.float64)
    x = tf.compat.v1.placeholder(tf.double, shape=shape_x)
    re = tf.math.is_inf(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")
    # read_data = write_file_txt(data_files[1], np.double)


def run():
    data_files = [
        "is_inf/data/is_inf_data_input_1.txt", "is_inf/data/is_inf_data_output_1.txt",
        "is_inf/data/is_inf_data_input_2.txt", "is_inf/data/is_inf_data_output_2.txt",
        "is_inf/data/is_inf_data_input_3.txt", "is_inf/data/is_inf_data_output_3.txt",
        "is_inf/data/is_inf_data_input_4.txt", "is_inf/data/is_inf_data_output_4.txt",
        "is_inf/data/is_inf_data_input_5.txt", "is_inf/data/is_inf_data_output_5.txt",
        "is_inf/data/is_inf_data_input_6.txt", "is_inf/data/is_inf_data_output_6.txt"
    ]

    gen_random_data_float16([data_files[0], data_files[1]])
    gen_random_data_float([data_files[2], data_files[3]])
    gen_random_data_double([data_files[4], data_files[5]])

    gen_random_data_float16_inf([data_files[6], data_files[7]])
    gen_random_data_float_inf([data_files[8], data_files[9]])
    gen_random_data_double_inf([data_files[10], data_files[11]])
