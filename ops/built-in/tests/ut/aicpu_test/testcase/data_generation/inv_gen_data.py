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
    """
    prama1: file_name: the file which store the data
    param2: data: data which will be stored
    param3: fmt: format
    """
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')


def read_file_txt(file_name, dtype, delim=None):
    """
    prama1: file_name: the file which store the data
    param2: dtype: data type
    param3: delim: delimiter which is used to split data
    """
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()


def read_file_txt_to_bool(file_name, delim=None):
    """
    prama1: file_name: the file which store the data
    param2: delim: delimiter which is used to split data
    """
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
    elif rand_type == "uniform":
        rand_data = np.random.uniform(low, high, size=shape)
    else:
        r1 = np.random.uniform(low, high, size=shape)
        r2 = np.random.uniform(low, high, size=shape)
        rand_data = np.empty((shape[0], shape[1], shape[2]), dtype=dtype)
        for i in range(0, shape[0]):
            for p in range(0, shape[1]):
                for k in range(0, shape[2]):
                    rand_data[i, p, k] = complex(r1[i, p, k], r2[i, p, k])
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


def gen_random_data_float16_1d():
    data_files = ["inv/data/inv_data_input1_1.txt",
                  "inv/data/inv_data_output1_1.txt"]
    np.random.seed(3457)
    shape_x = [2]
    a = gen_data_file(data_files[0], shape_x, np.float16, "uniform", -10, 10)
    x = tf.compat.v1.placeholder(tf.float16, shape=shape_x)
    re = tf.math.reciprocal(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_float32_1d_big():
    data_files = ["inv/data/inv_data_input1_1_big.txt",
                  "inv/data/inv_data_output1_1_big.txt"]
    np.random.seed(3457)
    shape_x = [128 * 1024]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", -10, 10)
    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    re = tf.math.reciprocal(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_double_3d():
    data_files = ["inv/data/inv_data_input1_2.txt",
                  "inv/data/inv_data_output1_2.txt"]
    np.random.seed(3457)
    shape_x = [256, 1024, 32]
    a = gen_data_file(data_files[0], shape_x, np.float64, "uniform", -100, 100)
    x = tf.compat.v1.placeholder(tf.float64, shape=shape_x)
    re = tf.math.reciprocal(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_float16():
    data_files = ["inv/data/inv_data_input1_3.txt",
                  "inv/data/inv_data_output1_3.txt"]
    np.random.seed(3457)
    shape_x = [12, 130]
    a = gen_data_file(data_files[0], shape_x, np.float16, "uniform", -10, 10)
    x = tf.compat.v1.placeholder(tf.float16, shape=shape_x)
    re = tf.math.reciprocal(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_float():
    data_files = ["inv/data/inv_data_input1_4.txt",
                  "inv/data/inv_data_output1_4.txt"]
    np.random.seed(23457)
    shape_x = [15, 12, 30]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    re = tf.math.reciprocal(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_double():
    data_files = ["inv/data/inv_data_input1_5.txt",
                  "inv/data/inv_data_output1_5.txt"]
    np.random.seed(3457)
    shape_x = [7, 12, 30]
    a = gen_data_file(data_files[0], shape_x, np.float64, "uniform", -100, 100)
    x = tf.compat.v1.placeholder(tf.float64, shape=shape_x)
    re = tf.math.reciprocal(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_complex64():
    data_files = ["inv/data/inv_data_input1_6.txt",
                  "inv/data/inv_data_output1_6.txt"]
    np.random.seed(3457)
    shape_x = [10, 5, 5]
    a = gen_data_file(data_files[0], shape_x, np.complex64, "complex", 0, 1000)
    x = tf.compat.v1.placeholder(tf.complex64, shape=shape_x)
    re = tf.math.reciprocal(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_complex128():
    data_files = ["inv/data/inv_data_input1_7.txt",
                  "inv/data/inv_data_output1_7.txt"]
    np.random.seed(677)
    shape_x = [6, 6, 6]
    a = gen_data_file(data_files[0], shape_x, np.complex128, "complex", 0, 1000)
    x = tf.compat.v1.placeholder(tf.complex128, shape=shape_x)
    re = tf.math.reciprocal(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")


def run():
    gen_random_data_float16_1d()
    gen_random_data_float32_1d_big()
    gen_random_data_double_3d()
    gen_random_data_float16()
    gen_random_data_float()
    gen_random_data_double()
    gen_random_data_complex64()
    gen_random_data_complex128()
