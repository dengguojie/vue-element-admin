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
import tensorflow as tf
import numpy as np


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
    """
    prama1: data_file: the file which store the generation data
    param2: shape: data shape
    param3: dtype: data type
    param4: rand_type: the method of generate data, select from "randint, uniform"
    param5: data lower limit
    param6: data upper limit
    """
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    elif rand_type == "uniform":
        rand_data = np.random.uniform(low, high, size=shape)
    elif rand_type == "complex":
        r1 = np.random.uniform(low, high, size=shape)
        r2 = np.random.uniform(low, high, size=shape)
        rand_data = np.empty((shape[0], shape[1], shape[2]), dtype=dtype)
        for i in range(0, shape[0]):
            for p in range(0, shape[1]):
                for k in range(0, shape[2]):
                    rand_data[i, p, k] = complex(r1[i, p, k], r2[i, p, k])
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


def gen_random_data_float16():
    data_files = ["erfc/data/erfc_data_input_float16.txt",
                  "erfc/data/erfc_data_output_float16.txt"]
    np.random.seed(3456)
    shape_input_data = [3, 4, 5]
    a = gen_data_file(data_files[0], shape_input_data, np.float16, "uniform", -1, 1)
    input_data = tf.compat.v1.placeholder(tf.float16, shape=shape_input_data)
    re = tf.math.erfc(input_data)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_float16_big():
    data_files = ["erfc/data/erfc_data_input_float16_big.txt",
                  "erfc/data/erfc_data_output_float16_big.txt"]
    np.random.seed(3456)
    shape_input_data = [4, 50, 8, 10]
    a = gen_data_file(data_files[0], shape_input_data, np.float16, "uniform", -1, 1)
    input_data = tf.compat.v1.placeholder(tf.float16, shape=shape_input_data)
    re = tf.math.erfc(input_data)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_float():
    data_files = ["erfc/data/erfc_data_input_float.txt",
                  "erfc/data/erfc_data_output_float.txt"]
    np.random.seed(3456)
    shape_input_data = [3, 4, 5]
    a = gen_data_file(data_files[0], shape_input_data, np.float32, "uniform", -1, 1)
    input_data = tf.compat.v1.placeholder(tf.float32, shape=shape_input_data)
    re = tf.math.erfc(input_data)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_float_big():
    data_files = ["erfc/data/erfc_data_input_float_big.txt",
                  "erfc/data/erfc_data_output_float_big.txt"]
    np.random.seed(3456)
    shape_input_data = [4, 50, 8, 10]
    a = gen_data_file(data_files[0], shape_input_data, np.float32, "uniform", -1, 1)
    input_data = tf.compat.v1.placeholder(tf.float32, shape=shape_input_data)
    re = tf.math.erfc(input_data)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_double():
    data_files = ["erfc/data/erfc_data_input_double.txt",
                  "erfc/data/erfc_data_output_double.txt"]
    np.random.seed(3456)
    shape_input_data = [3, 4, 5]
    a = gen_data_file(data_files[0], shape_input_data, np.float64, "uniform", -1, 1)
    input_data = tf.compat.v1.placeholder(tf.float64, shape=shape_input_data)
    re = tf.math.erfc(input_data)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_double_big():
    data_files = ["erfc/data/erfc_data_input_double_big.txt",
                  "erfc/data/erfc_data_output_double_big.txt"]
    np.random.seed(3456)
    shape_input_data = [4, 50, 8, 10]
    a = gen_data_file(data_files[0], shape_input_data, np.float64, "uniform", -1, 1)
    input_data = tf.compat.v1.placeholder(tf.float64, shape=shape_input_data)
    re = tf.math.erfc(input_data)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_float_1():
    data_files = ["erfc/data/erfc_data_input_float_1.txt",
                  "erfc/data/erfc_data_output_float_1.txt"]
    np.random.seed(3456)
    shape_input_data = [3, 4, 5]
    a = gen_data_file(data_files[0], shape_input_data, np.float32, "uniform", 1, 10)
    input_data = tf.compat.v1.placeholder(tf.float32, shape=shape_input_data)
    re = tf.math.erfc(input_data)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_double_1():
    data_files = ["erfc/data/erfc_data_input_double_1.txt",
                  "erfc/data/erfc_data_output_double_1.txt"]
    np.random.seed(3456)
    shape_input_data = [3, 4, 5]
    a = gen_data_file(data_files[0], shape_input_data, np.float64, "uniform", 1, 10)
    input_data = tf.compat.v1.placeholder(tf.float64, shape=shape_input_data)
    re = tf.math.erfc(input_data)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data: a})
    write_file_txt(data_files[1], data, fmt="%s")


def run():
    gen_random_data_float16()
    gen_random_data_float16_big()
    gen_random_data_float()
    gen_random_data_float_big()
    gen_random_data_double()
    gen_random_data_double_big()
    gen_random_data_float_1()
    gen_random_data_double_1()
