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
        real = np.random.uniform(low, high, size=shape)
        imag = np.random.uniform(low, high, size=shape)
        rand_data = real + imag * 1j
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


def gen_random_data_float_1d():
    data_files = ["sqrtgrad/data/sqrtgrad_data_input1_1.txt",
                  "sqrtgrad/data/sqrtgrad_data_input2_1.txt",
                  "sqrtgrad/data/sqrtgrad_data_output1_1.txt"]
    np.random.seed(23457)
    shape_x1 = [15]
    shape_x2 = [15]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.raw_ops.SqrtGrad(y=x1, dy=x2, name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_float_2d():
    data_files = ["sqrtgrad/data/sqrtgrad_data_input1_2.txt",
                  "sqrtgrad/data/sqrtgrad_data_input2_2.txt",
                  "sqrtgrad/data/sqrtgrad_data_output1_2.txt"]
    np.random.seed(23457)
    shape_x1 = [5, 4]
    shape_x2 = [5, 4]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.raw_ops.SqrtGrad(y=x1, dy=x2, name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_float_3d():
    data_files = ["sqrtgrad/data/sqrtgrad_data_input1_3.txt",
                  "sqrtgrad/data/sqrtgrad_data_input2_3.txt",
                  "sqrtgrad/data/sqrtgrad_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x1 = [5, 4, 10]
    shape_x2 = [5, 4, 10]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.raw_ops.SqrtGrad(y=x1, dy=x2, name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_float_4d():
    data_files = ["sqrtgrad/data/sqrtgrad_data_input1_4.txt",
                  "sqrtgrad/data/sqrtgrad_data_input2_4.txt",
                  "sqrtgrad/data/sqrtgrad_data_output1_4.txt"]
    np.random.seed(23457)
    shape_x1 = [5, 4, 10, 2]
    shape_x2 = [5, 4, 10, 2]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.raw_ops.SqrtGrad(y=x1, dy=x2, name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_float_5d():
    data_files = ["sqrtgrad/data/sqrtgrad_data_input1_5.txt",
                  "sqrtgrad/data/sqrtgrad_data_input2_5.txt",
                  "sqrtgrad/data/sqrtgrad_data_output1_5.txt"]
    np.random.seed(23457)
    shape_x1 = [5, 4, 10, 2, 2]
    shape_x2 = [5, 4, 10, 2, 2]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.raw_ops.SqrtGrad(y=x1, dy=x2, name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_float16_1d():
    data_files = ["sqrtgrad/data/sqrtgrad_data_input1_6.txt",
                  "sqrtgrad/data/sqrtgrad_data_input2_6.txt",
                  "sqrtgrad/data/sqrtgrad_data_output1_6.txt"]
    np.random.seed(23457)
    shape_x1 = [12]
    shape_x2 = [12]
    a = gen_data_file(data_files[0], shape_x1, np.float16, "uniform", -10, 10)
    b = gen_data_file(data_files[1], shape_x2, np.float16, "uniform", -10, 10)

    x1 = tf.compat.v1.placeholder(tf.float16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float16, shape=shape_x2)
    re = tf.raw_ops.SqrtGrad(y=x1, dy=x2, name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_float16_2d():
    data_files = ["sqrtgrad/data/sqrtgrad_data_input1_7.txt",
                  "sqrtgrad/data/sqrtgrad_data_input2_7.txt",
                  "sqrtgrad/data/sqrtgrad_data_output1_7.txt"]
    np.random.seed(23457)
    shape_x1 = [8, 1024]
    shape_x2 = [8, 1024]
    a = gen_data_file(data_files[0], shape_x1, np.float16, "uniform", -10, 10)
    b = gen_data_file(data_files[1], shape_x2, np.float16, "uniform", -10, 10)

    x1 = tf.compat.v1.placeholder(tf.float16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float16, shape=shape_x2)
    re = tf.raw_ops.SqrtGrad(y=x1, dy=x2, name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_float16_3d():
    data_files = ["sqrtgrad/data/sqrtgrad_data_input1_8.txt",
                  "sqrtgrad/data/sqrtgrad_data_input2_8.txt",
                  "sqrtgrad/data/sqrtgrad_data_output1_8.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 16, 4]
    shape_x2 = [4, 16, 4]
    a = gen_data_file(data_files[0], shape_x1, np.float16, "uniform", -10, 10)
    b = gen_data_file(data_files[1], shape_x2, np.float16, "uniform", -10, 10)

    x1 = tf.compat.v1.placeholder(tf.float16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float16, shape=shape_x2)
    re = tf.raw_ops.SqrtGrad(y=x1, dy=x2, name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_double():
    data_files = ["sqrtgrad/data/sqrtgrad_data_input1_9.txt",
                  "sqrtgrad/data/sqrtgrad_data_input2_9.txt",
                  "sqrtgrad/data/sqrtgrad_data_output1_9.txt"]
    np.random.seed(3457)
    shape_x1 = [1, 4]
    shape_x2 = [1, 4]
    a = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", -100, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", -100, 100)

    x1 = tf.compat.v1.placeholder(tf.float64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float64, shape=shape_x2)
    re = tf.raw_ops.SqrtGrad(y=x1, dy=x2, name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_complex64():
    data_files = ["sqrtgrad/data/sqrtgrad_data_input1_10.txt",
                  "sqrtgrad/data/sqrtgrad_data_input2_10.txt",
                  "sqrtgrad/data/sqrtgrad_data_output1_10.txt"]
    np.random.seed(3457)
    shape_x1 = [2, 1024, 4]
    shape_x2 = [2, 1024, 4]
    a = gen_data_file(data_files[0], shape_x1, np.complex64, "complex", 0, 1000)
    b = gen_data_file(data_files[1], shape_x2, np.complex64, "complex", 0, 1000)
    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x2)
    re = tf.raw_ops.SqrtGrad(y=x1, dy=x2, name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_complex128():
    data_files = ["sqrtgrad/data/sqrtgrad_data_input1_11.txt",
                  "sqrtgrad/data/sqrtgrad_data_input2_11.txt",
                  "sqrtgrad/data/sqrtgrad_data_output1_11.txt"]
    np.random.seed(677)
    shape_x1 = [6, 6, 6]
    shape_x2 = [6, 6, 6]
    a = gen_data_file(data_files[0], shape_x1, np.complex128, "complex", 0, 1000)
    b = gen_data_file(data_files[1], shape_x2, np.complex128, "complex", 0, 1000)
    x1 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x2)
    re = tf.raw_ops.SqrtGrad(y=x1, dy=x2, name="re")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def run():
    gen_random_data_float_1d()
    gen_random_data_float_2d()
    gen_random_data_float_3d()
    gen_random_data_float_4d()
    gen_random_data_float_5d()
    gen_random_data_float16_1d()
    gen_random_data_float16_2d()
    gen_random_data_float16_3d()
    gen_random_data_double()
    gen_random_data_complex64()
    gen_random_data_complex128()
