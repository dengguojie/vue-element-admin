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
from tensorflow.python.ops import bitwise_ops


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


def gen_random_data_int8():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_1.txt",
                  "bitwise_or/data/bitwise_or_data_input2_1.txt",
                  "bitwise_or/data/bitwise_or_data_output1_1.txt"]
    np.random.seed(3457)
    shape_x1 = [7, 12]
    shape_x2 = [12]
    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", -100, 100)
    b = gen_data_file(data_files[1], shape_x2, np.int8, "randint", -100, 100)

    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int8, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int16():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_2.txt",
                  "bitwise_or/data/bitwise_or_data_input2_2.txt",
                  "bitwise_or/data/bitwise_or_data_output1_2.txt"]
    np.random.seed(3457)
    shape_x1 = [12, 6]
    shape_x2 = [6]
    a = gen_data_file(data_files[0], shape_x1, np.int16, "randint", 0, 6000)
    b = gen_data_file(data_files[1], shape_x2, np.int16, "randint", 0, 6000)

    x1 = tf.compat.v1.placeholder(tf.int16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int16, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int32():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_3.txt",
                  "bitwise_or/data/bitwise_or_data_input2_3.txt",
                  "bitwise_or/data/bitwise_or_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x1 = [6, 12]
    shape_x2 = [12]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "randint", 0, 10)
    b = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0, 10)

    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int64():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_4.txt",
                  "bitwise_or/data/bitwise_or_data_input2_4.txt",
                  "bitwise_or/data/bitwise_or_data_output1_4.txt"]
    np.random.seed(23457)
    shape_x1 = [13, 10, 4]
    shape_x2 = [13, 10, 4]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "randint", -10000, 10000)
    b = gen_data_file(data_files[1], shape_x2, np.int32, "randint", -10000, 10000)

    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_uint8():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_5.txt",
                  "bitwise_or/data/bitwise_or_data_input2_5.txt",
                  "bitwise_or/data/bitwise_or_data_output1_5.txt"]
    np.random.seed(3457)
    shape_x1 = [7, 12]
    shape_x2 = [12]
    a = gen_data_file(data_files[0], shape_x1, np.uint8, "randint", 0, 200)
    b = gen_data_file(data_files[1], shape_x2, np.uint8, "randint", 0, 200)

    x1 = tf.compat.v1.placeholder(tf.uint8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.uint8, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_uint16():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_6.txt",
                  "bitwise_or/data/bitwise_or_data_input2_6.txt",
                  "bitwise_or/data/bitwise_or_data_output1_6.txt"]
    np.random.seed(3457)
    shape_x1 = [2, 3]
    shape_x2 = [3]
    a = gen_data_file(data_files[0], shape_x1, np.uint16, "randint", 0, 6000)
    b = gen_data_file(data_files[1], shape_x2, np.uint16, "randint", 0, 6000)

    x1 = tf.compat.v1.placeholder(tf.uint16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.uint16, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_uint32():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_7.txt",
                  "bitwise_or/data/bitwise_or_data_input2_7.txt",
                  "bitwise_or/data/bitwise_or_data_output1_7.txt"]
    np.random.seed(3457)
    shape_x1 = [2, 3]
    shape_x2 = [3]
    a = gen_data_file(data_files[0], shape_x1, np.uint32, "randint", 0, 6000)
    b = gen_data_file(data_files[1], shape_x2, np.uint32, "randint", 0, 6000)

    x1 = tf.compat.v1.placeholder(tf.uint32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.uint32, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_uint64():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_8.txt",
                  "bitwise_or/data/bitwise_or_data_input2_8.txt",
                  "bitwise_or/data/bitwise_or_data_output1_8.txt"]
    np.random.seed(3457)
    shape_x1 = [2, 3]
    shape_x2 = [3]
    a = gen_data_file(data_files[0], shape_x1, np.uint64, "randint", 0, 6000)
    b = gen_data_file(data_files[1], shape_x2, np.uint64, "randint", 0, 6000)

    x1 = tf.compat.v1.placeholder(tf.uint64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.uint64, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_x_one():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_9.txt",
                  "bitwise_or/data/bitwise_or_data_input2_9.txt",
                  "bitwise_or/data/bitwise_or_data_output1_9.txt"]
    np.random.seed(3457)
    shape_x1 = [1]
    shape_x2 = [12]
    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", -100, 100)
    b = gen_data_file(data_files[1], shape_x2, np.int8, "randint", -100, 100)

    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int8, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_y_one():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_10.txt",
                  "bitwise_or/data/bitwise_or_data_input2_10.txt",
                  "bitwise_or/data/bitwise_or_data_output1_10.txt"]
    np.random.seed(3457)
    shape_x1 = [12]
    shape_x2 = [1]
    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", -100, 100)
    b = gen_data_file(data_files[1], shape_x2, np.int8, "randint", -100, 100)

    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int8, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_size_bcast_64KB():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_11.txt",
                  "bitwise_or/data/bitwise_or_data_input2_11.txt",
                  "bitwise_or/data/bitwise_or_data_output1_11.txt"]
    np.random.seed(3457)
    shape_x1 = [4, 32, 64]
    shape_x2 = [4, 1, 64]
    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", -100, 100)
    b = gen_data_file(data_files[1], shape_x2, np.int8, "randint", -100, 100)

    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int8, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_size_bcast_256KB():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_12.txt",
                  "bitwise_or/data/bitwise_or_data_input2_12.txt",
                  "bitwise_or/data/bitwise_or_data_output1_12.txt"]
    np.random.seed(3457)
    shape_x1 = [4, 32, 256]
    shape_x2 = [4, 1, 256]
    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", -100, 100)
    b = gen_data_file(data_files[1], shape_x2, np.int8, "randint", -100, 100)

    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int8, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_size_no_bcast_512KB():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_13.txt",
                  "bitwise_or/data/bitwise_or_data_input2_13.txt",
                  "bitwise_or/data/bitwise_or_data_output1_13.txt"]
    np.random.seed(3457)
    shape_x1 = [4, 32, 512]
    shape_x2 = [4, 32, 512]
    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", -100, 100)
    b = gen_data_file(data_files[1], shape_x2, np.int8, "randint", -100, 100)

    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int8, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_size_no_bcast_2M():
    data_files = ["bitwise_or/data/bitwise_or_data_input1_14.txt",
                  "bitwise_or/data/bitwise_or_data_input2_14.txt",
                  "bitwise_or/data/bitwise_or_data_output1_14.txt"]
    np.random.seed(3457)
    shape_x1 = [4, 32, 512, 4]
    shape_x2 = [4, 32, 512, 4]
    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", -100, 100)
    b = gen_data_file(data_files[1], shape_x2, np.int8, "randint", -100, 100)

    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int8, shape=shape_x2)
    re = bitwise_ops.bitwise_or(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def run():
    gen_random_data_int8()
    gen_random_data_int16()
    gen_random_data_int32()
    gen_random_data_int64()
    gen_random_data_uint8()
    gen_random_data_uint16()
    gen_random_data_uint32()
    gen_random_data_uint64()
    gen_random_data_x_one()
    gen_random_data_y_one()
    gen_random_data_size_bcast_64KB()
    gen_random_data_size_bcast_256KB()
    gen_random_data_size_no_bcast_512KB()
    gen_random_data_size_no_bcast_2M()
