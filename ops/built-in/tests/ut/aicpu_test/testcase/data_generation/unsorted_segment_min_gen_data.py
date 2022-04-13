"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unrsqrt required by applicable law or agreed to in writing, software
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


def gen_random_data_file(data_file, shape, dtype, rand_type, low, high):
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


def gen_normal_data_file(data_file, normal_data, dtype):
    sess = tf.compat.v1.Session()
    normal_data = sess.run(normal_data)
    data = np.array(normal_data, dtype=dtype)
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
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_1.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_1.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_1.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_1.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.int32, "randint", 0, 10)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 0]), np.int32)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_int32_withint64id():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_2.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_2.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_2.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_2.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.int32, "randint", 0, 10)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 0]), np.int64)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int64)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_int16():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_3.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_3.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_3.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x1 = [5, 6]
    a = gen_random_data_file(data_files[0], shape_x1, np.int16, "randint", -100, 100)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 2, 3, 4]), np.int32)
    c = gen_normal_data_file(data_files[2], tf.constant(5), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_int16_withint64id():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_4.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_4.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_4.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_4.txt"]
    np.random.seed(23457)
    shape_x1 = [5, 6]
    a = gen_random_data_file(data_files[0], shape_x1, np.int16, "randint", -100, 100)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 2, 3, 4]), np.int64)
    c = gen_normal_data_file(data_files[2], tf.constant(5), np.int64)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_float():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_5.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_5.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_5.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_5.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.float, "uniform", 0, 10)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 2]), np.int32)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_float_withint64id():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_6.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_6.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_6.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_6.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.float, "uniform", 0, 10)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 2]), np.int64)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int64)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_double():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_7.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_7.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_7.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_7.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.double, "uniform", 0, 1000)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 2, 2]), np.int32)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_double_withint64id():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_8.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_8.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_8.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_8.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.double, "uniform", 0, 1000)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 2, 2]), np.int64)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int64)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_float16():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_9.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_9.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_9.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_9.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.float16, "randint", 0, 10)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 2]), np.int32)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_float16_withint64id():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_10.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_10.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_10.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_10.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.float16, "randint", 0, 10)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 2]), np.int64)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int64)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_int8():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_11.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_11.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_11.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_11.txt"]
    np.random.seed(23457)
    shape_x1 = [5, 5, 5]
    a = gen_random_data_file(data_files[0], shape_x1, np.int8, "randint", -1000, 1000)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 0, 2, 1]), np.int32)
    c = gen_normal_data_file(data_files[2], tf.constant(5), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_int8_withint64id():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_12.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_12.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_12.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_12.txt"]
    np.random.seed(23457)
    shape_x1 = [5, 5, 5]
    a = gen_random_data_file(data_files[0], shape_x1, np.int8, "randint", -1000, 1000)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 0, 2, 1]), np.int64)
    c = gen_normal_data_file(data_files[2], tf.constant(5), np.int64)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_int64():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_13.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_13.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_13.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_13.txt"]
    np.random.seed(23457)
    shape_x1 = [10, 10]
    a = gen_random_data_file(data_files[0], shape_x1, np.int64, "randint", 0, 10)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.int32)
    c = gen_normal_data_file(data_files[2], tf.constant(10), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_int64_withint64id():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_14.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_14.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_14.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_14.txt"]
    np.random.seed(23457)
    shape_x1 = [10, 10]
    a = gen_random_data_file(data_files[0], shape_x1, np.int64, "randint", 0, 10)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.int64)
    c = gen_normal_data_file(data_files[2], tf.constant(10), np.int64)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_uint8():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_15.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_15.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_15.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_15.txt"]
    np.random.seed(23457)
    shape_x1 = [10, 11, 12]
    a = gen_random_data_file(data_files[0], shape_x1, np.uint8, "randint", 0, 10)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 2, 4, 5, 3, 1, 1, 1, 2]), np.int32)
    c = gen_normal_data_file(data_files[2], tf.constant(10), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_uint8_withint64id():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_16.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_16.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_16.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_16.txt"]
    np.random.seed(23457)
    shape_x1 = [10, 11, 12]
    a = gen_random_data_file(data_files[0], shape_x1, np.uint8, "randint", 0, 10)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 2, 4, 5, 3, 1, 1, 1, 2]), np.int64)
    c = gen_normal_data_file(data_files[2], tf.constant(10), np.int64)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_uint16():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_17.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_17.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_17.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_17.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.uint16, "randint", 0, 50)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 1]), np.int32)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_uint16_withint64id():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_18.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_18.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_18.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_18.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.uint16, "randint", 0, 50)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 1, 1]), np.int64)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int64)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_uint32():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_19.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_19.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_19.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_19.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.uint32, "randint", 0, 10)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 0, 0]), np.int32)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_uint32_withint64id():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_20.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_20.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_20.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_20.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.uint32, "randint", 0, 10)
    b = gen_normal_data_file(data_files[1], tf.constant([0, 0, 0]), np.int64)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int64)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_uint64():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_21.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_21.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_21.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_21.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.uint64, "randint", -200, 200)
    b = gen_normal_data_file(data_files[1], tf.constant([2, 0, 0]), np.int32)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_uint64_withint64id():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_22.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_22.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_22.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_22.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3]
    a = gen_random_data_file(data_files[0], shape_x1, np.uint64, "randint", -200, 200)
    b = gen_normal_data_file(data_files[1], tf.constant([2, 0, 0]), np.int64)
    c = gen_normal_data_file(data_files[2], tf.constant(3), np.int64)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_bigdata_int32():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_23.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_23.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_23.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_23.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 9, 1024]
    shape_x2 = [4]
    a = gen_random_data_file(data_files[0], shape_x1, np.int32, "randint", 0, 10)
    b = gen_random_data_file(data_files[1], shape_x2, np.int32, "randint", 0, 3)
    c = gen_normal_data_file(data_files[2], tf.constant(4), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")



def gen_random_bigdata_int32_withint64id():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_24.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_24.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_24.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_24.txt"]
    np.random.seed(23457)
    shape_x1 = [33, 1024]
    shape_x2 = [33]
    a = gen_random_data_file(data_files[0], shape_x1, np.int32, "randint", 0, 10)
    b = gen_random_data_file(data_files[1], shape_x2, np.int64, "randint", 0, 32)
    c = gen_normal_data_file(data_files[2], tf.constant(33), np.int64)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_bigdata_int16():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_25.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_25.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_25.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_25.txt"]
    np.random.seed(23457)
    shape_x1 = [50, 1500]
    shape_x2 = [50]
    a = gen_random_data_file(data_files[0], shape_x1, np.int16, "randint", 0, 10)
    b = gen_random_data_file(data_files[1], shape_x2, np.int32, "randint", 0, 49)
    c = gen_normal_data_file(data_files[2], tf.constant(50), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_bigdata_double():
    data_files = ["unsorted_segment_min/data/unsorted_segment_min_data_input1_26.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input2_26.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_input3_26.txt",
                  "unsorted_segment_min/data/unsorted_segment_min_data_output1_26.txt"]
    np.random.seed(23457)
    shape_x1 = [1024, 33]
    shape_x2 = [1024]
    a = gen_random_data_file(data_files[0], shape_x1, np.double, "randint", 0, 10)
    b = gen_random_data_file(data_files[1], shape_x2, np.int32, "randint", 0, 1023)
    c = gen_normal_data_file(data_files[2], tf.constant(1024), np.int32)

    re = tf.unsorted_segment_min(a, b, c)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def run():
    gen_random_data_int32()
    gen_random_data_int32()
    gen_random_data_int32_withint64id()
    gen_random_data_int16()
    gen_random_data_int16_withint64id()
    gen_random_data_float()
    gen_random_data_float_withint64id()
    gen_random_data_double()
    gen_random_data_double_withint64id()
    gen_random_data_float16()
    gen_random_data_float16_withint64id()
    gen_random_data_int8()
    gen_random_data_int8_withint64id()
    gen_random_data_int64()
    gen_random_data_int64_withint64id()
    gen_random_data_uint8()
    gen_random_data_uint8_withint64id()
    gen_random_data_uint16()
    gen_random_data_uint16_withint64id()
    gen_random_data_uint32()
    gen_random_data_uint32_withint64id()
    gen_random_data_uint64()
    gen_random_data_uint64_withint64id()
    gen_random_bigdata_int32()
    gen_random_bigdata_int32_withint64id()
    gen_random_bigdata_int16()
    gen_random_bigdata_double()
