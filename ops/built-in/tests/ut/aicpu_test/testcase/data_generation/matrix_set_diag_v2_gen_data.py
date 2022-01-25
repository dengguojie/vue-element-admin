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
from tensorflow.python.ops import array_ops


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
    if dtype == np.bool:
        rand_data = rand_data > 0
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
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_1.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_1.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_1.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_1.txt"]
    shape_x = [6, 8, 8]
    shape_diag = [6, 8]
    a = gen_data_file(data_files[0], shape_x, np.int8, "randint", -100, 100)
    b = gen_data_file(data_files[1], shape_diag, np.int8, "randint", -100, 100)
    c = np.int32(0)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=0)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_int16():
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_2.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_2.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_2.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_2.txt"]
    shape_x = [6, 8, 8]
    shape_diag = [6, 8]
    a = gen_data_file(data_files[0], shape_x, np.int16, "randint", -1000, 1000)
    b = gen_data_file(data_files[1], shape_diag, np.int16, "randint", -1000, 1000)
    c = np.array([0, 0], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=(0, 0))
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_int32():
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_3.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_3.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_3.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_3.txt"]
    shape_x = [6, 8, 8]
    shape_diag = [6, 8]
    a = gen_data_file(data_files[0], shape_x, np.int32, "randint", -10000, 10000)
    b = gen_data_file(data_files[1], shape_diag, np.int32, "randint", -10000, 10000)
    c = np.int32(0)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=0)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_int64():
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_4.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_4.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_4.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_4.txt"]
    shape_x = [6, 8, 8]
    shape_diag = [6, 8]
    a = gen_data_file(data_files[0], shape_x, np.int64, "randint", -10000, 10000)
    b = gen_data_file(data_files[1], shape_diag, np.int64, "randint", -10000, 10000)
    c = np.array([0, 0], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=(0, 0))
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_uint8():
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_5.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_5.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_5.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_5.txt"]
    shape_x = [6, 7, 8]
    shape_diag = [6, 7]
    a = gen_data_file(data_files[0], shape_x, np.uint8, "randint", 0, 200)
    b = gen_data_file(data_files[1], shape_diag, np.uint8, "randint", 0, 200)
    c = np.int32(0)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=0)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_uint16():
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_6.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_6.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_6.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_6.txt"]
    shape_x = [6, 7, 8]
    shape_diag = [6, 7]
    a = gen_data_file(data_files[0], shape_x, np.uint16, "randint", 0, 2000)
    b = gen_data_file(data_files[1], shape_diag, np.uint16, "randint", 0, 2000)
    c = np.array([0, 0], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=(0, 0))
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_uint32():
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_7.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_7.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_7.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_7.txt"]
    shape_x = [6, 8, 7]
    shape_diag = [6, 7]
    a = gen_data_file(data_files[0], shape_x, np.uint32, "randint", 0, 20000)
    b = gen_data_file(data_files[1], shape_diag, np.uint32, "randint", 0, 20000)
    c = np.int32(0)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=0)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_uint64():
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_8.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_8.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_8.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_8.txt"]
    shape_x = [6, 8, 7]
    shape_diag = [6, 7]
    a = gen_data_file(data_files[0], shape_x, np.uint64, "randint", 0, 20000)
    b = gen_data_file(data_files[1], shape_diag, np.uint64, "randint", 0, 20000)
    c = np.array([0, 0], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=(0, 0))
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_float16():
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_9.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_9.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_9.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_9.txt"]
    shape_x = [6, 8, 8]
    shape_diag = [6, 8]
    a = gen_data_file(data_files[0], shape_x, np.float16, "uniform", 0, 2000)
    b = gen_data_file(data_files[1], shape_diag, np.float16, "uniform", 0, 2000)
    c = np.array([0, 0], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=(0, 0))
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_float():
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_10.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_10.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_10.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_10.txt"]
    shape_x = [6, 8, 8]
    shape_diag = [6, 8]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", -10000, 10000)
    b = gen_data_file(data_files[1], shape_diag, np.float32, "uniform", -10000, 10000)
    c = np.array([0, 0], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=(0, 0))
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_double():
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_11.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_11.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_11.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_11.txt"]
    shape_x = [64, 64, 64]
    shape_diag = [64, 64]
    a = gen_data_file(data_files[0], shape_x, np.double, "uniform", -10000, 10000)
    b = gen_data_file(data_files[1], shape_diag, np.double, "uniform", -10000, 10000)
    c = np.array([0, 0], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=(0, 0))
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_complex64():
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_12.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_12.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_12.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_12.txt"]
    shape_x = [6, 8, 8]
    shape_diag = [6, 8]
    a = gen_data_file(data_files[0], shape_x, np.complex64, "randint", -10000, 10000)
    b = gen_data_file(data_files[1], shape_diag, np.complex64, "randint", -10000, 10000)
    c = np.int32(0)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=0)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_complex128():
    data_files = ["matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_13.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_13.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_13.txt",
                  "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_13.txt"]
    shape_x = [6, 8, 8]
    shape_diag = [6, 8]
    a = gen_data_file(data_files[0], shape_x, np.complex128, "randint", -10000, 10000)
    b = gen_data_file(data_files[1], shape_diag, np.complex128, "randint", -10000, 10000)
    c = np.int32(0)
    write_file_txt(data_files[2], c, fmt="%s")

    re = tf.linalg.set_diag(a, b, k=0)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re)
    write_file_txt(data_files[3], data, fmt="%s")


def run():
    gen_random_data_int8()
    gen_random_data_int16()
    gen_random_data_int32()
    gen_random_data_int64()
    gen_random_data_uint8()
    gen_random_data_uint16()
    gen_random_data_uint32()
    gen_random_data_uint64()
    gen_random_data_float16()
    gen_random_data_float()
    gen_random_data_double()
    gen_random_data_complex64()
    gen_random_data_complex128()
