"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.       \
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
    data = data.flatten()
    np.savetxt(file_name, data, fmt=fmt, delimiter=' ', newline='\n')


def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    if rand_type == "randcomplex":
        rand_data_real = np.random.uniform(low, high, size=shape)
        rand_data_imag = np.random.uniform(low, high, size=shape)
        rand_data = []
        for i in range(len(rand_data_real)):
            rand_data.append(rand_data_real[i] + rand_data_imag[i] * 1j)
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    data = np.array(rand_data, dtype=dtype)
    write_file_txt(data_file, data, fmt="%s")
    return data


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


def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config


def gen_random_data_float():
    data_files = ["matrix_triangular_solve/data/matrix_triangular_solve_data_input1_1.txt",
                  "matrix_triangular_solve/data/matrix_triangular_solve_data_input2_1.txt",
                  "matrix_triangular_solve/data/matrix_triangular_solve_data_output1_1.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 2, 3, 3]
    shape_x2 = [2, 2, 3, 2]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", -1000, 1000)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", -1000, 1000)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.linalg.triangular_solve(x1, x2, True, True)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_double():
    data_files = ["matrix_triangular_solve/data/matrix_triangular_solve_data_input1_2.txt",
                  "matrix_triangular_solve/data/matrix_triangular_solve_data_input2_2.txt",
                  "matrix_triangular_solve/data/matrix_triangular_solve_data_output1_2.txt"]
    np.random.seed(23457)
    shape_x1 = [5, 5]
    shape_x2 = [5, 2]
    a = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.double, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.double, shape=shape_x2)
    re = tf.linalg.triangular_solve(x1, x2, False, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_complex64():
    data_files = ["matrix_triangular_solve/data/matrix_triangular_solve_data_input1_3.txt",
                  "matrix_triangular_solve/data/matrix_triangular_solve_data_input2_3.txt",
                  "matrix_triangular_solve/data/matrix_triangular_solve_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x1 = [10, 10, 10]
    shape_x2 = [10, 10, 10]
    a = gen_data_file(data_files[0], shape_x1, np.complex64, "randcomplex", -100, 0)
    b = gen_data_file(data_files[1], shape_x2, np.complex64, "randcomplex", -100, 0)

    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x2)
    re = tf.linalg.triangular_solve(x1, x2, False, True)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")
    data = data[0]


def gen_random_data_complex128():
    data_files = ["matrix_triangular_solve/data/matrix_triangular_solve_data_input1_4.txt",
                  "matrix_triangular_solve/data/matrix_triangular_solve_data_input2_4.txt",
                  "matrix_triangular_solve/data/matrix_triangular_solve_data_output1_4.txt"]
    np.random.seed(23457)
    shape_x1 = [1, 10, 10]
    shape_x2 = [1, 10, 1]

    a = gen_data_file(data_files[0], shape_x1, np.complex128, "randcomplex", -100, 100)
    b = gen_data_file(data_files[1], shape_x2, np.complex128, "randcomplex", -100, 100)

    x1 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x2)
    re = tf.linalg.triangular_solve(x1, x2, True, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def run():
    gen_random_data_float()
    gen_random_data_double()
    gen_random_data_complex64()
    gen_random_data_complex128()
