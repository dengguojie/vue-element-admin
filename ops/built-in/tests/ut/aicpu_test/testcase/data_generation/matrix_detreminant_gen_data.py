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


def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    # prama1: data_file: the file which store the generation data
    # param2: shape: data shape
    # param3: dtype: data type
    # param4: rand_type: the method of generate data, select from "randint, uniform"
    # param5: data lower limit
    # param6: data upper limit
    if rand_type == "randcomplex":
        real = np.random.uniform(low, high, size=shape)
        imag = np.random.uniform(low, high, size=shape)
        rand_data = np.complex64(real)
        rand_data.imag = imag
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    data = np.array(rand_data, dtype=dtype)
    write_file_txt(data_file, data, fmt="%s")
    return data


def gen_random_data_float():
    data_files = ["matrix_determinant/data/matrix_determinant_data_input1_1.txt",
                  "matrix_determinant/data/matrix_determinant_data_output1_1.txt"]
    np.random.seed(23457)
    N = 3
    M = 5
    shape_x = [N, M, M]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", -100, 100)
    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    with tf.compat.v1.Session() as session:
        numpy_x = x.eval({x: a})
    data = np.linalg.det(numpy_x)
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_double():
    data_files = ["matrix_determinant/data/matrix_determinant_data_input1_2.txt",
                  "matrix_determinant/data/matrix_determinant_data_output1_2.txt"]
    np.random.seed(23457)
    N = 3
    M = 5
    shape_x = [N, M, M]
    a = gen_data_file(data_files[0], shape_x, np.float64, "uniform", -100, 100)
    x = tf.compat.v1.placeholder(tf.float64, shape=shape_x)
    with tf.compat.v1.Session() as session:
        numpy_x = x.eval({x: a})
    data = np.linalg.det(numpy_x)
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_complex64():
    data_files = ["matrix_determinant/data/matrix_determinant_data_input1_3.txt",
                  "matrix_determinant/data/matrix_determinant_data_output1_3.txt"]
    np.random.seed(23457)
    N = 3
    M = 5
    shape_x = [N, M, M]
    a = gen_data_file(data_files[0], shape_x, np.complex64, "randcomplex", -100, 100)
    x = tf.compat.v1.placeholder(tf.complex64, shape=shape_x)
    with tf.compat.v1.Session() as session:
        numpy_x = x.eval({x: a})
    data = np.linalg.det(numpy_x)
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_complex128():
    data_files = ["matrix_determinant/data/matrix_determinant_data_input1_4.txt",
                  "matrix_determinant/data/matrix_determinant_data_output1_4.txt"]
    np.random.seed(23457)
    N = 3
    M = 5
    shape_x = [N, M, M]
    a = gen_data_file(data_files[0], shape_x, np.complex128, "randcomplex", -100, 100)
    x = tf.compat.v1.placeholder(tf.complex128, shape=shape_x)
    with tf.compat.v1.Session() as session:
        numpy_x = x.eval({x: a})
    data = np.linalg.det(numpy_x)
    write_file_txt(data_files[1], data, fmt="%s")


def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config


def run():
    gen_random_data_float()
    gen_random_data_double()
    gen_random_data_complex64()
    gen_random_data_complex128()
