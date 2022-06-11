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

def gen_complex_data_file(data_file, shape, dtype, low, high):
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


def gen_random_data_double():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_0.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_0.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.double, "uniform", -100, 100)
    perm = [0, 2, 1]

    x1 = tf.compat.v1.placeholder(tf.double, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_uint8():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_1.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_1.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.uint8, "randint", 0, 20)
    perm = [0, 2, 1]

    x1 = tf.compat.v1.placeholder(tf.uint8, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_uint16():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_2.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_2.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.uint16, "randint", 0, 20)
    perm = [0, 2, 1]

    x1 = tf.compat.v1.placeholder(tf.uint16, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_int8():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_5.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_5.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.int8, "randint", 0, 20)
    perm = [0, 2, 1]

    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_int16():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_6.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_6.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.int16, "randint", 0, 20)
    perm = [0, 2, 1]

    x1 = tf.compat.v1.placeholder(tf.int16, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_int32():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_7.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_7.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.int32, "randint", 0, 20)
    perm = [0, 2, 1]

    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_int64():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_8.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_8.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.int64, "randint", 0, 20)
    perm = [0, 2, 1]

    x1 = tf.compat.v1.placeholder(tf.int64, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_float():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_9.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_9.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 20)
    perm = [0, 2, 1]

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_float16():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_10.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_10.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.float16, "uniform", 0, 20)
    perm = [0, 2, 1]

    x1 = tf.compat.v1.placeholder(tf.float16, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex64():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_11.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_11.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.complex64, "uniform", -100, 100)
    perm = [0, 2, 1]

    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex128():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_12.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_12.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.complex128, "uniform",  -100, 100)
    perm = [0, 2, 1]

    x1 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_int8_4D():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_13.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_13.txt"]
    np.random.seed(23457)
    shape_x = [2, 3, 4, 5]
    a = gen_data_file(data_files[0], shape_x, np.int8, "randint", 0, 20)
    perm = [0, 1, 3, 2]

    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_int8_5D():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_14.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_14.txt"]
    np.random.seed(23457)
    shape_x = [2, 3, 4, 5, 6]
    a = gen_data_file(data_files[0], shape_x, np.int8, "randint", 0, 20)
    perm = [0, 1, 3, 2, 4]

    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_int8_6D():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_15.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_15.txt"]
    np.random.seed(23457)
    shape_x = [2, 3, 4, 5, 6, 7]
    a = gen_data_file(data_files[0], shape_x, np.int8, "randint", 0, 20)
    perm = [0, 1, 3, 2, 4, 5]

    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_int8_7D():
    data_files=["conjugate_transpose/data/conjugate_transpose_data_input_16.txt",
                "conjugate_transpose/data/conjugate_transpose_data_output_16.txt"]
    np.random.seed(23457)
    shape_x = [2, 3, 4, 5, 6, 7, 8]
    a = gen_data_file(data_files[0], shape_x, np.int8, "randint", 0, 20)
    perm = [0, 1, 3, 2, 4, 5, 6]

    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex64_2D():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_17.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_17.txt"]
    np.random.seed(23457)
    shape_x = [2, 2]
    a = gen_data_file(data_files[0], shape_x, np.complex64, "uniform", -100, 100)
    perm = [0, 1]

    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_complex64_3D_large():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_19.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_19.txt"]
    np.random.seed(23457)
    shape_x = [256, 2, 1024]
    a = gen_data_file(data_files[0], shape_x, np.complex64, "uniform", -100, 100)
    perm = [0, 1, 2]
    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex128_3D_large():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_20.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_20.txt"]
    np.random.seed(23457)
    shape_x = [256, 2, 1024]
    a = gen_data_file(data_files[0], shape_x, np.complex128, "uniform", -100, 100)
    perm = [2, 1, 0]
    x1 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex64_3D():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_21.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_21.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.complex64, "uniform", -100, 100)
    perm = [0, 2, 1]
    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex128_3D():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_22.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_22.txt"]
    np.random.seed(23457)
    shape_x = [3, 2, 3]
    a = gen_data_file(data_files[0], shape_x, np.complex128, "uniform", -100, 100)
    perm = [0, 2, 1]
    x1 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex64_4D():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_23.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_23.txt"]
    np.random.seed(23457)
    shape_x = [2, 3, 4, 5]
    a = gen_data_file(data_files[0], shape_x, np.complex64, "uniform", -100, 100)
    perm = [0, 1, 3, 2]
    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex128_4D():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_24.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_24.txt"]
    np.random.seed(23457)
    shape_x = [2, 3, 4, 5]
    a = gen_data_file(data_files[0], shape_x, np.complex128, "uniform", -100, 100)
    perm = [0, 1, 3, 2]
    x1 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex64_5D():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_25.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_25.txt"]
    np.random.seed(23457)
    shape_x = [2, 3, 4, 5, 6]
    a = gen_data_file(data_files[0], shape_x, np.complex64, "uniform", -100, 100)
    perm = [0, 1, 3, 2, 4]
    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex128_5D():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_26.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_26.txt"]
    np.random.seed(23457)
    shape_x = [2, 3, 4, 5, 6]
    a = gen_data_file(data_files[0], shape_x, np.complex128, "uniform", -100, 100)
    perm = [0, 1, 3, 2, 4]
    x1 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex64_6D():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_27.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_27.txt"]
    np.random.seed(23457)
    shape_x = [2, 3, 4, 5, 6, 7]
    a = gen_data_file(data_files[0], shape_x, np.complex64, "uniform", -100, 100)
    perm = [0, 1, 3, 2, 4, 5]
    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex128_6D():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_28.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_28.txt"]
    np.random.seed(23457)
    shape_x = [2, 3, 4, 5, 6, 7]
    a = gen_data_file(data_files[0], shape_x, np.complex128, "uniform", -100, 100)
    perm = [0, 1, 3, 2, 4, 5]
    x1 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex64_7D():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_29.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_29.txt"]
    np.random.seed(23457)
    shape_x = [2, 3, 4, 5, 6, 7, 8]
    a = gen_data_file(data_files[0], shape_x, np.complex64, "uniform", -100, 100)
    perm = [0, 1, 3, 2, 4, 5, 6]
    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_complex128_7D():
    data_files = ["conjugate_transpose/data/conjugate_transpose_data_input_30.txt",
                  "conjugate_transpose/data/conjugate_transpose_data_output_30.txt"]
    np.random.seed(23457)
    shape_x = [2, 3, 4, 5, 6, 7, 8]
    a = gen_data_file(data_files[0], shape_x, np.complex128, "uniform", -100, 100)
    perm = [0, 1, 3, 2, 4, 5, 6]
    x1 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x)
    re = tf.raw_ops.ConjugateTranspose(x=x1, perm=perm)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[1], data, fmt="%s")

def run():
    gen_random_data_double()
    gen_random_data_uint8()
    gen_random_data_uint16()
    gen_random_data_int8()
    gen_random_data_int16()
    gen_random_data_int32()
    gen_random_data_int64()
    gen_random_data_float()
    gen_random_data_float16()
    gen_random_data_complex64()
    gen_random_data_complex128()
    gen_random_data_int8_4D()
    gen_random_data_int8_5D()
    gen_random_data_int8_6D()
    gen_random_data_int8_7D()
    gen_random_data_complex64_2D()
    gen_random_data_complex64_3D_large()
    gen_random_data_complex128_3D_large()
    gen_random_data_complex64_3D()
    gen_random_data_complex128_3D()
    gen_random_data_complex64_4D()
    gen_random_data_complex128_4D()
    gen_random_data_complex64_5D()
    gen_random_data_complex128_5D()
    gen_random_data_complex64_6D()
    gen_random_data_complex128_6D()
    gen_random_data_complex64_7D()
    gen_random_data_complex128_7D()
