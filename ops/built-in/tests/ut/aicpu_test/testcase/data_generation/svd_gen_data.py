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

"""
prama1: file_name: the file which store the data
param2: data: data which will be stored
param3: fmt: format
"""
def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')

"""
prama1: file_name: the file which store the data
param2: dtype: data type
param3: delim: delimiter which is used to split data
"""
def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()

# prama1: file_name: the file which store the data
# param2: delim: delimiter which is used to split data
def read_file_txt(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)

"""
prama1: data_file: the file which store the generation data
param2: shape: data shape
param3: dtype: data type
param4: rand_type: the method of generate data, select from "randint, uniform"
param5: data lower limit
param6: data upper limit
"""
def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    elif rand_type=="uniform":
        rand_data = np.random.uniform(low, high, size=shape)
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

def gen_random_data_2d(data_files):
    np.random.seed(23457)
    shape_x = [3, 4]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", -100, 100)

    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    re = tf.linalg.svd(x, full_matrices = False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data[0], fmt="%s")
    write_file_txt(data_files[2], data[1], fmt="%s")
    write_file_txt(data_files[3], data[2], fmt="%s")

def gen_random_data_3d(data_files):
    np.random.seed(23457)
    shape_x = [2, 3, 4]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", -100, 100)

    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    re = tf.linalg.svd(x, full_matrices = False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data[0], fmt="%s")
    write_file_txt(data_files[2], data[1], fmt="%s")
    write_file_txt(data_files[3], data[2], fmt="%s")

def gen_random_data_float(data_files):
    np.random.seed(23457)
    shape_x = [2, 3, 4]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", -100, 100)

    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    re = tf.linalg.svd(x, full_matrices = False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data[0], fmt="%s")
    write_file_txt(data_files[2], data[1], fmt="%s")
    write_file_txt(data_files[3], data[2], fmt="%s")

def gen_random_data_double(data_files):
    np.random.seed(23457)
    shape_x = [4, 4, 16]
    a = gen_data_file(data_files[0], shape_x, np.double, "uniform", -100, 100)

    x = tf.compat.v1.placeholder(tf.double, shape=shape_x)
    re = tf.linalg.svd(x, full_matrices = False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data[0], fmt="%s")
    write_file_txt(data_files[2], data[1], fmt="%s")
    write_file_txt(data_files[3], data[2], fmt="%s")

def gen_random_data_float_full(data_files):
    np.random.seed(23457)
    shape_x = [2, 4, 3]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", -100, 100)

    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    re = tf.linalg.svd(x, full_matrices = True)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data[0], fmt="%s")
    write_file_txt(data_files[2], data[1], fmt="%s")
    write_file_txt(data_files[3], data[2], fmt="%s")

def gen_random_data_2d_only_s(data_files):
    np.random.seed(23457)
    shape_x = [3, 4]
    a = gen_data_file(data_files[0], shape_x, np.float32, "uniform", -100, 100)

    x = tf.compat.v1.placeholder(tf.float32, shape=shape_x)
    re = tf.linalg.svd(x, full_matrices = False, compute_uv = False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_double_16K(data_files):
    np.random.seed(23457)
    shape_x = [2, 32, 32]
    a = gen_data_file(data_files[0], shape_x, np.double, "uniform", -100, 100)

    x = tf.compat.v1.placeholder(tf.double, shape=shape_x)
    re = tf.linalg.svd(x, full_matrices = False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data[0], fmt="%s")
    write_file_txt(data_files[2], data[1], fmt="%s")
    write_file_txt(data_files[3], data[2], fmt="%s")

def gen_random_data_double_full_32K(data_files):
    np.random.seed(23457)
    shape_x = [2, 32, 64]
    a = gen_data_file(data_files[0], shape_x, np.double, "uniform", -100, 100)

    x = tf.compat.v1.placeholder(tf.double, shape=shape_x)
    re = tf.linalg.svd(x, full_matrices = True)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x:a})
    write_file_txt(data_files[1], data[0], fmt="%s")
    write_file_txt(data_files[2], data[1], fmt="%s")
    write_file_txt(data_files[3], data[2], fmt="%s")

def run():
    data_files=["svd/data/svd_data_input1_1.txt",
                "svd/data/svd_data_output1_1.txt",
                "svd/data/svd_data_output2_1.txt",
                "svd/data/svd_data_output3_1.txt",
                "svd/data/svd_data_input1_2.txt",
                "svd/data/svd_data_output1_2.txt",
                "svd/data/svd_data_output2_2.txt",
                "svd/data/svd_data_output3_2.txt", 
                "svd/data/svd_data_input1_3.txt",
                "svd/data/svd_data_output1_3.txt",
                "svd/data/svd_data_output2_3.txt",
                "svd/data/svd_data_output3_3.txt",
                "svd/data/svd_data_input1_4.txt",
                "svd/data/svd_data_output1_4.txt",
                "svd/data/svd_data_output2_4.txt",
                "svd/data/svd_data_output3_4.txt",
                "svd/data/svd_data_input1_5.txt",
                "svd/data/svd_data_output1_5.txt",
                "svd/data/svd_data_output2_5.txt",
                "svd/data/svd_data_output3_5.txt",
                "svd/data/svd_data_input1_6.txt",
                "svd/data/svd_data_output1_6.txt",
                "svd/data/svd_data_input1_7.txt",
                "svd/data/svd_data_output1_7.txt",
                "svd/data/svd_data_output2_7.txt",
                "svd/data/svd_data_output3_7.txt",
                "svd/data/svd_data_input1_8.txt",
                "svd/data/svd_data_output1_8.txt",
                "svd/data/svd_data_output2_8.txt",
                "svd/data/svd_data_output3_8.txt",]
    gen_random_data_2d([data_files[0], data_files[1], data_files[2], data_files[3]])
    gen_random_data_3d([data_files[4], data_files[5], data_files[6], data_files[7]])

    gen_random_data_float([data_files[8], data_files[9], data_files[10], data_files[11]])
    gen_random_data_double([data_files[12], data_files[13], data_files[14], data_files[15]])
    gen_random_data_float_full([data_files[16], data_files[17], data_files[18], data_files[19]])
    gen_random_data_2d_only_s([data_files[20], data_files[21]])
    gen_random_data_double_16K([data_files[22], data_files[23], data_files[24], data_files[25]])
    gen_random_data_double_full_32K([data_files[26], data_files[27], data_files[28], data_files[29]])