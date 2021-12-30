"""
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config

def gen_random_data_float(data_files):
    np.random.seed(23457)
    shape_start = []
    shape_stop = []
    shape_num = []

    a = gen_data_file(data_files[0], shape_start, np.float, "randint", 0, 100)
    b = gen_data_file(data_files[1], shape_stop, np.float, "randint", 0, 100)
    c = gen_data_file(data_files[2], shape_num, np.int32, "randint", 0, 100)

    start = tf.compat.v1.placeholder(tf.float32, shape=shape_start)
    stop = tf.compat.v1.placeholder(tf.float32, shape=shape_stop)
    num = tf.compat.v1.placeholder(tf.int32, shape=shape_num)

    re = tf.linspace(start, stop, num)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={start:a, stop:b, num:c})

    write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_double(data_files):
    np.random.seed(23457)
    shape_start = []
    shape_stop = []
    shape_num = []

    a = gen_data_file(data_files[0], shape_start, np.double, "randint", 0, 100)
    b = gen_data_file(data_files[1], shape_stop, np.double, "randint", 0, 100)
    c = gen_data_file(data_files[2], shape_num, np.int64, "randint", 0, 100)

    start = tf.compat.v1.placeholder(tf.double, shape=shape_start)
    stop = tf.compat.v1.placeholder(tf.double, shape=shape_stop)
    num = tf.compat.v1.placeholder(tf.int64, shape=shape_num)

    re = tf.linspace(start, stop, num)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={start:a, stop:b, num:c})
    write_file_txt(data_files[3], data, fmt="%s")

def run():
    data_files1 = ["linspace/data/linspace_data_start1.txt",
                  "linspace/data/linspace_data_stop1.txt",
                  "linspace/data/linspace_data_num1.txt",
                  "linspace/data/linspace_data_output1.txt"]
    gen_random_data_float(data_files1)
    data_files2 = ["linspace/data/linspace_data_start2.txt",
                  "linspace/data/linspace_data_stop2.txt",
                  "linspace/data/linspace_data_num2.txt",
                  "linspace/data/linspace_data_output2.txt"]
    gen_random_data_double(data_files2)
