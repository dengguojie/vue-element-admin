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
    data_files=["greater_equal/data/greater_equal_data_input1_1.txt",
                "greater_equal/data/greater_equal_data_input2_1.txt",
                "greater_equal/data/greater_equal_data_output1_1.txt"]
    np.random.seed(23333)
    shape_x1 = [6, 12]
    shape_x2 = [12]
    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", np.iinfo(np.int8).min, np.iinfo(np.int8).max) 
    b = gen_data_file(data_files[1], shape_x2, np.int8, "randint", np.iinfo(np.int8).min, np.iinfo(np.int8).max)
    
    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int8, shape=shape_x2)
    
    re = tf.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_int16():
    data_files=["greater_equal/data/greater_equal_data_input1_2.txt",
                "greater_equal/data/greater_equal_data_input2_2.txt",
                "greater_equal/data/greater_equal_data_output1_2.txt"]
    np.random.seed(23333)
    shape_x1 = [12, 6]
    shape_x2 = [12, 6]
    a = gen_data_file(data_files[0], shape_x1, np.int16, "randint", np.iinfo(np.int16).min, np.iinfo(np.int16).max) 
    b = gen_data_file(data_files[1], shape_x2, np.int16, "randint", np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    
    x1 = tf.compat.v1.placeholder(tf.int16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int16, shape=shape_x2)
    
    re = tf.math.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")
    
def gen_random_data_int32():
    data_files=["greater_equal/data/greater_equal_data_input1_3.txt",
                "greater_equal/data/greater_equal_data_input2_3.txt",
                "greater_equal/data/greater_equal_data_output1_3.txt"]   
    np.random.seed(23333)
    shape_x1 = [3, 12]
    shape_x2 = [12]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "randint", np.iinfo(np.int32).min, np.iinfo(np.int32).max) 
    b = gen_data_file(data_files[1], shape_x2, np.int32, "randint", np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    
    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    
    re = tf.math.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_int64():
    data_files=["greater_equal/data/greater_equal_data_input1_4.txt",
                "greater_equal/data/greater_equal_data_input2_4.txt",
                "greater_equal/data/greater_equal_data_output1_4.txt"]
    np.random.seed(23333)
    shape_x1 = [11, 10, 4]
    shape_x2 = [11, 10, 4]
    a = gen_data_file(data_files[0], shape_x1, np.int64, "randint", np.iinfo(np.int64).min, np.iinfo(np.int64).max) 
    b = gen_data_file(data_files[1], shape_x2, np.int64, "randint", np.iinfo(np.int64).min, np.iinfo(np.int64).max)
    
    x1 = tf.compat.v1.placeholder(tf.int64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int64, shape=shape_x2)
    
    re = tf.math.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_uint8():
    data_files=["greater_equal/data/greater_equal_data_input1_5.txt",
                "greater_equal/data/greater_equal_data_input2_5.txt",
                "greater_equal/data/greater_equal_data_output1_5.txt"]
    np.random.seed(23333)
    shape_x1 = [6, 12]
    shape_x2 = [12]
    a = gen_data_file(data_files[0], shape_x1, np.uint8, "randint", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.uint8, "randint", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.uint8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.uint8, shape=shape_x2)
    re = tf.math.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

#   device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, 
#   DT_INT8, DT_INT64, DT_BFLOAT16, DT_HALF, DT_UINT32, DT_UINT64]
#   "DT_INT16" is not in the support list
def gen_random_data_uint16():
    data_files=["greater_equal/data/greater_equal_data_input1_6.txt",
                "greater_equal/data/greater_equal_data_input2_6.txt",
                "greater_equal/data/greater_equal_data_output1_6.txt"]
    np.random.seed(23333)
    shape_x1 = [6, 12]
    shape_x2 = [12]
    a = gen_data_file(data_files[0], shape_x1, np.uint32, "randint", np.iinfo(np.uint8).max+1, np.iinfo(np.uint8).max+200)
    b = gen_data_file(data_files[1], shape_x2, np.uint32, "randint", np.iinfo(np.uint8).max+1, np.iinfo(np.uint8).max+200)

    x1 = tf.compat.v1.placeholder(tf.uint32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.uint32, shape=shape_x2)
    re = tf.math.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_uint32():
    data_files=["greater_equal/data/greater_equal_data_input1_7.txt",
                "greater_equal/data/greater_equal_data_input2_7.txt",
                "greater_equal/data/greater_equal_data_output1_7.txt"]
    np.random.seed(23333)
    shape_x1 = [7, 12, 10]
    shape_x2 = [12, 10]
    a = gen_data_file(data_files[0], shape_x1, np.uint32, "randint", np.iinfo(np.uint16).max+1, np.iinfo(np.uint16).max+200)
    b = gen_data_file(data_files[1], shape_x2, np.uint32, "randint", np.iinfo(np.uint16).max+1, np.iinfo(np.uint16).max+200)

    x1 = tf.compat.v1.placeholder(tf.uint32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.uint32, shape=shape_x2)
    re = tf.math.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_uint64():
    data_files=["greater_equal/data/greater_equal_data_input1_8.txt",
                "greater_equal/data/greater_equal_data_input2_8.txt",
                "greater_equal/data/greater_equal_data_output1_8.txt"]
    np.random.seed(23333)
    shape_x1 = [12, 10]
    shape_x2 = [7, 12, 10]
    a = gen_data_file(data_files[0], shape_x1, np.uint64, "randint", np.iinfo(np.uint32).max+1, np.iinfo(np.uint32).max+200)
    b = gen_data_file(data_files[1], shape_x2, np.uint64, "randint", np.iinfo(np.uint32).max+1, np.iinfo(np.uint32).max+200)

    x1 = tf.compat.v1.placeholder(tf.uint64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.uint64, shape=shape_x2)
    re = tf.math.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float16():
    data_files=["greater_equal/data/greater_equal_data_input1_9.txt",
                "greater_equal/data/greater_equal_data_input2_9.txt",
                "greater_equal/data/greater_equal_data_output1_9.txt"]
    np.random.seed(23333)
    shape_x1 = [15, 12, 30]
    shape_x2 = [15, 12, 30]
    a = gen_data_file(data_files[0], shape_x1, np.float16, "uniform", -100, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float16, "uniform", -100, 100)

    x1 = tf.compat.v1.placeholder(tf.float16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float16, shape=shape_x2)
    re = tf.math.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float():
    data_files=["greater_equal/data/greater_equal_data_input1_10.txt",
                "greater_equal/data/greater_equal_data_input2_10.txt",
                "greater_equal/data/greater_equal_data_output1_10.txt"]
    np.random.seed(66666)
    shape_x1 = [15, 12, 30]
    shape_x2 = [12, 30]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", -100, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", -100, 100)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.math.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_double():
    data_files=["greater_equal/data/greater_equal_data_input1_11.txt",
                "greater_equal/data/greater_equal_data_input2_11.txt",
                "greater_equal/data/greater_equal_data_output1_11.txt"]
    np.random.seed(55555)
    shape_x1 = [7, 12, 30]
    shape_x2 = [30]
    a = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", -100, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", -100, 100)

    x1 = tf.compat.v1.placeholder(tf.float64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float64, shape=shape_x2)
    re = tf.math.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_x_one_elem():
    data_files=["greater_equal/data/greater_equal_data_input1_x_one_elem.txt",
                "greater_equal/data/greater_equal_data_input2_x_one_elem.txt",
                "greater_equal/data/greater_equal_data_output1_x_one_elem.txt"]
    np.random.seed(666)
    shape_x1 = [1]
    shape_x2 = [1024, 1024]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "uniform", -10, 10)
    b = gen_data_file(data_files[1], shape_x2, np.int32, "uniform", -10, 10)

    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    re = tf.math.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_y_one_elem():
    data_files=["greater_equal/data/greater_equal_data_input1_y_one_elem.txt",
                "greater_equal/data/greater_equal_data_input2_y_one_elem.txt",
                "greater_equal/data/greater_equal_data_output1_y_one_elem.txt"]
    np.random.seed(666)
    shape_x1 = [1024, 1024]
    shape_x2 = [1]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "uniform", -10, 10)
    b = gen_data_file(data_files[1], shape_x2, np.int32, "uniform", -10, 10)

    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    re = tf.math.greater_equal(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
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
    gen_random_data_float16()
    gen_random_data_float()
    gen_random_data_double()
    gen_random_data_x_one_elem()
    gen_random_data_y_one_elem()
