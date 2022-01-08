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
import random
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
    elif rand_type == "uniform":
        rand_data = np.random.uniform(low, high, size=shape)
    elif rand_type == "complex":
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

def gen_data_file2(data_file, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = random.randint(low, high)
    else:
        rand_data = random.uniform(low, high)
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

def gen_random_data_int8_1X():
    data_files=["reduce_sum/data/reduce_sum_data_input_int8_1X.txt",
                "reduce_sum/data/reduce_sum_data_axis_int8_1X.txt",
                "reduce_sum/data/reduce_sum_data_output_int8_1X.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.int8,"randint",-10,10)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.int8,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.reduce_sum(input_data, axis_data, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_int16_1X():
    data_files=["reduce_sum/data/reduce_sum_data_input_int16_1X.txt",
                "reduce_sum/data/reduce_sum_data_axis_int16_1X.txt",
                "reduce_sum/data/reduce_sum_data_output_int16_1X.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.int16,"randint",-100,100)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.int16,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.reduce_sum(input_data, axis_data, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_int32_1X():
    data_files=["reduce_sum/data/reduce_sum_data_input_int32_1X.txt",
                "reduce_sum/data/reduce_sum_data_axis_int32_1X.txt",
                "reduce_sum/data/reduce_sum_data_output_int32_1X.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.int32,"randint",-1000,1000)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.int32,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.reduce_sum(input_data, axis_data, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_int64_1X():
    data_files=["reduce_sum/data/reduce_sum_data_input_int64_1X.txt",
                "reduce_sum/data/reduce_sum_data_axis_int64_1X.txt",
                "reduce_sum/data/reduce_sum_data_output_int64_1X.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.int64,"randint",-10000,10000)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.int64,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.reduce_sum(input_data, axis_data, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_uint64_1X():
    data_files=["reduce_sum/data/reduce_sum_data_input_uint64_1X.txt",
                "reduce_sum/data/reduce_sum_data_axis_uint64_1X.txt",
                "reduce_sum/data/reduce_sum_data_output_uint64_1X.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.uint64,"randint",1,10000)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.uint64,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.reduce_sum(input_data, axis_data, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_float16_1X():
    data_files=["reduce_sum/data/reduce_sum_data_input_float16_1X.txt",
                "reduce_sum/data/reduce_sum_data_axis_float16_1X.txt",
                "reduce_sum/data/reduce_sum_data_output_float16_1X.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.float16,"randint",1,100)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.float16,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.reduce_sum(input_data, axis_data, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_float32_1X():
    data_files=["reduce_sum/data/reduce_sum_data_input_float32_1X.txt",
                "reduce_sum/data/reduce_sum_data_axis_float32_1X.txt",
                "reduce_sum/data/reduce_sum_data_output_float32_1X.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.float32,"randint",1,100)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.float32,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.reduce_sum(input_data, axis_data, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_float64_1X():
    data_files=["reduce_sum/data/reduce_sum_data_input_float64_1X.txt",
                "reduce_sum/data/reduce_sum_data_axis_float64_1X.txt",
                "reduce_sum/data/reduce_sum_data_output_float64_1X.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.float64,"randint",1,100)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.float64,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.reduce_sum(input_data, axis_data, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_int8_1X_ET():
    data_files=["reduce_sum/data/reduce_sum_data_input_int8_1X_ET.txt",
                "reduce_sum/data/reduce_sum_data_axis_int8_1X_ET.txt",
                "reduce_sum/data/reduce_sum_data_output_int8_1X_ET.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.int8,"randint",1,10)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.int8,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.reduce_sum(input_data, axis_data, True)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_int8_2X():
    data_files=["reduce_sum/data/reduce_sum_data_input_int8_2X.txt",
                "reduce_sum/data/reduce_sum_data_axis_int8_2X.txt",
                "reduce_sum/data/reduce_sum_data_output_int8_2X.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.int8,"randint",1,10)
    shape_axes_data = [2]
    b = gen_data_file(data_files[1],shape_axes_data,np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.int8,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32,shape=shape_axes_data)
    re = tf.reduce_sum(input_data, axis_data, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_int8_2X_ET():
    data_files=["reduce_sum/data/reduce_sum_data_input_int8_2X_ET.txt",
                "reduce_sum/data/reduce_sum_data_axis_int8_2X_ET.txt",
                "reduce_sum/data/reduce_sum_data_output_int8_2X_ET.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.int8,"randint",1,10)
    shape_axes_data = [2]
    b = gen_data_file(data_files[1],shape_axes_data,np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.int8,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32,shape=shape_axes_data)
    re = tf.reduce_sum(input_data, axis_data, True)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_complex64_1X():
    data_files=["reduce_sum/data/reduce_sum_data_input_complex64_1X.txt",
                "reduce_sum/data/reduce_sum_data_axis_complex64_1X.txt",
                "reduce_sum/data/reduce_sum_data_output_complex64_1X.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.complex64,"complex",1,10)
    b = gen_data_file2(data_files[1],np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.complex64,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32)
    re = tf.reduce_sum(input_data, axis_data, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_complex64_2X():
    data_files=["reduce_sum/data/reduce_sum_data_input_complex64_2X.txt",
                "reduce_sum/data/reduce_sum_data_axis_complex64_2X.txt",
                "reduce_sum/data/reduce_sum_data_output_complex64_2X.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5]
    a = gen_data_file(data_files[0],shape_input_data,np.complex64,"complex",1,10)
    shape_axes_data = [2]
    b = gen_data_file(data_files[1],shape_axes_data,np.int32,"randint",-3,2)
    input_data = tf.compat.v1.placeholder(tf.complex64,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32,shape=shape_axes_data)
    re = tf.reduce_sum(input_data, axis_data, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def gen_random_data_big_2X():
    data_files=["reduce_sum/data/reduce_sum_data_input_big_2X.txt",
                "reduce_sum/data/reduce_sum_data_axis_big_2X.txt",
                "reduce_sum/data/reduce_sum_data_output_big_2X.txt"]
    np.random.seed(3457)
    shape_input_data = [3,4,5,6,7,8]
    a = gen_data_file(data_files[0],shape_input_data,np.uint32,"randint",1,100)
    shape_axes_data = [2]
    b = gen_data_file(data_files[1],shape_axes_data,np.int32,"randint",-6,5)
    input_data = tf.compat.v1.placeholder(tf.uint32,shape=shape_input_data)
    axis_data = tf.compat.v1.placeholder(tf.int32,shape=shape_axes_data)
    re = tf.reduce_sum(input_data, axis_data, False)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input_data:a, axis_data:b})
    write_file_txt(data_files[2],data,fmt="%s")

def run():
    gen_random_data_int8_1X()
    gen_random_data_int16_1X()
    gen_random_data_int32_1X()
    gen_random_data_int64_1X()
    gen_random_data_uint64_1X()
    gen_random_data_float16_1X()
    gen_random_data_float32_1X()
    gen_random_data_float64_1X()
    gen_random_data_int8_1X_ET()
    gen_random_data_int8_2X()
    gen_random_data_int8_2X_ET()
    gen_random_data_complex64_1X()
    gen_random_data_complex64_2X()
    gen_random_data_big_2X()

