"""
Copyright 2020 Huawei Technologies Co., Ltd

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

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config

def gen_random_data_int32():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_1.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_1.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_1.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_1.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_1.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_1.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3,3]
    shape_x2 = [2]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.int32, "randint", 0, 10)

    b=np.array([-1,1], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([3], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([3], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")

    e=np.array([0], dtype=np.int32)
    write_file_txt(data_files[4], e, fmt="%s")

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "RIGHT_LEFT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")
    #print(data)
    
    
def gen_random_data_float():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_2.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_2.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_2.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_2.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_2.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_2.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3,3]
    shape_x2 = [2]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 10)

    b=np.array([-1,1], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([3], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([3], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")
    
    e = gen_data_file(data_files[4], shape_x3, np.float32, "uniform", 0, 10)

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.float32, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "RIGHT_LEFT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")
    
def gen_random_data_double():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_3.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_3.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_3.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_3.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_3.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3,3]
    shape_x2 = [2]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 10)

    b=np.array([-1,1], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([3], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([3], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")

    e = gen_data_file(data_files[4], shape_x3, np.float64, "uniform", 0, 10)

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.float64, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.float64, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "RIGHT_LEFT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")
    
    
def gen_random_data_double_32k():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_13.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_13.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_13.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_13.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_13.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_13.txt"]
    np.random.seed(23457)
    shape_x1 = [8,8,8,8]
    shape_x2 = [2]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 10)

    b=np.array([-3,4], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([8], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([8], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")

    e = gen_data_file(data_files[4], shape_x3, np.float64, "uniform", 0, 10)

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.float64, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.float64, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "RIGHT_LEFT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")

def gen_random_data_int64():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_4.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_4.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_4.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_4.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_4.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_4.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3,3]
    shape_x2 = [2]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.int64, "randint", 0, 10)

    b=np.array([-1,1], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([3], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([3], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")

    e = gen_data_file(data_files[4], shape_x3, np.int64, "randint", 0, 10)

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.int64, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.int64, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "RIGHT_LEFT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")

def gen_random_data_complex64():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_5.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_5.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_5.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_5.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_5.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_5.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3,3]
    shape_x2 = [2]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.complex64, "uniform", 0, 10)

    b=np.array([-1,1], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([3], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([3], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")

    e = gen_data_file(data_files[4], shape_x3, np.complex64, "uniform", 0, 10)

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.complex64, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.complex64, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "RIGHT_LEFT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")
    
def gen_random_data_float16():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_6.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_6.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_6.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_6.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_6.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_6.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3,3]
    shape_x2 = [2]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.float16, "uniform", 0, 10)

    b=np.array([-1,1], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([3], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([3], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")

    e = gen_data_file(data_files[4], shape_x3, np.float16, "uniform", 0, 10)

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.float16, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.float16, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "LEFT_RIGHT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")
    
def gen_random_data_uint8():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_7.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_7.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_7.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_7.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_7.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_7.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3,4]
    shape_x2 = [1]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.uint8, "randint", 0, 10)

    b=np.array([-1], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([5], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([5], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")

    e = gen_data_file(data_files[4], shape_x3, np.uint8, "randint", 0, 10)

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.uint8, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.uint8, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "LEFT_RIGHT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")
    
def gen_random_data_uint16():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_8.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_8.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_8.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_8.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_8.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_8.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3,4]
    shape_x2 = [1]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.uint16, "randint", 0, 10)

    b=np.array([-1], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([5], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([5], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")

    e = gen_data_file(data_files[4], shape_x3, np.uint16, "randint", 0, 10)

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.uint16, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.uint16, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "LEFT_RIGHT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")

def gen_random_data_uint32():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_9.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_9.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_9.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_9.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_9.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_9.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3,4]
    shape_x2 = [1]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.uint32, "randint", 0, 10)

    b=np.array([-1], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([5], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([5], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")

    e = gen_data_file(data_files[4], shape_x3, np.uint32, "randint", 0, 10)

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.uint32, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.uint32, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "RIGHT_RIGHT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")
    
def gen_random_data_uint64():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_10.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_10.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_10.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_10.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_10.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_10.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3,4]
    shape_x2 = [1]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.uint64, "randint", 0, 10)

    b=np.array([-1], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([5], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([5], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")

    e = gen_data_file(data_files[4], shape_x3, np.uint64, "randint", 0, 10)

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.uint64, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.uint64, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "LEFT_LEFT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")

def gen_random_data_complex128():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_11.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_11.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_11.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_11.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_11.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_11.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3,3]
    shape_x2 = [2]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.complex128, "uniform", 0, 10)

    b=np.array([-1,1], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([3], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([3], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")

    e = gen_data_file(data_files[4], shape_x3, np.complex128, "uniform", 0, 10)

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.complex128, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.complex128, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "RIGHT_LEFT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")
    
def gen_random_data_int8():
    data_files=["matrix_diag_v3/data/matrix_diag_v3_data_input1_12.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input2_12.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input3_12.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input4_12.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_input5_12.txt",
                "matrix_diag_v3/data/matrix_diag_v3_data_output1_12.txt"]
    np.random.seed(23457)
    shape_x1 = [2,3,4]
    shape_x2 = [1]
    shape_x3 = [1]


    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", 0, 10)

    b=np.array([-1], dtype=np.int32)
    write_file_txt(data_files[1], b, fmt="%s")

    c=np.array([5], dtype=np.int32)
    write_file_txt(data_files[2], c, fmt="%s")

    d=np.array([5], dtype=np.int32)
    write_file_txt(data_files[3], d, fmt="%s")

    e = gen_data_file(data_files[4], shape_x3, np.int8, "randint", 0, 10)

    #tf.compat.v1.disable_eager_execution()
    #diagonal = tf.compat.v1.placeholder(tf.int8, shape=shape_x1)
    #k = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    #num_rows = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #num_cols = tf.compat.v1.placeholder(tf.int32, shape=shape_x3)
    #padding_value = tf.compat.v1.placeholder(tf.int8, shape=shape_x3)
    #re = tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows[0], num_cols=num_cols[0], padding_value=padding_value[0], align = "LEFT_LEFT")
    #with tf.compat.v1.Session(config=config('cpu')) as session:
    #    data = session.run(re, feed_dict={diagonal:a, k:b, num_rows:c, num_cols:d, padding_value:e})
    #write_file_txt(data_files[5], data, fmt="%s")

def run():
    gen_random_data_int32()
    gen_random_data_float()
    gen_random_data_double()
    gen_random_data_double_32k()
    gen_random_data_int64()
    gen_random_data_complex64()
    gen_random_data_float16()
    gen_random_data_uint8()
    gen_random_data_uint16()
    gen_random_data_uint32()
    gen_random_data_uint64()
    gen_random_data_complex128()
    gen_random_data_int8()
