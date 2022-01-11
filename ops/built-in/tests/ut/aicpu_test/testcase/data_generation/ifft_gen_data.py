"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

UnIFFT required by applicable law or agreed to in writing, software
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


def read_file_txt(file_name, delim=None):
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


def gen_data_file_2d(data_file, shape, dtype, low, high):
    """
    prama1: data_file: the file which store the generation data
    param2: shape: data shape
    param3: dtype: data type
    param4: data lower limit
    param5: data upper limit
    """
    r1=np.random.uniform(low,high,size=shape)
    r2=np.random.uniform(low,high,size=shape)
    rand_data=np.empty((shape[0],shape[1]),dtype=dtype)
    for i in range(0,shape[0]):
        for p in range(0,shape[1]):
            rand_data[i,p]=complex(r1[i,p],r2[i,p])
    data = np.array(rand_data, dtype=dtype)
    write_file_txt(data_file, data, fmt="%s")
    return data


def gen_data_file_3d(data_file, shape, dtype, low, high):
    """
    prama1: data_file: the file which store the generation data
    param2: shape: data shape
    param3: dtype: data type
    param4: data lower limit
    param5: data upper limit
    """
    r1=np.random.uniform(low,high,size=shape)
    r2=np.random.uniform(low,high,size=shape)
    rand_data=np.empty((shape[0],shape[1],shape[2]),dtype=dtype)
    for i in range(0,shape[0]):
        for p in range(0,shape[1]):
            for k in range(0,shape[2]):  
                rand_data[i,p,k]=complex(r1[i,p,k],r2[i,p,k])
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

def gen_random_data_COMPLEX64_2d():
    data_files=["ifft/data/ifft_data_input1_1.txt",
                "ifft/data/ifft_data_output1_1.txt"]
    np.random.seed(23458)
    shape_x1 = [4, 32]
    a = gen_data_file_2d(data_files[0], shape_x1, np.complex64, 0, 10)
    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x1)
    re = tf.signal.ifft(x1)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_COMPLEX64_3d():
    data_files=["ifft/data/ifft_data_input1_2.txt",
                "ifft/data/ifft_data_output1_2.txt"]
    np.random.seed(23458)
    shape_x1 = [8, 32, 32]
    a = gen_data_file_3d(data_files[0], shape_x1, np.complex64, 0, 10)
    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x1)
    re = tf.signal.ifft(x1)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def gen_random_data_COMPLEX128_3d():
    data_files=["ifft/data/ifft_data_input1_3.txt",
                "ifft/data/ifft_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x1 = [32, 32, 32]
    a = gen_data_file_3d(data_files[0], shape_x1, np.complex128, -100, 100)
    x1 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x1)
    re = tf.signal.ifft(x1)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
    write_file_txt(data_files[1], data, fmt="%s")

def run():
    gen_random_data_COMPLEX64_2d()
    gen_random_data_COMPLEX64_3d()
    gen_random_data_COMPLEX128_3d()