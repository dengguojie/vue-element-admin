"""
Copyright 2021 Huawei Technologies Co., Ltd.

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
    """
    prama1: file_name: the file which store the data
    param2: data: data which will be stored
    param3: fmt: format
    """
    if file_name is None:
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


def read_file_txt_to_boll(file_name, delim=None):
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


def gen_data(shape, dtype, rand_type, low, high):
    """
    param1: shape: data shape
    param2: dtype: data type
    param3: rand_type: the method of generate data, select from "randint, uniform"
    param4: data lower limit
    param5: data upper limit
    """
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    else:
        if dtype == np.complex64:
            real = np.random.uniform(low, high, size=shape)
            imag = np.random.uniform(low, high, size=shape)
            rand_data = np.complex64(real)
            rand_data.imag = imag
        elif dtype == np.complex128:
            real = np.random.uniform(low, high, size=shape)
            imag = np.random.uniform(low, high, size=shape)
            rand_data = np.complex128(real)
            rand_data.imag = imag
        else:
            rand_data = np.random.uniform(low, high, size=shape)
    return np.array(rand_data, dtype=dtype)


def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
    return session_config


def gen_random_data(nptype, shape, rand_type, low, high):
    np.random.seed(23457)
    x_batch = gen_data(shape, nptype, rand_type, low, high)
    write_file_txt("sin/data/sin_data_input_" +
                   nptype.__name__ + ".txt", x_batch, fmt="%s")
    write_file_txt("sin/data/sin_data_" + nptype.__name__ +
                   "_dim.txt", np.array([len(shape)]), fmt="%s")
    write_file_txt("sin/data/sin_data_" + nptype.__name__ +
                   "_shape.txt", np.array(shape), fmt="%s")
    x_batch.tofile("sin/data/sin_data_input_" + nptype.__name__ + ".bin")
    x_placeholder = tf.compat.v1.placeholder(nptype, shape=shape)
    result = tf.sin(x_placeholder)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(result, feed_dict={x_placeholder: x_batch})
    write_file_txt("sin/data/sin_data_output_" +
                   nptype.__name__ + ".txt", data, fmt="%s")
    data.tofile("sin/data/sin_data_output_" + nptype.__name__ + ".bin")


def run():
    gen_random_data(np.float16, [2, 10], "uniform", -6, 6)
    gen_random_data(np.float32, [15, 12, 30], "uniform", -100, 100)
    gen_random_data(np.float64, [70, 120, 30], "uniform", -100000, 10000)
    gen_random_data(np.complex64, [15, 12, 30], "uniform", -10, 10)
    gen_random_data(np.complex128, [2, 2, 2], "uniform", -999, 999)
