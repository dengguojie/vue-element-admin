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
from numpy import testing
import tensorflow as tf
import numpy as np

# prama1: file_name: the file which store the data
# param2: data: data which will be stored
# param3: fmt: format
def write_file_txt(file_name, data, fmt="%s"):
    if file_name is None:
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
def read_file_txt_to_boll(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)

# param1: shape: data shape
# param2: dtype: data type
# param3: rand_type: the method of generate data, select from "randint, uniform"
# param4: data lower limit
# param5: data upper limit
def gen_data(shape_y, shape_dy, dtype, rand_type, low, high):
    if rand_type == "randint":        
        if dtype == "complex64":
            y_real = np.random.randint(low, high, size=shape_y)
            y_imag = np.random.randint(low, high, size=shape_y)
            rand_data_y = np.complex64(y_real)
            rand_data_y.imag = y_imag
            dy_real = np.random.randint(low, high, size=shape_dy)
            dy_imag = np.random.randint(low, high, size=shape_dy)
            rand_data_dy = np.complex64(dy_real)
            rand_data_dy.imag = dy_imag
        elif dtype == "complex128":
            y_real = np.random.randint(low, high, size=shape_y)
            y_imag = np.random.randint(low, high, size=shape_y)
            rand_data_y = np.complex128(y_real)
            rand_data_y.imag = y_imag
            dy_real = np.random.randint(low, high, size=shape_dy)
            dy_imag = np.random.randint(low, high, size=shape_dy)
            rand_data_dy = np.complex128(dy_real)
            rand_data_dy.imag = dy_imag
        else:
            rand_data_y = np.random.randint(low, high, size=shape_y)
            rand_data_dy = np.random.randint(low, high, size=shape_dy)
    else:
        if dtype == "complex64":
            y_real = np.random.uniform(low, high, size=shape_y)
            y_imag = np.random.uniform(low, high, size=shape_y)
            rand_data_y = np.complex64(y_real)
            rand_data_y.imag = y_imag
            dy_real = np.random.uniform(low, high, size=shape_dy)
            dy_imag = np.random.uniform(low, high, size=shape_dy)
            rand_data_dy = np.complex64(dy_real)
            rand_data_dy.imag = dy_imag
        elif dtype == "complex128":
            y_real = np.random.uniform(low, high, size=shape_y)
            y_imag = np.random.uniform(low, high, size=shape_y)
            rand_data_y = np.complex128(y_real)
            rand_data_y.imag = y_imag
            dy_real = np.random.uniform(low, high, size=shape_dy)
            dy_imag = np.random.uniform(low, high, size=shape_dy)
            rand_data_dy = np.complex128(dy_real)
            rand_data_dy.imag = dy_imag
        else:
            rand_data_y = np.random.uniform(low, high, size=shape_y)
            rand_data_dy = np.random.uniform(low, high, size=shape_dy)
    return np.array(rand_data_y, dtype=dtype), np.array(rand_data_dy, dtype=dtype)

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    return session_config

def gen_random_data(nptype, shape_y, shape_dy, rand_type, low, high, flag):
    if(shape_y == shape_dy):
        if flag == 1:
            np.random.seed(23457)
            y_batch, dy_batch = gen_data(shape_y,shape_dy, nptype, rand_type, low, high)
            write_file_txt("tanhgrad/data/tanhgrad_data_input_y_" + nptype.__name__ + ".txt", y_batch, fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_data_input_dy_" + nptype.__name__ + ".txt", dy_batch, fmt="%s")
            y_batch.tofile("tanhgrad/data/tanhgrad_data_input_y_" + nptype.__name__ + ".bin")
            dy_batch.tofile("tanhgrad/data/tanhgrad_data_input_dy_" + nptype.__name__ + ".bin")
            write_file_txt("tanhgrad/data/tanhgrad_data_y_" + nptype.__name__ + "_dim.txt", np.array([len(shape_y)]), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_data_y_" + nptype.__name__ + "_shape.txt", np.array(shape_y), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_data_dy_" + nptype.__name__ + "_dim.txt", np.array([len(shape_dy)]), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_data_dy_" + nptype.__name__ + "_shape.txt", np.array(shape_dy), fmt="%s")

            y_placeholder = tf.compat.v1.placeholder(nptype, shape=shape_y)
            dy_placeholder = tf.compat.v1.placeholder(nptype, shape=shape_dy)
            result = tf.raw_ops.TanhGrad(y=y_placeholder, dy=dy_placeholder)
            with tf.compat.v1.Session(config=config('cpu')) as session:
                data = session.run(result, feed_dict={y_placeholder: y_batch, dy_placeholder: dy_batch})
            write_file_txt("tanhgrad/data/tanhgrad_data_output_" + nptype.__name__ + ".txt", data, fmt="%s")
            data.tofile("tanhgrad/data/tanhgrad_data_output_" + nptype.__name__ + ".bin")
            write_file_txt("tanhgrad/data/tanhgrad_data_output_" + nptype.__name__ + "_dim.txt", np.array([len(data.shape)]), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_data_output_" + nptype.__name__ + "_shape.txt", np.array(data.shape), fmt="%s")

        else:
            np.random.seed(23457)
            y_batch, dy_batch = gen_data(shape_y,shape_dy, nptype, rand_type, low, high)
            write_file_txt("tanhgrad/data/tanhgrad_bigdata_input_y_" + nptype.__name__ + ".txt", y_batch, fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_bigdata_input_dy_" + nptype.__name__ + ".txt", dy_batch, fmt="%s")
            y_batch.tofile("tanhgrad/data/tanhgrad_bigdata_input_y_" + nptype.__name__ + ".bin")
            dy_batch.tofile("tanhgrad/data/tanhgrad_bigdata_input_dy_" + nptype.__name__ + ".bin")
            write_file_txt("tanhgrad/data/tanhgrad_bigdata_y_" + nptype.__name__ + "_dim.txt", np.array([len(shape_y)]), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_bigdata_y_" + nptype.__name__ + "_shape.txt", np.array(shape_y), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_bigdata_dy_" + nptype.__name__ + "_dim.txt", np.array([len(shape_dy)]), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_bigdata_dy_" + nptype.__name__ + "_shape.txt", np.array(shape_dy), fmt="%s")

            y_placeholder = tf.compat.v1.placeholder(nptype, shape=shape_y)
            dy_placeholder = tf.compat.v1.placeholder(nptype, shape=shape_dy)
            result = tf.raw_ops.TanhGrad(y=y_placeholder, dy=dy_placeholder)
            with tf.compat.v1.Session(config=config('cpu')) as session:
                data = session.run(result, feed_dict={y_placeholder: y_batch, dy_placeholder: dy_batch})
            write_file_txt("tanhgrad/data/tanhgrad_bigdata_output_" + nptype.__name__ + ".txt", data, fmt="%s")
            data.tofile("tanhgrad/data/tanhgrad_bigdata_output_" + nptype.__name__ + ".bin")
            write_file_txt("tanhgrad/data/tanhgrad_bigdata_output_" + nptype.__name__ + "_dim.txt", np.array([len(data.shape)]), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_bigdata_output_" + nptype.__name__ + "_shape.txt", np.array(data.shape), fmt="%s")
    else:
        if flag == 1:
            np.random.seed(23457)
            y_batch, dy_batch = gen_data(shape_y,shape_dy, nptype, rand_type, low, high)
            write_file_txt("tanhgrad/data/tanhgrad_diff_data_input_y_" + nptype.__name__ + ".txt", y_batch, fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_diff_data_input_dy_" + nptype.__name__ + ".txt", dy_batch, fmt="%s")
            y_batch.tofile("tanhgrad/data/tanhgrad_diff_data_input_y_" + nptype.__name__ + ".bin")
            dy_batch.tofile("tanhgrad/data/tanhgrad_diff_data_input_dy_" + nptype.__name__ + ".bin")
            write_file_txt("tanhgrad/data/tanhgrad_diff_data_y_" + nptype.__name__ + "_dim.txt", np.array([len(shape_y)]), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_diff_data_y_" + nptype.__name__ + "_shape.txt", np.array(shape_y), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_diff_data_dy_" + nptype.__name__ + "_dim.txt", np.array([len(shape_dy)]), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_diff_data_dy_" + nptype.__name__ + "_shape.txt", np.array(shape_dy), fmt="%s")

            y_placeholder = tf.compat.v1.placeholder(nptype, shape=shape_y)
            dy_placeholder = tf.compat.v1.placeholder(nptype, shape=shape_dy)
            result = tf.raw_ops.TanhGrad(y=y_placeholder, dy=dy_placeholder)
            with tf.compat.v1.Session(config=config('cpu')) as session:
                data = session.run(result, feed_dict={y_placeholder: y_batch, dy_placeholder: dy_batch})
            write_file_txt("tanhgrad/data/tanhgrad_diff_data_output_" + nptype.__name__ + ".txt", data, fmt="%s")
            data.tofile("tanhgrad/data/tanhgrad_diff_data_output_" + nptype.__name__ + ".bin")
            write_file_txt("tanhgrad/data/tanhgrad_diff_data_output_" + nptype.__name__ + "_dim.txt", np.array([len(data.shape)]), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_diff_data_output_" + nptype.__name__ + "_shape.txt", np.array(data.shape), fmt="%s")

        else:
            np.random.seed(23457)
            y_batch, dy_batch = gen_data(shape_y,shape_dy, nptype, rand_type, low, high)
            write_file_txt("tanhgrad/data/tanhgrad_diff_bigdata_input_y_" + nptype.__name__ + ".txt", y_batch, fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_diff_bigdata_input_dy_" + nptype.__name__ + ".txt", dy_batch, fmt="%s")
            y_batch.tofile("tanhgrad/data/tanhgrad_diff_bigdata_input_y_" + nptype.__name__ + ".bin")
            dy_batch.tofile("tanhgrad/data/tanhgrad_diff_bigdata_input_dy_" + nptype.__name__ + ".bin")
            write_file_txt("tanhgrad/data/tanhgrad_diff_bigdata_y_" + nptype.__name__ + "_dim.txt", np.array([len(shape_y)]), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_diff_bigdata_y_" + nptype.__name__ + "_shape.txt", np.array(shape_y), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_diff_bigdata_dy_" + nptype.__name__ + "_dim.txt", np.array([len(shape_dy)]), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_diff_bigdata_dy_" + nptype.__name__ + "_shape.txt", np.array(shape_dy), fmt="%s")

            y_placeholder = tf.compat.v1.placeholder(nptype, shape=shape_y)
            dy_placeholder = tf.compat.v1.placeholder(nptype, shape=shape_dy)
            result = tf.raw_ops.TanhGrad(y=y_placeholder, dy=dy_placeholder)
            with tf.compat.v1.Session(config=config('cpu')) as session:
                data = session.run(result, feed_dict={y_placeholder: y_batch, dy_placeholder: dy_batch})
            write_file_txt("tanhgrad/data/tanhgrad_diff_bigdata_output_" + nptype.__name__ + ".txt", data, fmt="%s")
            data.tofile("tanhgrad/data/tanhgrad_diff_bigdata_output_" + nptype.__name__ + ".bin")
            write_file_txt("tanhgrad/data/tanhgrad_diff_bigdata_output_" + nptype.__name__ + "_dim.txt", np.array([len(data.shape)]), fmt="%s")
            write_file_txt("tanhgrad/data/tanhgrad_diff_bigdata_output_" + nptype.__name__ + "_shape.txt", np.array(data.shape), fmt="%s")
def run():
    gen_random_data(np.float16, [7, 12, 30],[7, 12, 30], "uniform", -6, 6, 1)
    gen_random_data(np.float16, [20, 100, 30],[20, 100, 30], "uniform", -6, 6, 0)
    gen_random_data(np.float32, [7, 12, 30],[7, 12, 30], "uniform", -6, 6, 1)
    gen_random_data(np.float64, [7, 12, 30],[7, 12, 30], "uniform", -6, 6, 1)
    gen_random_data(np.complex64, [15, 12, 30],[15, 12, 30], "uniform", -6, 6, 1)
    gen_random_data(np.complex128, [7, 12, 30],[7, 12, 30], "uniform", -6, 6, 1)
    # test broadcast data
    gen_random_data(np.float16, [2, 2, 1], [2, 2, 2], "uniform", -6, 6, 1)
    gen_random_data(np.float32, [2, 1, 4], [1, 2, 4], "uniform", -6, 6, 1)
    gen_random_data(np.float64, [1], [2, 2, 2], "uniform", -6, 6, 1)
