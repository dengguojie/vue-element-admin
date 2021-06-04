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
def gen_data(shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    return np.array(rand_data, dtype=dtype)

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    return session_config

def gen_random_data(nptype, shape, rand_type, low, high):
    np.random.seed(23457)
    x_batch = gen_data(shape, nptype, rand_type, low, high)
    write_file_txt("asin/data/asin_data_input_" + nptype.__name__ + ".txt", x_batch, fmt="%s")
    x_batch.tofile("asin/data/asin_data_input_" + nptype.__name__ + ".bin")
    write_file_txt("asin/data/asin_data_" + nptype.__name__ + "_dim.txt", np.array([len(shape)]), fmt="%s")
    write_file_txt("asin/data/asin_data_" + nptype.__name__ + "_shape.txt", np.array(shape), fmt="%s")

    x_placeholder = tf.compat.v1.placeholder(nptype, shape=shape)
    result = tf.asin(x_placeholder)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(result, feed_dict={x_placeholder: x_batch})
    write_file_txt("asin/data/asin_data_output_" + nptype.__name__ + ".txt", data, fmt="%s")
    data.tofile("asin/data/asin_data_output_" + nptype.__name__ + ".bin")

def gen_random_data_numpy(nptype, shape, rand_type, low, high):
    np.random.seed(23457)
    x_batch = gen_data(shape, nptype, rand_type, low, high)
    write_file_txt("asin/data/asin_data_input_" + nptype.__name__ + ".txt", x_batch, fmt="%s")
    x_batch.tofile("asin/data/asin_data_input_" + nptype.__name__ + ".bin")
    write_file_txt("asin/data/asin_data_" + nptype.__name__ + "_dim.txt", np.array([len(shape)]), fmt="%s")
    write_file_txt("asin/data/asin_data_" + nptype.__name__ + "_shape.txt", np.array(shape), fmt="%s")

    data = np.arcsin(x_batch, dtype=nptype)
    write_file_txt("asin/data/asin_data_output_" + nptype.__name__ + ".txt", data, fmt="%s")
    data.tofile("asin/data/asin_data_output_" + nptype.__name__ + ".bin")

def run():
    gen_random_data_numpy(np.float16, [12, 130], "uniform", -1, 1)
    gen_random_data(np.float32, [15, 12, 30], "uniform", -1, 1)
    gen_random_data(np.float64, [7, 12, 30], "uniform", -1, 1)
