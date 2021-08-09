"""
Copyright 2021 Jilin University
Copyright 2020 Huawei Technologies Co., Ltd.

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
def write_file_txt(file_name, data):
    if np.iscomplexobj(data):
    	fmt="(%s,%s)"
    else:
    	fmt="%s"
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


# param1: shape: data shape
# param2: dtype: data type
# param3: rand_type: the method of generate data, select from "randint, uniform"
# param4: data lower limit
# param5: data upper limit
def gen_data(shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    elif issubclass(dtype, np.complexfloating):
        rand_data = np.random.uniform(low, high, size=shape) + 1j * np.random.uniform(low, high, size=shape)
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
    write_file_txt("asinh/data/asinh_data_input_0_" + nptype.__name__ + ".txt", x_batch)
    x_batch.tofile("asinh/data/asinh_data_input_0_" + nptype.__name__ + ".bin")
    x_placeholder = tf.compat.v1.placeholder(nptype, shape=shape)

    write_file_txt("asinh/data/asinh_data_" + nptype.__name__ + "_dim.txt", np.array([len(shape)]))
    write_file_txt("asinh/data/asinh_data_" + nptype.__name__ + "_shape.txt", np.array(shape))
    result = tf.asinh(x_placeholder)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(result, feed_dict={x_placeholder: x_batch})
    write_file_txt("asinh/data/asinh_data_output_" + nptype.__name__ + "_expect.txt", data)
    data.tofile("asinh/data/asinh_data_output_" + nptype.__name__ + "_expect.bin")

def run():
    gen_random_data(np.float16, [12, 130, 3], "uniform", -100, 100)
    gen_random_data(np.float32, [12, 130, 3], "uniform", -100, 100)
    gen_random_data(np.float64, [12, 130, 3], "uniform", -100, 100)
    gen_random_data(np.complex64, [12, 130, 3], "uniform", -100, 100)
    gen_random_data(np.complex128, [12, 130, 3], "uniform", -100, 100)
