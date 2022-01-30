"""
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

def gen_random_data(caseno, nptype, shape, rand_type, low, high, thresh):
    np.random.seed(23457)
    write_file_txt("compare_and_bitpack/data/compare_and_bitpack_input_data_shape_" + caseno + ".txt", 
                   np.array(shape), fmt="%s")
    input = gen_data_file("compare_and_bitpack/data/compare_and_bitpack_input_data_" + caseno + ".txt", 
                          shape, nptype, rand_type, low, high)
    result = tf.raw_ops.CompareAndBitpack(input=input, threshold=thresh)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        output = session.run(result)
    write_file_txt("compare_and_bitpack/data/compare_and_bitpack_output_data_" + caseno + ".txt",
                   output, fmt="%s")

def run():
    gen_random_data("1", np.float, [1, 16], "uniform", 1, 100, 20.0)
    gen_random_data("2", np.double, [1, 16], "uniform", 1, 100, 20.0)
    gen_random_data("3", np.int8, [1,16], "randint", 1, 100, 20)
    gen_random_data("4", np.int16, [1, 16], "randint", 1, 100, 20)
    gen_random_data("5", np.int32, [1, 16], "randint", 1, 100, 20)
    gen_random_data("6", np.int64, [1, 16], "randint", 1,  100, 20)
    gen_random_data("7", np.int64, [1, 2, 8, 16, 8, 1, 8, 8], "randint", 1,  100, 20)
