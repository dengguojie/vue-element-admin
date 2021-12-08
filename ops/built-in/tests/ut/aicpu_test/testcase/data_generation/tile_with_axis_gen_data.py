"""
Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import tensorflow as tf
import numpy as np

# shape_broad = (1, 2, 1) # axis = 1, tiles = 2

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

def gen_random_data(caseno, nptype, shape, rand_type, low, high, shape_broad):
    np.random.seed(23457)
    write_file_txt("tile_with_axis/data/tilewithaxis_data_input_shape_" + caseno + ".txt",
                   np.array(shape), fmt="%s")
    input = gen_data_file("tile_with_axis/data/tilewithaxis_input_data_" + caseno + ".txt",
                          shape, nptype, rand_type, low, high)
    output = np.tile(input, shape_broad)
    write_file_txt("tile_with_axis/data/tilewithaxis_output_data_" + caseno + ".txt",
                   output, fmt="%s")

def run():
    gen_random_data("1", np.int64, [1, 2, 3], "randint", -100, 100, [1, 2, 1])    
    gen_random_data("2", np.int32, [3, 2, 1], "randint", -100, 100, [1, 2, 1])
    gen_random_data("3", np.int16, [2, 3, 2], "randint", -100, 100, [2, 1, 1])
    gen_random_data("4", np.int8, [2, 3, 1], "randint", -100, 100, [1, 1, 2])

    gen_random_data("5", np.uint64, [1, 2, 3], "randint", 0, 100, [1, 2, 1])
    gen_random_data("6", np.uint32, [3, 2, 1], "randint", 0, 100, [1, 2, 1])
    gen_random_data("7", np.uint16, [2, 3, 2], "randint", 0, 100, [2, 1, 1])
    gen_random_data("8", np.uint8, [2, 3, 1], "randint", 0, 100, [1, 1, 2])

    gen_random_data("9", np.float16, [2, 3, 2], "uniform", -100, 100, [1, 2, 1])
    gen_random_data("10", np.float, [2, 3, 2], "uniform", -100, 100, [1, 2, 1])

    gen_random_data("11", np.int64, [1, 2, 3, 4], "randint",
                    -100, 100, [1, 2, 1, 1])
    gen_random_data("12", np.int32, [1, 2, 3, 4, 1], "randint",
                    -100, 100, [1, 2, 1, 1, 1])
    gen_random_data("13", np.int16, [1, 2, 3, 4, 1, 1], "randint", 
                    -100, 100, [1, 2, 1, 1, 1, 1])
    gen_random_data("14", np.int8, [1, 2, 3, 4, 1, 1, 1], "randint", 
                    -100, 100, [1, 1, 1, 1, 1, 1, 2])
    gen_random_data("15", np.int8, [1, 2, 3, 4, 1, 1, 1, 1], "randint", 
                    -100, 100, [1, 1, 1, 1, 1, 1, 2 ,1])
    gen_random_data("16", np.int32, [1, 2], "randint", -100, 100, [2, 1])
    gen_random_data("17", np.int32, [2], "randint", -100, 100, [2])
    gen_random_data("18", np.int32, [1, 2], "randint", -100, 100, [1, 1])