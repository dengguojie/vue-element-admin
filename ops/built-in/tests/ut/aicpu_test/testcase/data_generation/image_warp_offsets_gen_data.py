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
import numpy as np
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
def read_file_txt_to_boll(file_name, delim=None):
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

def gen_indexs(x1_shape, x2):
    x2_ceil = np.ceil(x2)
    x2_floor = np.floor(x2)
    x_ceil, y_ceil = np.split(x2_ceil, 2, -1)
    x_floor, y_floor = np.split(x2_floor, 2, -1)
    index1 = (y_floor * x1_shape[2] + x_floor) * x1_shape[3]
    index2 = (y_floor * x1_shape[2] + x_ceil) * x1_shape[3]
    index3 = (y_ceil * x1_shape[2] + x_floor) * x1_shape[3]
    index4 = (y_ceil * x1_shape[2] + x_ceil) * x1_shape[3]
    index = np.transpose(np.concatenate([index1, index2, index3, index4], -1), [0, 3, 1, 2]).astype(np.float32)
    return index

def gen_random_data_10x100x200x3_50x100():
    data_files=["image_warp_offsets/data/image_warp_offsets_data_input1_10x100x200x3_float16.txt",
                "image_warp_offsets/data/image_warp_offsets_data_input2_10x4x50x100_float32.txt",
                "image_warp_offsets/data/image_warp_offsets_data_output1_10x4x500x100x3_float16.txt"]
    n = 10
    i_h = 100
    i_w = 200
    c = 3
    x1_shape = [n, i_h, i_w, c]
    x1 = np.random.randint(0, 256, x1_shape).astype(np.float16)
    write_file_txt(data_files[0], x1, fmt="%s")

    o_h = 50
    o_w = 100
    x2_shape = [n, o_h, o_w, 2]
    x2_1 = np.random.uniform(0, i_w - 1, [n, o_h, o_w, 1])
    x2_2 = np.random.uniform(0, i_h - 1, [n, o_h, o_w, 1])
    x2 = np.concatenate([x2_1, x2_2], -1)

    indexs = gen_indexs(x1_shape, x2)
    write_file_txt(data_files[1], indexs, fmt="%s")

    output_shape = [n, 4, o_h, o_w, c]
    output = np.zeros(output_shape).astype(np.float16)
    image = x1.reshape(n, -1)
    for i_n in range(n):
        for i_i in range(4):
            for i_h in range(o_h):
                for i_w in range(o_w):
                    output[i_n][i_i][i_h][i_w][:c] = image[i_n][int(indexs[i_n][i_i][i_h][i_w]):(int(indexs[i_n][i_i][i_h][i_w])+c)]
    write_file_txt(data_files[2], output, fmt="%s")



def run():
    gen_random_data_10x100x200x3_50x100()
