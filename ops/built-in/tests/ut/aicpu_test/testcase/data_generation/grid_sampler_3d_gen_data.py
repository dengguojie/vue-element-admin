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
import torch
import numpy as np

'''
prama1: file_name: the file which store the data
param2: data: data which will be stored
param3: fmt: format
'''
def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')

'''
prama1: file_name: the file which store the data
param2: dtype: data type
param3: delim: delimiter which is used to split data
'''
def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()

'''
prama1: file_name: the file which store the data
param2: delim: delimiter which is used to split data
'''
def read_file_txt_to_boll(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)

'''
prama1: data_file: the file which store the generation data
param2: shape: data shape
param3: dtype: data type
param4: rand_type: the method of generate data, select from "randint, uniform"
param5: data lower limit
param6: data upper limit
'''
def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    data = np.array(rand_data, dtype=dtype)
    write_file_txt(data_file, data, fmt="%s")
    return data

def gen_random_x_double_with_grid_double1():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_1.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_1.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_1.txt"]
    np.random.seed(2356)
    shape_x = [5, 2, 4, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.double, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.double, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=0, padding_mode=0, align_corners=True)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float_with_grid_float1():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_2.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_2.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_2.txt"]
    np.random.seed(23)
    shape_x = [5, 3, 3, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.float32, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=0, padding_mode=0, align_corners=True)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_double_with_grid_double2():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_3.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_3.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_3.txt"]
    np.random.seed(2)
    shape_x = [5, 3, 2, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.double, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.double, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=0, padding_mode=1, align_corners=True)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float_with_grid_float2():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_4.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_4.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_4.txt"]
    np.random.seed(767)
    shape_x = [32, 2, 4, 112, 112]
    x = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [32, 2, 36, 36, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.float32, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=0, padding_mode=1, align_corners=True)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_double_with_grid_double3():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_5.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_5.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_5.txt"]
    np.random.seed(116)
    shape_x = [5, 3, 2, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.double, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.double, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=0, padding_mode=2, align_corners=True)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float_with_grid_float3():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_6.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_6.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_6.txt"]
    np.random.seed(26)
    shape_x = [5, 3, 2, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.float32, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=0, padding_mode=2, align_corners=True)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_double_with_grid_double4():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_7.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_7.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_7.txt"]
    np.random.seed(2135)
    shape_x = [5, 3, 2, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.double, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.double, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=0, padding_mode=0, align_corners=False)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float_with_grid_float4():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_8.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_8.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_8.txt"]
    np.random.seed(231)
    shape_x = [5, 3, 2, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.float32, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=0, padding_mode=0, align_corners=False)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_double_with_grid_double5():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_9.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_9.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_9.txt"]
    np.random.seed(126)
    shape_x = [5, 3, 2, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.double, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.double, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=0, padding_mode=1, align_corners=False)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float_with_grid_float5():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_10.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_10.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_10.txt"]
    np.random.seed(296)
    shape_x = [5, 3, 2, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.float32, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=0, padding_mode=1, align_corners=False)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_double_with_grid_double6():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_11.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_11.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_11.txt"]
    np.random.seed(2656)
    shape_x = [5, 3, 2, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.double, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.double, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=0, padding_mode=2, align_corners=False)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float_with_grid_float6():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_12.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_12.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_12.txt"]
    np.random.seed(456)
    shape_x = [5, 3, 2, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.float32, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=0, padding_mode=2, align_corners=False)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_double_with_grid_double7():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_13.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_13.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_13.txt"]
    np.random.seed(466)
    shape_x = [5, 3, 2, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.double, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.double, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=1, padding_mode=0, align_corners=True)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float_with_grid_float7():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_14.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_14.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_14.txt"]
    np.random.seed(266)
    shape_x = [5, 7, 21, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 2, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.float32, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=1, padding_mode=0, align_corners=True)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_double_with_grid_double8():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_15.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_15.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_15.txt"]
    np.random.seed(776)
    shape_x = [5, 1, 5, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.double, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.double, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=1, padding_mode=1, align_corners=True)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float_with_grid_float8():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_16.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_16.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_16.txt"]
    np.random.seed(746)
    shape_x = [5, 2, 4, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.float32, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=1, padding_mode=1, align_corners=True)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_double_with_grid_double9():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_17.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_17.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_17.txt"]
    np.random.seed(876)
    shape_x = [5, 2, 4, 11, 7]
    x = gen_data_file(data_files[0], shape_x, np.double, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.double, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=1, padding_mode=2, align_corners=True)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float_with_grid_float9():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_18.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_18.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_18.txt"]
    np.random.seed(95)
    shape_x = [1, 3, 4, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [1, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.float32, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=1, padding_mode=2, align_corners=True)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_double_with_grid_double10():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_19.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_19.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_19.txt"]
    np.random.seed(256)
    shape_x = [5, 2, 4, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.double, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.double, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=1, padding_mode=0, align_corners=False)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float_with_grid_float10():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_20.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_20.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_20.txt"]
    np.random.seed(216)
    shape_x = [5, 2, 4, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.float32, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=1, padding_mode=0, align_corners=False)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_double_with_grid_double11():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_21.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_21.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_21.txt"]
    np.random.seed(16)
    shape_x = [5, 2, 4, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.double, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.double, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=1, padding_mode=1, align_corners=False)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float_with_grid_float11():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_22.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_22.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_22.txt"]
    np.random.seed(16)
    shape_x = [5, 2, 4, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.float32, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=1, padding_mode=1, align_corners=False)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_double_with_grid_double12():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_23.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_23.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_23.txt"]
    np.random.seed(6)
    shape_x = [32, 2, 4, 224, 224]
    x = gen_data_file(data_files[0], shape_x, np.double, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [32, 2, 36, 36, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.double, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=1, padding_mode=2, align_corners=False)
    write_file_txt(data_files[2], result, fmt="%s")

def gen_random_x_float_with_grid_float12():
    data_files=["grid_sampler_3d/data/gridsampler3d_data_input1_24.txt",
                "grid_sampler_3d/data/gridsampler3d_data_input2_24.txt",
                "grid_sampler_3d/data/gridsampler3d_data_output1_24.txt"]
    np.random.seed(206)
    shape_x = [5, 2, 4, 4, 4]
    x = gen_data_file(data_files[0], shape_x, np.float32, "uniform", 0, 100)
    x = torch.from_numpy(x)
    shape_grid = [5, 1, 3, 2, 3]
    grid = gen_data_file(data_files[1], shape_grid, np.float32, "uniform", -1, 1)
    grid = torch.from_numpy(grid)
    result = torch.grid_sampler(x, grid, interpolation_mode=1, padding_mode=2, align_corners=False)
    write_file_txt(data_files[2], result, fmt="%s")

def run():
    gen_random_x_double_with_grid_double1()
    gen_random_x_float_with_grid_float1()
    gen_random_x_double_with_grid_double2()
    gen_random_x_float_with_grid_float2()
    gen_random_x_double_with_grid_double3()
    gen_random_x_float_with_grid_float3()
    gen_random_x_double_with_grid_double4()
    gen_random_x_float_with_grid_float4()
    gen_random_x_double_with_grid_double5()
    gen_random_x_float_with_grid_float5()
    gen_random_x_double_with_grid_double6()
    gen_random_x_float_with_grid_float6()
    gen_random_x_double_with_grid_double7()
    gen_random_x_float_with_grid_float7()
    gen_random_x_double_with_grid_double8()
    gen_random_x_float_with_grid_float8()
    gen_random_x_double_with_grid_double9()
    gen_random_x_float_with_grid_float9()
    gen_random_x_double_with_grid_double10()
    gen_random_x_float_with_grid_float10()
    gen_random_x_double_with_grid_double11()
    gen_random_x_float_with_grid_float11()
    gen_random_x_double_with_grid_double12()
    gen_random_x_float_with_grid_float12()
