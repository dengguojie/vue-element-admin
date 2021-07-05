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

def gen_random_data(data_type, indices_type, axis):
    data_files = [
        "scatter_elements/data/scatter_elements_data_input1_"+data_type,
        "scatter_elements/data/scatter_elements_data_input2_"+data_type,
        "scatter_elements/data/scatter_elements_data_input3_"+data_type,
        "scatter_elements/data/scatter_elements_data_attr_"+data_type,
        "scatter_elements/data/scatter_elements_data_output_"+data_type,
    ]
    np.random.seed(234567)
    shape_x1 = [10, 5, 10, 5, 5]
    shape_x2 = [1, 2, 1, 1, 2]
    shape_x3 = [1, 2, 1, 1, 2]
    datas = 0
    indices = 0
    updates = 0
    write_file_txt(data_files[3], np.array([axis], dtype=np.int64), fmt="%s")
    if axis < 0:
        axis = axis + 5
    if indices_type == "DT_INT32":
        indices = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0, shape_x1[axis])
    if indices_type == "DT_INT64":
        indices = gen_data_file(data_files[1], shape_x2, np.int64, "randint", 0, shape_x1[axis])
    if data_type == "DT_INT8":
        datas = gen_data_file(data_files[0], shape_x1, np.int8, "randint", -10000, 10000)
        updates = gen_data_file(data_files[2], shape_x3, np.int8, "randint", -10000, 10000)
    if data_type == "DT_INT16":
        datas = gen_data_file(data_files[0], shape_x1, np.int16, "randint", -10000, 10000)
        updates = gen_data_file(data_files[2], shape_x3, np.int16, "randint", -10000, 10000)
    if data_type == "DT_INT32":
        datas = gen_data_file(data_files[0], shape_x1, np.int32, "randint", -10000, 10000)
        updates = gen_data_file(data_files[2], shape_x3, np.int32, "randint", -10000, 10000)
    if data_type == "DT_INT64":
        datas = gen_data_file(data_files[0], shape_x1, np.int64, "randint", -10000, 10000)
        updates = gen_data_file(data_files[2], shape_x3, np.int64, "randint", -10000, 10000)
    if data_type == "DT_UINT8":
        datas = gen_data_file(data_files[0], shape_x1, np.uint8, "randint", -10000, 10000)
        updates = gen_data_file(data_files[2], shape_x3, np.uint8, "randint", -10000, 10000)
    if data_type == "DT_UINT16":
        datas = gen_data_file(data_files[0], shape_x1, np.uint16, "randint", -10000, 10000)
        updates = gen_data_file(data_files[2], shape_x3, np.uint16, "randint", -10000, 10000)
    if data_type == "DT_UINT32":
        datas = gen_data_file(data_files[0], shape_x1, np.uint32, "randint", -10000, 10000)
        updates = gen_data_file(data_files[2], shape_x3, np.uint32, "randint", -10000, 10000)
    if data_type == "DT_UINT64":
        datas = gen_data_file(data_files[0], shape_x1, np.uint64, "randint", -10000, 10000)
        updates = gen_data_file(data_files[2], shape_x3, np.uint64, "randint", -10000, 10000)
    if data_type == "DT_COMPLEX64":
        datas = gen_data_file(data_files[0], shape_x1, np.complex64, "randint", -10000, 10000)
        updates = gen_data_file(data_files[2], shape_x3, np.complex64, "randint", -10000, 10000)
    if data_type == "DT_COMPLEX128":
        datas = gen_data_file(data_files[0], shape_x1, np.complex128, "randint", -10000, 10000)
        updates = gen_data_file(data_files[2], shape_x3, np.complex128, "randint", -10000, 10000)
    for i in range(shape_x2[0]):
        for j in range(shape_x2[1]):
            for k in range(shape_x2[2]):
                for m in range(shape_x2[3]):
                    for n in range(shape_x2[4]):
                        if axis == 0:
                            datas[indices[i][j][k][m][n]][j][k][m][n] = updates[i][j][k][m][n]
                        if axis == 1:
                            datas[i][indices[i][j][k][m][n]][k][m][n] = updates[i][j][k][m][n]
                        if axis == 2:
                            datas[i][j][indices[i][j][k][m][n]][m][n] = updates[i][j][k][m][n]
                        if axis == 3:
                            datas[i][j][k][indices[i][j][k][m][n]][n] = updates[i][j][k][m][n]
                        if axis == 4:
                            datas[i][j][k][m][indices[i][j][k][m][n]] = updates[i][j][k][m][n]
    write_file_txt(data_files[4], datas, fmt="%s")

    

def run():
    gen_random_data("DT_INT8", "DT_INT32", 3)
    gen_random_data("DT_INT16", "DT_INT64", 2)
    gen_random_data("DT_INT32", "DT_INT32", 0)
    gen_random_data("DT_INT64", "DT_INT64", 4)
    gen_random_data("DT_UINT8", "DT_INT32", -1)
    gen_random_data("DT_UINT16", "DT_INT32", 3)
    gen_random_data("DT_UINT32", "DT_INT32", -3)
    gen_random_data("DT_UINT64", "DT_INT64", 3)
    gen_random_data("DT_COMPLEX64", "DT_INT64", 2)
    gen_random_data("DT_COMPLEX128", "DT_INT64", 1)
