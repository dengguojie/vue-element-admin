from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import torch
import numpy as np


"""
prama1: file_name: the file which store the data
param2: data: data which will be stored
param3: fmt: format
"""
def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')

"""
prama1: file_name: the file which store the data
param2: dtype: data type
param3: delim: delimiter which is used to split data
"""
def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()

"""
prama1: file_name: the file which store the data
param2: delim: delimiter which is used to split data
"""
def read_file_txt_to_bool(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)

"""
prama1: data_file: the file which store the generation data
param2: shape: data shape
param3: dtype: data type
param4: rand_type: the method of generate data, select from "randint, uniform"
param5: data lower limit
param6: data upper limit
"""
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

def gen_random_data_float_b_b():
    data_files = ["addcdiv/data/addcdiv_data_input1_1.txt",
                  "addcdiv/data/addcdiv_data_input2_1.txt",
                  "addcdiv/data/addcdiv_data_input3_1.txt",
                  "addcdiv/data/addcdiv_data_input4_1.txt",
                  "addcdiv/data/addcdiv_data_output1_1.txt"]
    np.random.seed(23457)
    shape_x1 = [1, 4, 1]
    shape_x2 = [4, 1, 4]
    shape_x3 = [1, 4, 4]
    shape_x4 = [1]
    x1 = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 10)
    x3 = gen_data_file(data_files[2], shape_x3, np.float32, "uniform", 0, 10)
    x4 = gen_data_file(data_files[3], shape_x4, np.float32, "uniform", 0, 10)
    re = torch.addcdiv(input=torch.tensor(x1, dtype=torch.float32), tensor1=torch.tensor(x2, dtype=torch.float32), tensor2=torch.tensor(x3, dtype=torch.float32), value=x4[0])
    data = re.numpy()
    write_file_txt(data_files[4], data, fmt="%s")

def gen_random_data_float_b_s():
    data_files = ["addcdiv/data/addcdiv_data_input1_2.txt",
                  "addcdiv/data/addcdiv_data_input2_2.txt",
                  "addcdiv/data/addcdiv_data_input3_2.txt",
                  "addcdiv/data/addcdiv_data_input4_2.txt",
                  "addcdiv/data/addcdiv_data_output1_2.txt"]
    np.random.seed(23457)
    shape_x1 = [1, 4]
    shape_x2 = [4, 4]
    shape_x3 = [4, 4]
    shape_x4 = [1]
    x1 = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 10)
    x3 = gen_data_file(data_files[2], shape_x3, np.float32, "uniform", 0, 10)
    x4 = gen_data_file(data_files[3], shape_x4, np.float32, "uniform", 0, 10)
    re = torch.addcdiv(input=torch.tensor(x1, dtype=torch.float32), tensor1=torch.tensor(x2, dtype=torch.float32), tensor2=torch.tensor(x3, dtype=torch.float32), value=x4[0])
    data = re.numpy()
    write_file_txt(data_files[4], data, fmt="%s")

def gen_random_data_float_s_b():
    data_files = ["addcdiv/data/addcdiv_data_input1_3.txt",
                  "addcdiv/data/addcdiv_data_input2_3.txt",
                  "addcdiv/data/addcdiv_data_input3_3.txt",
                  "addcdiv/data/addcdiv_data_input4_3.txt",
                  "addcdiv/data/addcdiv_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 4]
    shape_x2 = [4, 1]
    shape_x3 = [1, 4]
    shape_x4 = [1]
    x1 = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 10)
    x3 = gen_data_file(data_files[2], shape_x3, np.float32, "uniform", 0, 10)
    x4 = gen_data_file(data_files[3], shape_x4, np.float32, "uniform", 0, 10)
    re = torch.addcdiv(input=torch.tensor(x1, dtype=torch.float32), tensor1=torch.tensor(x2, dtype=torch.float32), tensor2=torch.tensor(x3, dtype=torch.float32), value=x4[0])
    data = re.numpy()
    write_file_txt(data_files[4], data, fmt="%s")

def gen_random_data_float_s_s():
    data_files = ["addcdiv/data/addcdiv_data_input1_4.txt",
                  "addcdiv/data/addcdiv_data_input2_4.txt",
                  "addcdiv/data/addcdiv_data_input3_4.txt",
                  "addcdiv/data/addcdiv_data_input4_4.txt",
                  "addcdiv/data/addcdiv_data_output1_4.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 4]
    shape_x2 = [4, 4]
    shape_x3 = [4, 4]
    shape_x4 = [1]
    x1 = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 10)
    x3 = gen_data_file(data_files[2], shape_x3, np.float32, "uniform", 0, 10)
    x4 = gen_data_file(data_files[3], shape_x4, np.float32, "uniform", 0, 10)
    re = torch.addcdiv(input=torch.tensor(x1, dtype=torch.float32), tensor1=torch.tensor(x2, dtype=torch.float32), tensor2=torch.tensor(x3, dtype=torch.float32), value=x4[0])
    data = re.numpy()
    write_file_txt(data_files[4], data, fmt="%s")

def gen_random_data_double_s_o():
    data_files = ["addcdiv/data/addcdiv_data_input1_5.txt",
                  "addcdiv/data/addcdiv_data_input2_5.txt",
                  "addcdiv/data/addcdiv_data_input3_5.txt",
                  "addcdiv/data/addcdiv_data_input4_5.txt",
                  "addcdiv/data/addcdiv_data_output1_5.txt"]
    np.random.seed(23457)
    shape_x1 = [16, 1024]
    shape_x2 = [1]
    shape_x3 = [1]
    shape_x4 = [1]
    x1 = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", 0, 10)
    x3 = gen_data_file(data_files[2], shape_x3, np.float64, "uniform", 0, 10)
    x4 = gen_data_file(data_files[3], shape_x4, np.float64, "uniform", 0, 10)
    re = torch.addcdiv(input=torch.tensor(x1, dtype=torch.float64), tensor1=torch.tensor(x2, dtype=torch.float64),
                       tensor2=torch.tensor(x3, dtype=torch.float64), value=x4[0])
    data = re.numpy()
    write_file_txt(data_files[4], data, fmt="%s")

def gen_random_data_double_o_s():
    data_files = ["addcdiv/data/addcdiv_data_input1_6.txt",
                  "addcdiv/data/addcdiv_data_input2_6.txt",
                  "addcdiv/data/addcdiv_data_input3_6.txt",
                  "addcdiv/data/addcdiv_data_input4_6.txt",
                  "addcdiv/data/addcdiv_data_output1_6.txt"]
    np.random.seed(23457)
    shape_x1 = [1]
    shape_x2 = [16, 1024]
    shape_x3 = [16, 1024]
    shape_x4 = [1]
    x1 = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", 0, 10)
    x3 = gen_data_file(data_files[2], shape_x3, np.float64, "uniform", 0, 10)
    x4 = gen_data_file(data_files[3], shape_x4, np.float64, "uniform", 0, 10)
    re = torch.addcdiv(input=torch.tensor(x1, dtype=torch.float64), tensor1=torch.tensor(x2, dtype=torch.float64),
                       tensor2=torch.tensor(x3, dtype=torch.float64), value=x4[0])
    data = re.numpy()
    write_file_txt(data_files[4], data, fmt="%s")

def gen_random_data_double_b_o():
    data_files = ["addcdiv/data/addcdiv_data_input1_7.txt",
                  "addcdiv/data/addcdiv_data_input2_7.txt",
                  "addcdiv/data/addcdiv_data_input3_7.txt",
                  "addcdiv/data/addcdiv_data_input4_7.txt",
                  "addcdiv/data/addcdiv_data_output1_7.txt"]
    np.random.seed(23457)
    shape_x1 = [16, 1024]
    shape_x2 = [1]
    shape_x3 = [16, 1]
    shape_x4 = [1]
    x1 = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", 0, 10)
    x3 = gen_data_file(data_files[2], shape_x3, np.float64, "uniform", 0, 10)
    x4 = gen_data_file(data_files[3], shape_x4, np.float64, "uniform", 0, 10)
    re = torch.addcdiv(input=torch.tensor(x1, dtype=torch.float64), tensor1=torch.tensor(x2, dtype=torch.float64),
                       tensor2=torch.tensor(x3, dtype=torch.float64), value=x4[0])
    data = re.numpy()
    write_file_txt(data_files[4], data, fmt="%s")

#when tenser1/tenser2, the shape of tenser1 is 1
def gen_random_data_double_div_x():
    data_files = ["addcdiv/data/addcdiv_data_input1_8.txt",
                  "addcdiv/data/addcdiv_data_input2_8.txt",
                  "addcdiv/data/addcdiv_data_input3_8.txt",
                  "addcdiv/data/addcdiv_data_input4_8.txt",
                  "addcdiv/data/addcdiv_data_output1_8.txt"]
    np.random.seed(567)
    shape_x1 = [1]
    shape_x2 = [1]
    shape_x3 = [1,2]
    shape_x4 = [1]
    x1 = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", 0, 10)
    x3 = gen_data_file(data_files[2], shape_x3, np.float64, "uniform", 0, 10)
    x4 = gen_data_file(data_files[3], shape_x4, np.float64, "uniform", 0, 10)
    re = torch.addcdiv(input=torch.tensor(x1, dtype=torch.float64), tensor1=torch.tensor(x2, dtype=torch.float64),
                       tensor2=torch.tensor(x3, dtype=torch.float64), value=x4[0])
    data = re.numpy()
    write_file_txt(data_files[4], data, fmt="%s")

#when tenser1/tenser2, the shape of tenser2 is 1
def gen_random_data_double_div_y():
    data_files = ["addcdiv/data/addcdiv_data_input1_9.txt",
                  "addcdiv/data/addcdiv_data_input2_9.txt",
                  "addcdiv/data/addcdiv_data_input3_9.txt",
                  "addcdiv/data/addcdiv_data_input4_9.txt",
                  "addcdiv/data/addcdiv_data_output1_9.txt"]
    np.random.seed(652)
    shape_x1 = [1]
    shape_x2 = [1,2]
    shape_x3 = [1]
    shape_x4 = [1]
    x1 = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", 0, 10)
    x3 = gen_data_file(data_files[2], shape_x3, np.float64, "uniform", 0, 10)
    x4 = gen_data_file(data_files[3], shape_x4, np.float64, "uniform", 0, 10)
    re = torch.addcdiv(input=torch.tensor(x1, dtype=torch.float64), tensor1=torch.tensor(x2, dtype=torch.float64),
                       tensor2=torch.tensor(x3, dtype=torch.float64), value=x4[0])
    data = re.numpy()
    write_file_txt(data_files[4], data, fmt="%s")

#when inputdata + divinfo , the shape of inputdata is 1
def gen_random_data_double_add_x():
    data_files = ["addcdiv/data/addcdiv_data_input1_10.txt",
                  "addcdiv/data/addcdiv_data_input2_10.txt",
                  "addcdiv/data/addcdiv_data_input3_10.txt",
                  "addcdiv/data/addcdiv_data_input4_10.txt",
                  "addcdiv/data/addcdiv_data_output1_10.txt"]
    np.random.seed(2222)
    shape_x1 = [1]
    shape_x2 = [1,3]
    shape_x3 = [1,3]
    shape_x4 = [1]
    x1 = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", 0, 10)
    x3 = gen_data_file(data_files[2], shape_x3, np.float64, "uniform", 0, 10)
    x4 = gen_data_file(data_files[3], shape_x4, np.float64, "uniform", 0, 10)
    re = torch.addcdiv(input=torch.tensor(x1, dtype=torch.float64), tensor1=torch.tensor(x2, dtype=torch.float64),
                       tensor2=torch.tensor(x3, dtype=torch.float64), value=x4[0])
    data = re.numpy()
    write_file_txt(data_files[4], data, fmt="%s")

#when inputdata + divinfo , the shape of divinfo is 1
def gen_random_data_double_add_y():
    data_files = ["addcdiv/data/addcdiv_data_input1_11.txt",
                  "addcdiv/data/addcdiv_data_input2_11.txt",
                  "addcdiv/data/addcdiv_data_input3_11.txt",
                  "addcdiv/data/addcdiv_data_input4_11.txt",
                  "addcdiv/data/addcdiv_data_output1_11.txt"]
    np.random.seed(2222)
    shape_x1 = [1,3]
    shape_x2 = [1]
    shape_x3 = [1]
    shape_x4 = [1]
    x1 = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", 0, 10)
    x3 = gen_data_file(data_files[2], shape_x3, np.float64, "uniform", 0, 10)
    x4 = gen_data_file(data_files[3], shape_x4, np.float64, "uniform", 0, 10)
    re = torch.addcdiv(input=torch.tensor(x1, dtype=torch.float64), tensor1=torch.tensor(x2, dtype=torch.float64),
                       tensor2=torch.tensor(x3, dtype=torch.float64), value=x4[0])
    data = re.numpy()
    write_file_txt(data_files[4], data, fmt="%s")

def gen_random_data_float16_o_b():
    data_files = ["addcdiv/data/addcdiv_data_input1_12.txt",
                  "addcdiv/data/addcdiv_data_input2_12.txt",
                  "addcdiv/data/addcdiv_data_input3_12.txt",
                  "addcdiv/data/addcdiv_data_input4_12.txt",
                  "addcdiv/data/addcdiv_data_output1_12.txt"]
    np.random.seed(23457)
    shape_x1 = [1]
    shape_x2 = [16, 1024]
    shape_x3 = [1, 1024]
    shape_x4 = [1]
    x1 = gen_data_file(data_files[0], shape_x1, np.float16, "uniform", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float16, "uniform", 0, 10)
    x3 = gen_data_file(data_files[2], shape_x3, np.float16, "uniform", 0, 10)
    x4 = gen_data_file(data_files[3], shape_x4, np.float16, "uniform", 0, 1)
    re = torch.addcdiv(input=torch.tensor(x1, dtype=torch.float32), tensor1=torch.tensor(x2, dtype=torch.float32), tensor2=torch.tensor(x3, dtype=torch.float32), value=x4[0])
    data = re.numpy()
    data = np.array(data, dtype=np.float16)
    write_file_txt(data_files[4], data, fmt="%s")

def run():
    gen_random_data_float_b_b()
    gen_random_data_float_b_s()
    gen_random_data_float_s_b()
    gen_random_data_float_s_s()
    gen_random_data_double_s_o()
    gen_random_data_double_o_s()
    gen_random_data_double_b_o()
    gen_random_data_double_div_x()
    gen_random_data_double_div_y()
    gen_random_data_double_add_x()
    gen_random_data_double_add_y()
    gen_random_data_float16_o_b()
 
    
   
