import os
import sys
import logging
import torch
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

def gen_data_file_double1(data_file):    
    data = np.array([[[255.123456,255.1234],[2.53,12.3],[9.999,2.911231]],[[45.1234,9.1232],[87,15],[17.23,25.123125]]],dtype=np.double)
    write_file_txt(data_file, data, fmt="%s")
    return data

def gen_data_file_double2(data_file):    
    data = np.array([[[255.123456,255.1234],[2.53,12.3],[9.999,2.911231]],[[45.1234,9.122],[87,15],[17.23,25.123125]]],dtype=np.double)
    write_file_txt(data_file, data, fmt="%s")
    return data

def gen_data_file_float16(data_file):    
    data = np.array([[[255.123456,255.1234],[2.53,12.3],[9.999,2.911231]],[[45.1234,9.1232],[87,15],[17.23,25.123125]]],dtype=np.float32)
    write_file_txt(data_file, data, fmt="%s")
    data = np.array(data, dtype=np.float16)    
    return data

def gen_random_data_int32():
    data_files=["tensor_equal/data/tensor_equal_data_input1_1.txt",
                  "tensor_equal/data/tensor_equal_data_input2_1.txt",
                  "tensor_equal/data/tensor_equal_data_output1_1.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 1, 3]
    shape_x2 = [2, 1, 3]
    x1 = gen_data_file(data_files[0], shape_x1, np.int32, "randint", 0, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0, 10)
    re = torch.equal(torch.from_numpy(x1), torch.from_numpy(x2))
    data = torch.tensor(re).numpy()
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_int64():
    data_files = ["tensor_equal/data/tensor_equal_data_input1_2.txt",
                  "tensor_equal/data/tensor_equal_data_input2_2.txt",
                  "tensor_equal/data/tensor_equal_data_output1_2.txt"]
    np.random.seed(23457)
    shape_x1 = [1024, 8]
    shape_x2 = [1024, 8]
    x1 = gen_data_file(data_files[0], shape_x1, np.int64, "randint", -10000, 10000)
    x2 = gen_data_file(data_files[1], shape_x2, np.int64, "randint", -10000, 10000)
    re = torch.equal(torch.from_numpy(x1), torch.from_numpy(x2))
    data = torch.tensor(re).numpy()
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float():
    data_files = ["tensor_equal/data/tensor_equal_data_input1_3.txt",
                  "tensor_equal/data/tensor_equal_data_input2_3.txt",
                  "tensor_equal/data/tensor_equal_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 1024]
    shape_x2 = [4, 1024]
    x1 = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", -100, 100)
    x2 = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", -100, 100)
    re = torch.equal(torch.from_numpy(x1), torch.from_numpy(x2))
    data = torch.tensor(re).numpy()
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_double():
    data_files = ["tensor_equal/data/tensor_equal_data_input1_4.txt",
                  "tensor_equal/data/tensor_equal_data_input2_4.txt",
                  "tensor_equal/data/tensor_equal_data_output1_4.txt"]
    np.random.seed(23457)
    shape_x1 = [1, 3]
    shape_x2 = [1, 3]
    x1 = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", -100, 100)
    x2 = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", -100, 100)
    re = torch.equal(torch.from_numpy(x1), torch.from_numpy(x2))
    data = torch.tensor(re).numpy()
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_int8():
    data_files = ["tensor_equal/data/tensor_equal_data_input1_5.txt",
                  "tensor_equal/data/tensor_equal_data_input2_5.txt",
                  "tensor_equal/data/tensor_equal_data_output1_5.txt"]
    np.random.seed(23457)
    shape_x1 = [7, 12]
    shape_x2 = [7, 12]
    x1 = gen_data_file(data_files[0], shape_x1, np.int8, "randint", -100, 100)
    x2 = gen_data_file(data_files[1], shape_x2, np.int8, "randint", -100, 100)
    re = torch.equal(torch.from_numpy(x1), torch.from_numpy(x2))
    data = torch.tensor(re).numpy()
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_uint8():
    data_files=["tensor_equal/data/tensor_equal_data_input1_6.txt",
                "tensor_equal/data/tensor_equal_data_input2_6.txt",
                "tensor_equal/data/tensor_equal_data_output1_6.txt"]
    np.random.seed(23457)
    shape_x1 = [7, 12]
    shape_x2 = [7, 12]
    x1 = gen_data_file(data_files[0], shape_x1, np.uint8, "randint", 0, 200)
    x2 = gen_data_file(data_files[1], shape_x2, np.uint8, "randint", 0, 200)
    re = torch.equal(torch.from_numpy(x1), torch.from_numpy(x2))
    data = torch.tensor(re).numpy()
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float16():
    data_files = ["tensor_equal/data/tensor_equal_data_input1_7.txt",
                  "tensor_equal/data/tensor_equal_data_input2_7.txt",
                  "tensor_equal/data/tensor_equal_data_output1_7.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 2, 2]
    shape_x2 = [2, 2, 2]
    x1 = gen_data_file(data_files[0], shape_x1, np.float16, "uniform", -10, 10)
    x2 = gen_data_file(data_files[1], shape_x2, np.float16, "uniform", -10, 10)
    #re = torch.equal(torch.from_numpy(x1), torch.from_numpy(x2))
    re = np.array_equal(x1,x2)
    data = torch.tensor(re).numpy()
    write_file_txt(data_files[2], data, fmt="%s")

def gen_fixed_data_double():
    data_files = ["tensor_equal/data/tensor_equal_data_input1_8.txt",
                  "tensor_equal/data/tensor_equal_data_input2_8.txt",
                  "tensor_equal/data/tensor_equal_data_output1_8.txt"]
    x1 = gen_data_file_double1(data_files[0])
    x2 = gen_data_file_double2(data_files[1])    
    re = torch.equal(torch.from_numpy(x1), torch.from_numpy(x2))
    data = torch.tensor(re).numpy()
    write_file_txt(data_files[2], data, fmt="%s")

def gen_fixed_data_float16():
    data_files = ["tensor_equal/data/tensor_equal_data_input1_9.txt",
                  "tensor_equal/data/tensor_equal_data_input2_9.txt",
                  "tensor_equal/data/tensor_equal_data_output1_9.txt"]
    x1 = gen_data_file_float16(data_files[0])
    x2 = gen_data_file_float16(data_files[1])
    #re = torch.equal(torch.from_numpy(x1), torch.from_numpy(x2))
    re = np.array_equal(x1,x2)
    data = torch.tensor(re).numpy()
    write_file_txt(data_files[2], data, fmt="%s")

def run():
    gen_random_data_int32()
    gen_random_data_int64()
    gen_random_data_float()
    gen_random_data_double()
    gen_random_data_int8()
    gen_random_data_uint8()
    gen_random_data_float16()
    gen_fixed_data_double()
    gen_fixed_data_float16()