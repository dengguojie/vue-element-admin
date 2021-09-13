import os

import tensorflow as tf
import numpy as np


def write_file_txt(file_name, data, fmt="%s"):
    if(file_name is None):
        print("file name is none, do not write data to file")
        return
        
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')

def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()

def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    if rand_type == "uniform":
        rand_data1 = np.random.uniform(low, high, size=shape)
    elif rand_type == "loguniform":
        rand_data1 = np.exp(np.random.uniform(low, high, size=shape))

    data1 = np.array(rand_data1, dtype=dtype)
    write_file_txt(data_file, data1)
    return data1

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config

def gen_random_data(shape_a = [1600, 1600], shape_b = [1600, 1600], shape_x = [1600, 1600], 
    npdtype = np.float32, tfdtype = tf.float32, 
    num = "1", min_val = -10, max_val = 20000, mode = "uniform"):

    data_files=["betainc/data/betainc_input1_" + num +".txt",
                "betainc/data/betainc_input2_" + num +".txt",
                "betainc/data/betainc_input3_" + num +".txt",
                "betainc/data/betainc_output_" + num +".txt",
                "betainc/data/betainc_output_ascalar_" + num +".txt",
                "betainc/data/betainc_output_bcscalar_" + num +".txt",
                ]
    np.random.seed(23457)

    a = gen_data_file(data_files[0], shape_a, npdtype, mode, min_val, max_val)
    b = gen_data_file(data_files[1], shape_b, npdtype, mode, min_val, max_val)
    x = gen_data_file(data_files[2], shape_x, npdtype, "uniform", 0, 1)

    input1 = tf.compat.v1.placeholder(tfdtype, shape=shape_a)
    input1_scalar = tf.compat.v1.placeholder(tfdtype, shape=[])
    input2 = tf.compat.v1.placeholder(tfdtype, shape=shape_b)
    input2_scalar = tf.compat.v1.placeholder(tfdtype, shape=[])
    input3 = tf.compat.v1.placeholder(tfdtype, shape=shape_x)
    input3_scalar = tf.compat.v1.placeholder(tfdtype, shape=[])

    re = tf.math.betainc(input1, input2, input3)
    re_ascalar = tf.math.betainc(0.5, input2, input3)
    re_bcscalar = tf.math.betainc(input1, 0.5, 0.5)

    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={input1:a, input2:b, input3:x})
        data_ascalar = session.run(re_ascalar, feed_dict={input1_scalar:0.5, input2:b, input3:x})
        data_bcscalar = session.run(re_bcscalar, feed_dict=
            {input1:a, input2_scalar:0.5, input3_scalar:0.5})

    #计算结果作为对标数据写入文件
    write_file_txt(data_files[3], data)
    write_file_txt(data_files[4], data_ascalar)
    write_file_txt(data_files[5], data_bcscalar)

def run():
    gen_random_data(shape_a = [1600,], shape_b = [1600, ], shape_x = [1600, ], num = "1", 
    min_val = -5, max_val = 3, mode = "loguniform")
