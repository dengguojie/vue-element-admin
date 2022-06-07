from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import logging
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op, dtypes

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

def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config

def gen_random_data_int32():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_1.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_1.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_1.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_1.txt"]
    np.random.seed(23457)
    shape_x1 = [4,5,3,2]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.int32, "randint", 0, 50) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    
    padding_value = gen_data_file(data_files[2], shape_x3, np.int32, "randint", 0, 50)
    
    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(tf.int32,shape = shape_x3)

    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="LEFT_LEFT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_float():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_2.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_2.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_2.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_2.txt"]
    np.random.seed(23457)
    shape_x1 = [5,3,2]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.float32, "randint", 0, 10) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    padding_value = gen_data_file(data_files[2], shape_x3, np.float32, "randint", 0, 10)
    
    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(tf.float32,shape = shape_x3)

    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="RIGHT_LEFT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_double():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_3.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_3.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_3.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x1 = [5,3,2]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.float64, "randint", 0, 10) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    padding_value = gen_data_file(data_files[2], shape_x3, np.float64, "randint", 0, 10)
    
    x1 = tf.compat.v1.placeholder(tf.float64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(tf.float64,shape = shape_x3)

    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="RIGHT_LEFT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_complex64():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_4.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_4.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_4.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_4.txt"]
    np.random.seed(23457)
    shape_x1 = [5,3,2]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.complex64, "randint", 0, 10) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    padding_value = gen_data_file(data_files[2], shape_x3,np.complex64, "randint", 0, 10)
    
    x1 = tf.compat.v1.placeholder(np.complex64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(np.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(np.complex64,shape = shape_x3)

    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="RIGHT_LEFT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_int64():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_5.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_5.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_5.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_5.txt"]
    np.random.seed(23457)
    shape_x1 = [5,3,2]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.int64, "randint", 0, 10) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    padding_value = gen_data_file(data_files[2], shape_x3,np.int64, "randint", 0, 10)
    
    x1 = tf.compat.v1.placeholder(np.int64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(np.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(np.int64,shape = shape_x3)

    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="LEFT_LEFT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_float16():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_6.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_6.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_6.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_6.txt"]
    np.random.seed(23457)
    shape_x1 = [6,3,4]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.float16, "randint", 0, 10) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    padding_value = gen_data_file(data_files[2], shape_x3,np.float16, "randint", 0, 10)  
    x1 = tf.compat.v1.placeholder(np.float16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(np.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(np.float16,shape = shape_x3)

    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="RIGHT_LEFT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_int16():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_7.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_7.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_7.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_7.txt"]
    np.random.seed(23457)
    shape_x1 = [5,6,3,4]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.int16, "randint", 0, 10) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    padding_value = gen_data_file(data_files[2], shape_x3,np.int16, "randint", 0, 10)  
    x1 = tf.compat.v1.placeholder(np.int16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(np.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(np.int16,shape = shape_x3)
    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="RIGHT_RIGHT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_int8():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_8.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_8.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_8.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_8.txt"]
    np.random.seed(23457)
    shape_x1 = [5,6,3,4]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.int8, "randint", 0, 10) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    padding_value = gen_data_file(data_files[2], shape_x3,np.int8, "randint", 0, 10)  
    x1 = tf.compat.v1.placeholder(np.int8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(np.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(np.int8,shape = shape_x3)
    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="LEFT_RIGHT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_uint8():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_9.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_9.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_9.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_9.txt"]
    np.random.seed(23457)
    shape_x1 = [2,6,3,4]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.uint8, "randint", 0, 10) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    padding_value = gen_data_file(data_files[2], shape_x3,np.uint8, "randint", 0, 10)  
    x1 = tf.compat.v1.placeholder(np.uint8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(np.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(np.uint8,shape = shape_x3)
    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="RIGHT_LEFT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_uint16():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_10.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_10.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_10.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_10.txt"]
    np.random.seed(23457)
    shape_x1 = [2,6,3,4]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.uint16, "randint", 0, 10) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    padding_value = gen_data_file(data_files[2], shape_x3,np.uint16, "randint", 0, 10)  
    x1 = tf.compat.v1.placeholder(np.uint16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(np.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(np.uint16,shape = shape_x3)
    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="RIGHT_LEFT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_uint32():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_11.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_11.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_11.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_11.txt"]
    np.random.seed(23457)
    shape_x1 = [2,6,3,4]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.uint32, "randint", 0, 10) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    padding_value = gen_data_file(data_files[2], shape_x3,np.uint32, "randint", 0, 10)  
    x1 = tf.compat.v1.placeholder(np.uint32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(np.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(np.uint32,shape = shape_x3)
    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="RIGHT_LEFT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_uint64():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_12.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_12.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_12.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_12.txt"]
    np.random.seed(23457)
    shape_x1 = [2,6,3,4]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.uint64, "randint", 0, 10) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    padding_value = gen_data_file(data_files[2], shape_x3,np.uint64, "randint", 0, 10)  
    x1 = tf.compat.v1.placeholder(np.uint64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(np.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(np.uint64,shape = shape_x3)
    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="RIGHT_LEFT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def gen_random_data_complex128():
    data_files=["matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_13.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_13.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_13.txt",
                "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_13.txt"]
    np.random.seed(23457)
    shape_x1 = [2,6,3,4]
    shape_x2 = [2]
    shape_x3 = [1]
    input = gen_data_file(data_files[0], shape_x1, np.complex128, "randint", 0, 10) 
    k = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0,1)
    padding_value = gen_data_file(data_files[2], shape_x3,np.complex128, "randint", 0, 10)  
    x1 = tf.compat.v1.placeholder(np.complex128, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(np.int32, shape=shape_x2)
    x3 = tf.compat.v1.placeholder(np.complex128,shape = shape_x3)
    # re =tf.raw_ops.MatrixDiagPartV3(input = x1,k=x2,padding_value = x3[0],align ="RIGHT_LEFT")
    # with tf.compat.v1.Session(config=config('cpu')) as session:
    #     data = session.run(re, feed_dict={x1:input, x2:k,x3:padding_value})
    # write_file_txt(data_files[3], data, fmt="%s")

def run():
    gen_random_data_int32()
    gen_random_data_float()
    gen_random_data_double()
    gen_random_data_complex64()
    gen_random_data_int64()
    gen_random_data_float16()
    gen_random_data_int16()
    gen_random_data_int8()
    gen_random_data_uint8()
    gen_random_data_uint16()
    gen_random_data_uint32()
    gen_random_data_uint64()
    gen_random_data_complex128()
