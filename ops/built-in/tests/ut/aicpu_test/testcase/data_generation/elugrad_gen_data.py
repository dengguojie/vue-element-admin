from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op, dtypes
from binascii import unhexlify


def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')


def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()

#
def read_file_txt_to_bool(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)


def gen_data_file( shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    if dtype == np.float16:
        data = np.array(rand_data, dtype=np.float32)
        data = np.array(data, dtype=np.float16)
    else:
        data = np.array(rand_data, dtype=dtype)
    return data

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config

def gen_random_data_float16():
    data_files=["elugrad/data/elugrad_data_input1_1.txt",
                 "elugrad/data/elugrad_data_input1_2.txt",
                 "elugrad/data/elugrad_data_output_1.txt"]
    np.random.seed(3457)
    shape_x1 = [2, 5, 10]
    a = gen_data_file(shape_x1, np.float16, "uniform",-100,100)
    x1 = tf.compat.v1.placeholder(tf.float16, shape=shape_x1)
    with tf.GradientTape() as tape:
        tape.watch(x1)
        y = tf.nn.elu(x1)
    grads = tape.gradient(y,x1)
    re = tf.raw_ops.EluGrad(gradients = grads,outputs = y)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
        grads_s = session.run(grads,feed_dict={x1:a})
        y_s = session.run(y,feed_dict={x1:a})
    write_file_txt(data_files[0],grads_s, fmt="%s")
    write_file_txt(data_files[1],y_s,fmt="%s")
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float():
    data_files=["elugrad/data/elugrad_data_input2_1.txt",
                 "elugrad/data/elugrad_data_input2_2.txt",
                 "elugrad/data/elugrad_data_output_2.txt"]
    np.random.seed(3457)
    shape_x1 = [2, 5, 10]
    a = gen_data_file(shape_x1, np.float, "uniform",-100,100)
    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    with tf.GradientTape() as tape:
        tape.watch(x1)
        y = tf.nn.elu(x1)
    grads = tape.gradient(y,x1)
    re = tf.raw_ops.EluGrad(gradients = grads,outputs = y)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
        grads_s = session.run(grads,feed_dict={x1:a})
        y_s = session.run(y,feed_dict={x1:a})
    write_file_txt(data_files[0],grads_s, fmt="%s")
    write_file_txt(data_files[1],y_s,fmt="%s")
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_double():
    data_files=["elugrad/data/elugrad_data_input3_1.txt",
                 "elugrad/data/elugrad_data_input3_2.txt",
                 "elugrad/data/elugrad_data_output_3.txt"]
    np.random.seed(3457)
    shape_x1 = [2, 5, 10]
    a = gen_data_file(shape_x1, np.double, "uniform",-100,100)
    x1 = tf.compat.v1.placeholder(tf.double, shape=shape_x1)
    with tf.GradientTape() as tape:
        tape.watch(x1)
        y = tf.nn.elu(x1)
    grads = tape.gradient(y,x1)
    re = tf.raw_ops.EluGrad(gradients = grads,outputs = y)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
        grads_s = session.run(grads,feed_dict={x1:a})
        y_s = session.run(y,feed_dict={x1:a})
    write_file_txt(data_files[0],grads_s, fmt="%s")
    write_file_txt(data_files[1],y_s,fmt="%s")
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float16_elugrad():
    data_files=["elugrad/data/elugrad_data_input4_1.txt",
                 "elugrad/data/elugrad_data_input4_2.txt",
                 "elugrad/data/elugrad_data_output_4.txt"]
    np.random.seed(3457)
    shape_x1 = [12,15,300]
    a = gen_data_file(shape_x1, np.float16, "uniform",-100,100)
    x1 = tf.compat.v1.placeholder(tf.float16, shape=shape_x1)
    with tf.GradientTape() as tape:
        tape.watch(x1)
        y = tf.nn.elu(x1)
    grads = tape.gradient(y,x1)
    re = tf.raw_ops.EluGrad(gradients = grads,outputs = y)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
        grads_s = session.run(grads,feed_dict={x1:a})
        y_s = session.run(y,feed_dict={x1:a})
    write_file_txt(data_files[0],grads_s, fmt="%s")
    write_file_txt(data_files[1],y_s,fmt="%s")
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float_elugrad():
    data_files=["elugrad/data/elugrad_data_input5_1.txt",
                 "elugrad/data/elugrad_data_input5_2.txt",
                 "elugrad/data/elugrad_data_output_5.txt"]
    np.random.seed(3457)
    shape_x1 = [12, 15, 300]
    a = gen_data_file(shape_x1, np.float, "uniform",-100,100)
    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    with tf.GradientTape() as tape:
        tape.watch(x1)
        y = tf.nn.elu(x1)
    grads = tape.gradient(y,x1)
    re = tf.raw_ops.EluGrad(gradients = grads,outputs = y)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
        grads_s = session.run(grads,feed_dict={x1:a})
        y_s = session.run(y,feed_dict={x1:a})
    write_file_txt(data_files[0],grads_s, fmt="%s")
    write_file_txt(data_files[1],y_s,fmt="%s")
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_double_elugrad():
    data_files=["elugrad/data/elugrad_data_input6_1.txt",
                 "elugrad/data/elugrad_data_input6_2.txt",
                 "elugrad/data/elugrad_data_output_6.txt"]
    np.random.seed(3457)
    shape_x1 = [6, 15, 100]
    a = gen_data_file(shape_x1, np.double, "uniform",-100,100)
    x1 = tf.compat.v1.placeholder(tf.double, shape=shape_x1)
    with tf.GradientTape() as tape:
        tape.watch(x1)
        y = tf.nn.elu(x1)
    grads = tape.gradient(y,x1)
    re = tf.raw_ops.EluGrad(gradients = grads,outputs = y)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a})
        grads_s = session.run(grads,feed_dict={x1:a})
        y_s = session.run(y,feed_dict={x1:a})
    write_file_txt(data_files[0],grads_s, fmt="%s")
    write_file_txt(data_files[1],y_s,fmt="%s")
    write_file_txt(data_files[2], data, fmt="%s")

def run():
    gen_random_data_float16()
    gen_random_data_float()
    gen_random_data_double()
    gen_random_data_double_elugrad()