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
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import image_ops


def write_file_txt(file_name, data, fmt="%s"):
    # prama1: file_name: the file which store the data
    # param2: data: data which will be stored
    # param3: fmt: format
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')


def read_file_txt(file_name, dtype, delim=None):
    # prama1: file_name: the file which store the data
    # param2: dtype: data type
    # param3: delim: delimiter which is used to split data
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()


def read_file_txt_to_boll(file_name, delim=None):
    # prama1: file_name: the file which store the data
    # param2: delim: delimiter which is used to split data
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)


def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    # prama1: data_file: the file which store the generation data
    # param2: shape: data shape
    # param3: dtype: data type
    # param4: rand_type: the method of generate data, select from "randint, uniform"
    # param5: data lower limit
    # param6: data upper limit
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    data = np.array(rand_data, dtype=dtype)
    write_file_txt(data_file, data, fmt="%s")
    return data


def input_data_file(data_file, input_data):
    data = np.array(input_data)
    write_file_txt(data_file, data, fmt="%s")
    return data


def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config


def gen_random_data(in_shape=[1, 2, 3, 1], out_shape=[1, 4, 6, 1], num="float_lanczos3", Dtype=np.float,
                    tfDtype=tf.float32, Kernel_type="lanczos3", Antialias=True):
    data_files = ["scale_and_translate/data/scale_and_translate_data_image_" + num + ".txt",
                  "scale_and_translate/data/scale_and_translate_data_size_" + num + ".txt",
                  "scale_and_translate/data/scale_and_translate_data_scale_" + num + ".txt",
                  "scale_and_translate/data/scale_and_translate_data_translate_" + num + ".txt",
                  "scale_and_translate/data/scale_and_translate_data_output_" + num + ".txt"]
    np.random.seed(23457)

    size_input = out_shape[1:3]
    size_shape = [2]
    scale_shape = [2]
    translate_shape = [2]

    image = gen_data_file(data_files[0], in_shape, Dtype, "uniform", 0, 255)
    size = input_data_file(data_files[1], size_input)
    scale = gen_data_file(data_files[2], scale_shape, np.float, "uniform", 0, 1)
    translate = gen_data_file(data_files[3], translate_shape, np.float, "uniform", 0, 1)

    x1 = tf.compat.v1.placeholder(tfDtype, shape=in_shape)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=size_shape)
    x3 = tf.compat.v1.placeholder(tf.float32, shape=scale_shape)
    x4 = tf.compat.v1.placeholder(tf.float32, shape=translate_shape)

    re = image_ops.scale_and_translate(
        x1, x2, scale=x3, translation=x4, kernel_type=Kernel_type,
        antialias=Antialias)

    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: image, x2: size, x3: scale, x4: translate})
    write_file_txt(data_files[4], data, fmt="%s")


def run():
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], num="int32_lanczos1", Dtype=np.int32,
                    tfDtype=tf.int32, Kernel_type="lanczos1", Antialias=True)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], num="int64_lanczos1", Dtype=np.int64,
                    tfDtype=tf.int64, Kernel_type="lanczos1", Antialias=False)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], num="int16_lanczos3", Dtype=np.int16,
                    tfDtype=tf.int16, Kernel_type="lanczos3", Antialias=False)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], num="uint16_lanczos3", Dtype=np.uint16,
                    tfDtype=tf.uint16, Kernel_type="lanczos3", Antialias=False)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], num="int8_lanczos5", Dtype=np.int8, tfDtype=tf.int8,
                    Kernel_type="lanczos5", Antialias=True)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], num="uint8_lanczos5", Dtype=np.uint8,
                    tfDtype=tf.uint8, Kernel_type="lanczos5", Antialias=True)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], num="float_lanczos3", Dtype=np.float,
                    tfDtype=tf.float32, Kernel_type="lanczos3", Antialias=True)

    gen_random_data(in_shape=[1, 2, 3, 1], out_shape=[1, 4, 6, 1], num="float_box", Dtype=np.float, tfDtype=tf.float32,
                    Kernel_type="box", Antialias=True)
    gen_random_data(in_shape=[1, 2, 3, 1], out_shape=[1, 4, 6, 1], num="float_gaussian", Dtype=np.float,
                    tfDtype=tf.float32, Kernel_type="gaussian", Antialias=True)
    gen_random_data(in_shape=[1, 2, 3, 1], out_shape=[1, 4, 6, 1], num="float_mitchellcubic", Dtype=np.float,
                    tfDtype=tf.float32, Kernel_type="mitchellcubic", Antialias=True)
    gen_random_data(in_shape=[1, 2, 3, 1], out_shape=[1, 4, 6, 1], num="float_keyscubic", Dtype=np.float,
                    tfDtype=tf.float32, Kernel_type="keyscubic", Antialias=True)

    gen_random_data(in_shape=[1, 2, 3, 1], out_shape=[1, 4, 6, 1], num="int32_box", Dtype=np.int32, tfDtype=tf.int32,
                    Kernel_type="box", Antialias=False)
    gen_random_data(in_shape=[1, 2, 3, 1], out_shape=[1, 4, 6, 1], num="int32_gaussian", Dtype=np.int32,
                    tfDtype=tf.int32, Kernel_type="gaussian", Antialias=False)
    gen_random_data(in_shape=[1, 2, 3, 1], out_shape=[1, 4, 6, 1], num="int32_mitchellcubic", Dtype=np.int32,
                    tfDtype=tf.int32, Kernel_type="mitchellcubic", Antialias=False)
    gen_random_data(in_shape=[1, 2, 3, 1], out_shape=[1, 4, 6, 1], num="int32_keyscubic", Dtype=np.int32,
                    tfDtype=tf.int32, Kernel_type="keyscubic", Antialias=False)

    gen_random_data(in_shape=[1, 2, 3, 1], out_shape=[1, 4, 6, 1], num="double_triangle", Dtype=np.double,
                    tfDtype=tf.float64, Kernel_type="triangle", Antialias=False)
    gen_random_data(in_shape=[1, 2, 3, 1], out_shape=[1, 4, 6, 1], num="float16_triangle", Dtype=np.float16,
                    tfDtype=tf.float16, Kernel_type="triangle", Antialias=False)
