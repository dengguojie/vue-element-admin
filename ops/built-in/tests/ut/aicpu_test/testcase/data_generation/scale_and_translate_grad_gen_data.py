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
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.eager import backprop


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


def gen_random_data(in_shape=[1, 2, 3, 1], out_shape=[1, 4, 6, 1], Kernel_type="lanczos3", Antialias=True):
    if Antialias == True:
        data_files = ["scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" + Kernel_type + "_true.txt",
                      "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" + Kernel_type + "_true.txt",
                      "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" + Kernel_type + "_true.txt",
                      "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" + Kernel_type + "_true.txt",
                      "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" + Kernel_type + "_true.txt",
                      "scale_and_translate_grad/data/scale_and_translate_data_size_" + Kernel_type + "_true.txt", ]
    else:
        data_files = ["scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" + Kernel_type + "_false.txt",
                      "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" + Kernel_type + "_false.txt",
                      "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" + Kernel_type + "_false.txt",
                      "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" + Kernel_type + "_false.txt",
                      "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" + Kernel_type + "_false.txt",
                      "scale_and_translate_grad/data/scale_and_translate_data_size_" + Kernel_type + "_false.txt", ]

    np.random.seed(23457)

    size_input = out_shape[1:3]
    scale_shape = [2]
    translate_shape = [2]

    image_data = np.arange(0, np.prod(in_shape))
    write_file_txt(data_files[1], image_data, fmt="%s")
    image = image_data.reshape(in_shape).astype(np.float32)

    size = np.array(size_input, dtype=np.int32)
    scale = gen_data_file(data_files[2], scale_shape, np.float, "uniform", 0, 1)
    translate = gen_data_file(data_files[3], translate_shape, np.float, "uniform", 0, 1)

    image = constant_op.constant(image, shape=in_shape)
    with backprop.GradientTape() as tape:
        tape.watch(image)
        scale_and_translate_out = image_ops.scale_and_translate(
            image, size, scale=scale, translation=translate, kernel_type=Kernel_type,
            antialias=Antialias)

    grad = tape.gradient(scale_and_translate_out, image)

    with tf.compat.v1.Session():
        grad_data = grad.eval()
    write_file_txt(data_files[0], grad_data, fmt="%s")

    re = gen_image_ops.scale_and_translate_grad(
        grad, image, scale=scale, translation=translate, kernel_type=Kernel_type,
        antialias=Antialias)
    with tf.compat.v1.Session():
        re_data = re.eval()

    write_file_txt(data_files[4], re_data, fmt="%s")


def run():
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="lanczos1", Antialias=True)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="lanczos3", Antialias=True)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="lanczos5", Antialias=True)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="box", Antialias=True)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="gaussian", Antialias=True)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="mitchellcubic", Antialias=True)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="keyscubic", Antialias=True)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="triangle", Antialias=True)

    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="lanczos1", Antialias=False)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="lanczos3", Antialias=False)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="lanczos5", Antialias=False)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="box", Antialias=False)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="gaussian", Antialias=False)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="mitchellcubic", Antialias=False)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="keyscubic", Antialias=False)
    gen_random_data(in_shape=[2, 4, 6, 2], out_shape=[2, 8, 12, 2], Kernel_type="triangle", Antialias=False)
