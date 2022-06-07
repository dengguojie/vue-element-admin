# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
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


def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config


def gen_random_data(data_files, shape, dtype):
    data = tf.raw_ops.RandomStandardNormal(
        shape=shape, dtype=dtype, seed=10, seed2=5, name=None
    )

    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(data)
    write_file_txt(data_files, data, fmt="%s")


def run():
    data_files = [
        "random_standard_normal/data/random_standard_normal_data_output_float.txt",
        "random_standard_normal/data/random_standard_normal_data_output_float_1024.txt",
        "random_standard_normal/data/random_standard_normal_data_output_double.txt",
        "random_standard_normal/data/random_standard_normal_data_output_half.txt",
    ]
    gen_random_data(data_files[0], shape=[100, 100], dtype=tf.float32)
    gen_random_data(data_files[1], shape=[1, 1024], dtype=tf.float32)
    gen_random_data(data_files[2], shape=[100, 100], dtype=tf.double)
    gen_random_data(data_files[3], shape=[100, 100], dtype=tf.half)


if __name__ == "__main__":
    run()
