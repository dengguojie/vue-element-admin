# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unlog required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
test_log_golden.py
"""
import tensorflow as tf


# pylint: disable=unused-argument,invalid-name
def log_by_tf(x):
    """
    log_v2_by_tf
    """
    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    re = tf.log(x_holder)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x})
    return result


def calc_expect_func(x, y, base, scale, shift):
    """
    calc_expect_func
    """
    res = log_by_tf(x["value"])
    return [res]
