# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
"""
test_less_golden.py
"""
import tensorflow as tf


# pylint: disable=unused-argument,invalid-name
def less_by_tf(x1, x2):
    """
    less_v2_by_tf
    """
    x1_holder = tf.placeholder(x1.dtype, shape=x1.shape)
    x2_holder = tf.placeholder(x2.dtype, shape=x2.shape)
    re = tf.less(x1_holder, x2_holder)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x1_holder: x1, x2_holder: x2})
    return result


def calc_expect_func(x1, x2, y):
    """
    calc_expect_func
    """
    res = less_by_tf(x1["value"], x2["value"])
    return [res]
