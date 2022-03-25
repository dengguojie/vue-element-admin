# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
test_sqrt_golden.py
"""
import tensorflow as tf

# pylint: disable=unused-argument,invalid-name
def slice_by_tf(x):
    """
    slice_by_tf
    """
    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    re = tf.sqrt(x_holder)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x})
    return result


def calc_expect_func(x1, y, impl_mode):
    """
    calc_expect_func
    """
    res = slice_by_tf(x1["value"])
    return [res]