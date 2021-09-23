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
test_strided_slice_d_golden.py
"""
import tensorflow as tf


# pylint: disable=unused-argument,invalid-name
def strided_slice_by_tf(x, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):
    """
    strided_slice_by_tf
    """
    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    re = tf.strided_slice(x_holder, begin, end, strides, begin_mask, end_mask,
                          ellipsis_mask, new_axis_mask, shrink_axis_mask)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: x})
    return result


def calc_expect_func(x, y, begin, end, strides,
                     begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    """
    calc_expect_func
    """
    res = strided_slice_by_tf(x["value"], begin["value"], end["value"], strides["value"], begin_mask, end_mask,
                              ellipsis_mask, new_axis_mask, shrink_axis_mask)
    return [res]
