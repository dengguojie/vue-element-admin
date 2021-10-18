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
test_pack_golden.py
"""
import tensorflow as tf


# pylint: disable=unused-argument,invalid-name
def pack_by_tf(x, axis):
    """
    pack_by_tf
    """
    x_holders = [tf.placeholder(item.dtype, shape=item.shape) for item in x]
    feed_dict = dict(zip(x_holders, x))
    re = tf.pack(x_holders, axis)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict=feed_dict)
    return result


def calc_expect_func(*args):
    """
    calc_expect_func
    """
    param_count = len(args)
    res = None
    if param_count > 2:
        input_count = param_count[-1]
        axis = param_count[-2]
        input_values = [x["value"] for x in args[0:input_count]]
        res = pack_by_tf(input_values, axis)
    return [res]

