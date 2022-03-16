# -*- coding:utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
test_max_pool_v3_grad_impl.py
"""
import sys
import math
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("max_pool_v3_grad")


def data4Dto5D(input_data, C0, data_format):
    if data_format == "NHWC":
        N, H, W, C = input_data
    else:
        N, C, H, W = input_data
    C1 = math.ceil(C / C0)
    input_data = (N, C1, H, W, C0)
    return input_data


def data5Dto4D(input_data):
    input_data = input_data.get("value")
    f_temp = np.shape(input_data)
    f = [f_temp[0], f_temp[1] * f_temp[4], f_temp[2], f_temp[3]]
    output_data = np.zeros(f)
    for i in range(f_temp[0]):
        for j in range(f_temp[1]):
            for k in range(f_temp[4]):
                output_data[i, j * f_temp[4] + k, :, :] = input_data[i, j, :, :, k]
    return output_data


def _remove_padding(z, padding):
    """
    :param z: (N,C,H,W)
    :param paddings: (p1,p2)
    :return:
    """
    if padding[0] > 0 and padding[1] > 0:
        return z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0] > 0:
        return z[:, :, padding[0]:-padding[0], :]
    elif padding[1] > 0:
        return z[:, :, :, padding[1]:-padding[1]]
    else:
        return z


def max_pooling_backward(ori_input,
                         ori_output,
                         grad,
                         y,
                         ksize,
                         strides,
                         padding="CALCULATED",
                         pads=(0, 0, 0, 0),
                         data_format="NCHW",
                         global_pooling=False,
                         ceil_mode=False,
                         kernel_name="max_pool_v3_grad"):
    """
    :param next_dz dy
    :param z:input
    :param pooling: kernel
    :param strides: strides
    :param padding: pads
    :return:
    """
    result = ori_input.get("value")
    result = result.astype("float16")

    return result


# coding expect function here
def max_pooling_forward(input_data, output_data, ksize, strides, padding_mode="CALCULATED", pads=(0, 0, 0, 0),
                        data_format="NCHW", global_pooling=False, ceil_mode=False, kernel_name="max_pool_v3"):
    # def max_pooling_forward(z, pooling, strides=(2, 2), padding=(0, 0), global_pooling=False, ceil_mode=False):
    """
    :param z: input
    :param pooling: kernel
    :param strides: strides
    :param padding: pads
    :return:
    """
    if data_format == "NHWC":
        N, H, W, C = input_data.get('ori_shape')
        h_index, w_index = 1, 2
    else:
        N, C, H, W = input_data.get('ori_shape')
        h_index, w_index = 2, 3
    input_data = input_data.get("value")
    pooling = (ksize[h_index], ksize[w_index])
    strides = (strides[h_index], strides[w_index])

    if global_pooling:
        pads = (0, 0, 0, 0)
        pooling = (H, W)
    padding_z = np.lib.pad(input_data, ((0, 0), (0, 0), (pads[0], pads[1]), (pads[2], pads[3])), 'constant',
                           constant_values=0)
    if ceil_mode:
        out_h = (H + pads[0] + pads[1] - pooling[0] + strides[0] - 1) // strides[0] + 1
        out_w = (W + pads[2] + pads[3] - pooling[1] + strides[1] - 1) // strides[1] + 1
    else:
        out_h = (H + pads[0] + pads[1] - pooling[0]) // strides[0] + 1
        out_w = (W + pads[2] + pads[3] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.max(padding_z[n, c,
                                                strides[0] * i:strides[0] * i + pooling[0],
                                                strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z
ori_input_value_1 = np.random.uniform(0.1, 1.0, size=(1, 64, 4, 4)).astype(np.float16)
ori_input_1 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 64, 4, 4),
             "shape": (1, 4, 4, 4, 16), "value": ori_input_value_1, "param_type": "input"}

ori_output_value_1 = max_pooling_forward(ori_input_1, None, [1, 1, 1, 1], [1, 1, 1, 1], "CALCULATED", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_1 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 64, 4, 4),
              "shape": (1, 4, 4, 4, 16), "value": ori_output_value_1, "param_type": "input"}

grad_1 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 64, 4, 4),
        "shape": (1, 4, 4, 4, 16), "value": ori_output_value_1, "param_type": "input"}

output_1 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 64, 4, 4),
          "shape": (1, 4, 4, 4, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_1, ori_output_1, grad_1, output_1, [1, 1, 1, 1], [1, 1, 1, 1], "CALCULATED", [0, 0, 0, 0], "NCHW", False, False]     
 })

ori_input_value_4 = np.random.uniform(0.1, 1.0, size=(32, 32, 32, 16)).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 32, 32, 16),
             "shape": (32, 1, 32, 32, 16), "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 11, 11, 16),
              "shape": (32, 1, 11, 11, 16), "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 11, 11, 16),
        "shape": (32, 1, 11, 11, 16), "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32, 11, 11, 16),
          "shape": (32, 1, 11, 11, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0], "NHWC", False, False]
 })

ori_input_value_4 = np.random.uniform(0.1, 1.0, size=(16, 3, 3, 16)).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (16, 3, 3, 16),
             "shape": (16, 1, 3, 3, 16), "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, [1, 3, 3, 1], [1, 2, 2, 1], "VALID", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (16, 1, 1, 16),
              "shape": (16, 1, 1, 1, 16), "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (16, 1, 1, 16),
        "shape": (16, 1, 1, 1, 16), "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 1, 1, 16),
          "shape": (16, 1, 1, 1, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, [1, 3, 3, 1], [1, 2, 2, 1], "VALID", [0, 0, 0, 0], "NHWC", False, False]
 })
#aaa
ori_input_value_4 = np.random.uniform(0.1, 1.0, size=(32, 3, 3, 48)).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 3, 3, 48),
             "shape": (32, 3, 3, 3, 16), "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 1, 1, 48),
              "shape": (32, 3, 1, 1, 16), "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 1, 1, 48),
        "shape": (32, 3, 1, 1, 16), "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32, 1, 1, 48),
          "shape": (32, 3, 1, 1, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0], "NHWC", False, False]
 })


# tiling_do_ho
ori_input_value_4 = np.random.uniform(0.1, 1.0, size=(32, 64, 64, 16)).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 64, 64, 16),
             "shape": (32, 1, 64, 64, 16), "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 21, 21, 16),
              "shape": (32, 1, 21, 21, 16), "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 21, 21, 16),
        "shape": (32, 1, 21, 21, 16), "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32, 21, 21, 16),
          "shape": (32, 1, 21, 21, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0], "NHWC", False, False]
 })
 
#tiling_do_ho_wo
ori_input_value_4 = np.random.uniform(0.1, 1.0, size=(32, 16, 1280, 16)).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 16, 1280, 16),
             "shape": (32, 1, 16, 1280, 16), "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 8, 640, 16),
              "shape": (32, 1, 8, 640, 16), "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 8, 640, 16),
        "shape": (32, 1, 8, 640, 16), "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32, 8, 640, 16),
          "shape": (32, 1, 8, 640, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", [0, 0, 0, 0], "NHWC", False, False]
 })
 
# tiling_do
ori_input_value_4 = np.random.uniform(0.1, 1.0, size=(16, 3, 3, 16)).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (16, 3, 3, 16),
             "shape": (16, 1, 3, 3, 16), "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (16, 1, 1, 16),
              "shape": (16, 1, 1, 1, 16), "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (16, 1, 1, 16),
        "shape": (16, 1, 1, 1, 16), "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 1, 1, 16),
          "shape": (16, 1, 1, 1, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0], "NHWC", False, False]
 })
 
# tiling_do_ho
ori_input_value_4 = np.random.uniform(0.1, 1.0, size=(16, 64, 64, 16)).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (16, 64, 64, 16),
             "shape": (16, 1, 64, 64, 16), "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (16, 21, 21, 16),
              "shape": (16, 1, 21, 21, 16), "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (16, 21, 21, 16),
        "shape": (16, 1, 21, 21, 16), "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 21, 21, 16),
          "shape": (16, 1, 21, 21, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0], "NHWC", False, False],
 })

# tiling_do_ho_wo
ori_input_value_4 = np.random.uniform(0.1, 1.0, size=(16, 16, 1280, 16)).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (16, 16, 1280, 16),
             "shape": (16, 1, 16, 1280, 16), "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (16, 5, 427, 16),
              "shape": (16, 1, 5, 427, 16), "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (16, 5, 427, 16),
        "shape": (16, 1, 5, 427, 16), "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 5, 427, 16),
          "shape": (16, 1, 5, 427, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0], "NHWC", False, False]
 })
 
# SPECIAL SPLIT CORE
ori_input_value_4 = np.random.uniform(0.1, 1.0, size=(1, 13, 4, 16)).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (1, 13, 4, 16),
             "shape": (1, 1, 13, 4, 16), "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (1, 4, 1, 16),
              "shape": (1, 1, 4, 1, 16), "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (1, 4, 1, 16),
        "shape": (1, 1, 4, 1, 16), "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4, 1, 16),
          "shape": (1, 1, 4, 1, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, [1, 2, 2, 1], [1, 3, 3, 1], "VALID", [0, 0, 0, 0], "NHWC", False, False]
 })
 
# SAME:SPLIT DIFFERENT AXIS AS CORE
ori_input_value_4 = np.random.uniform(0.1, 1.0, size=(32, 3, 3, 16)).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 3, 3, 16),
             "shape": (32, 1, 3, 3, 16), "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, [1, 2, 2, 1], [1, 3, 3, 1], "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 1, 1, 16),
              "shape": (32, 1, 1, 1, 16), "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (32, 1, 1, 16),
        "shape": (32, 1, 1, 1, 16), "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32, 1, 1, 16),
          "shape": (32, 1, 1, 1, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, [1, 2, 2, 1], [1, 3, 3, 1], "SAME", [0, 0, 0, 0], "NHWC", False, False]
 })

ori_input_value_4 = np.random.uniform(0.1, 1.0, size=(1, 64, 200000, 1)).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 64, 200000, 1),
             "shape": (1, 4, 200000, 1, 16), "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, [1, 1, 1, 1], [1, 1, 1, 1], "CALCULATED", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 64, 200000, 1),
              "shape": (1, 4, 200000, 1, 16), "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 64, 200000, 1),
        "shape": (1, 4, 200000, 1, 16), "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 64, 200000, 1),
          "shape": (1, 4, 200000, 1, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, [1, 1, 1, 1], [1, 1, 1, 1], "CALCULATED", [0, 0, 0, 0], "NCHW", False, False]
 })

ori_input_value_4 = np.random.uniform(0.1, 1.0, size=(1, 64, 200000, 1)).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 64, 200000, 1),
             "shape": (1, 4, 200000, 1, 16), "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, [1, 1, 1, 1], [1, 1, 1, 1], "CALCULATED", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 64, 200000, 1),
              "shape": (1, 4, 200000, 1, 16), "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 64, 200000, 1),
        "shape": (1, 4, 200000, 1, 16), "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 64, 200000, 1),
          "shape": (1, 4, 200000, 1, 16), "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, [1, 1, 1, 1], [1, 1, 1, 1], "CALCULATED", [0, 0, 0, 0], "NCHW", False, True]
 })

ori_shape0 = (47, 72, 51, 42*16)
ori_shape1 = (47, 1, 1, 42*16)
ksize = [1, 13, 19, 1]
strides = [1, 60, 39, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })


ori_shape0 = [283, 50, 41, 11]
ori_shape1 = [283, 1, 1, 11]
ksize = [1, 7, 3, 1]
strides = [1, 53, 55, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })


ori_shape0 = [28, 78, 48, 9]
ori_shape1 = [28, 1, 1, 9]
ksize = [1, 25, 5, 1]
strides = [1, 54, 45, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [298, 46, 104, 7]
ori_shape1 = [298, 1, 2, 7]
ksize = [1, 2, 4, 1]
strides = [1, 56, 58, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [2, 352, 274, 94]
ori_shape1 = [2, 6, 6, 94]
ksize = [1, 28, 1, 1]
strides = [1, 57, 47, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [2, 367, 362, 15]
ori_shape1 = [2, 8, 7, 15]
ksize = [1, 3, 3, 1]
strides = [1, 51, 53, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [31, 15, 115, 64]
ori_shape1 = [31, 1, 3, 64]
ksize = [1, 2, 11, 1]
strides = [1, 48, 56, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [322, 68, 62, 3]
ori_shape1 = [322, 2, 2, 3]
ksize = [1, 16, 9, 1]
strides = [1, 47, 59, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [339, 23, 73, 107]
ori_shape1 = [339, 1, 2, 107]
ksize = [1, 19, 10, 1]
strides = [1, 54, 59, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [379, 98, 3, 94]
ori_shape1 = [379, 2, 1, 94]
ksize = [1, 17, 2, 1]
strides = [1, 59, 63, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}
"""
ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })
"""
ori_shape0 = [38, 167, 5, 156]
ori_shape1 = [38, 4, 1, 156]
ksize = [1, 12, 5, 1]
strides = [1, 52, 61, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}
"""
ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })
"""
ori_shape0 = [39, 23, 249, 84]
ori_shape1 = [39, 1, 23, 84]
ksize = [1, 14, 10, 1]
strides = [1, 42, 11, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [3, 13, 69, 146]
ori_shape1 = [3, 1, 2, 146]
ksize = [1, 13, 2, 1]
strides = [1, 59, 61, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })


ori_shape0 = [43, 10, 33, 30]
ori_shape1 = [43, 1, 1, 30]
ksize = [1, 7, 20, 1]
strides = [1, 62, 56, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [43, 40, 7, 154]
ori_shape1 = [43, 1, 1, 154]
ksize = [1, 14, 1, 1]
strides = [1, 49, 63, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [4, 33, 91, 3]
ori_shape1 = [4, 1, 2, 3]
ksize = [1, 21, 6, 1]
strides = [1, 57, 53, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [4, 39, 74, 336]
ori_shape1 = [4, 1, 2, 336]
ksize = [1, 6, 3, 1]
strides = [1, 55, 52, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [4, 41, 323, 171]
ori_shape1 = [4, 2, 11, 171]
ksize = [1, 8, 30, 1]
strides = [1, 25, 30, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [4, 56, 56, 75]
ori_shape1 = [4, 1, 5, 75]
ksize = [1, 8, 12, 1]
strides = [1, 51, 10, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [4, 87, 282, 3]
ori_shape1 = [4, 2, 5, 3]
ksize = [1, 12, 1, 1]
strides = [1, 44, 63, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [52, 59, 83, 128]
ori_shape1 = [52, 2, 2, 128]
ksize = [1, 9, 8, 1]
strides = [1, 48, 59, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [59, 10, 139, 248]
ori_shape1 = [59, 1, 3, 248]
ksize = [1, 4, 30, 1]
strides = [1, 53, 60, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [60, 45, 166, 112]
ori_shape1 = [60, 1, 3, 112]
ksize = [1, 30, 1, 1]
strides = [1, 58, 60, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [60, 50, 87, 68]
ori_shape1 = [60, 1, 8, 68]
ksize = [1, 3, 9, 1]
strides = [1, 63, 12, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [63, 76, 8, 59]
ori_shape1 = [63, 2, 1, 59]
ksize = [1, 12, 5, 1]
strides = [1, 55, 60, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}
"""
ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })
"""
ori_shape0 = [64, 284, 3, 1]
ori_shape1 = [64, 5, 1, 1]
ksize = [1, 17, 1, 1]
strides = [1, 57, 48, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}
"""
ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })
"""
ori_shape0 = [67, 45, 29, 114]
ori_shape1 = [67, 1, 1, 114]
ksize = [1, 18, 5, 1]
strides = [1, 60, 60, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [70, 360, 8, 278]
ori_shape1 = [70, 6, 1, 278]
ksize = [1, 15, 3, 1]
strides = [1, 58, 56, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}
"""
ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })
"""
ori_shape0 = [72, 9, 131, 9]
ori_shape1 = [72, 1, 3, 9]
ksize = [1, 3, 23, 1]
strides = [1, 48, 56, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [73, 85, 34, 1]
ori_shape1 = [73, 2, 5, 1]
ksize = [1, 22, 3, 1]
strides = [1, 33, 7, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [76, 124, 306, 13]
ori_shape1 = [76, 3, 5, 13]
ksize = [1, 19, 6, 1]
strides = [1, 59, 62, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [77, 22, 220, 3]
ori_shape1 = [77, 1, 5, 3]
ksize = [1, 21, 12, 1]
strides = [1, 63, 47, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [7, 29, 90, 24]
ori_shape1 = [7, 1, 2, 24]
ksize = [1, 23, 7, 1]
strides = [1, 63, 57, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [85, 62, 24, 83]
ori_shape1 = [85, 1, 1, 83]
ksize = [1, 29, 4, 1]
strides = [1, 62, 56, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [86, 14, 155, 118]
ori_shape1 = [86, 1, 4, 118]
ksize = [1, 3, 10, 1]
strides = [1, 59, 46, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [8, 103, 167, 2]
ori_shape1 = [8, 2, 4, 2]
ksize = [1, 5, 24, 1]
strides = [1, 55, 51, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [8, 268, 144, 153]
ori_shape1 = [8, 5, 3, 153]
ksize = [1, 12, 10, 1]
strides = [1, 54, 61, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [8, 79, 190, 35]
ori_shape1 = [8, 1, 5, 35]
ksize = [1, 30, 2, 1]
strides = [1, 61, 45, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [96, 15, 154, 54]
ori_shape1 = [96, 1, 3, 54]
ksize = [1, 1, 10, 1]
strides = [1, 49, 63, 1]
paddings = "SAME"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

ori_shape0 = [14, 13, 13, 34]
ori_shape1 = [14, 1, 1, 34]
ksize = [1, 13, 13, 1]
strides = [1, 12, 51, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}
"""
ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })
"""
ori_shape0 = [24, 5, 5, 5]
ori_shape1 = [24, 1, 1, 5]
ksize = [1, 5, 5, 1]
strides = [1, 22, 46, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}
"""
ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })
"""
ori_shape0 = [22, 64, 45, 67*16]
ori_shape1 = [22, 1, 4, 67*16]
ksize = [1, 42, 6, 1]
strides = [1, 32, 12, 1]
paddings = "VALID"
data_format = "NHWC"
ori_shape0_5 = data4Dto5D(ori_shape0, 16, data_format)
ori_shape1_5 = data4Dto5D(ori_shape1, 16, data_format)
ori_input_value_4 = np.random.uniform(0.1, 1.0, ori_shape0).astype(np.float16)
ori_input_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape0,
             "shape": ori_shape0_5, "value": ori_input_value_4, "param_type": "input"}

ori_output_value_4 = max_pooling_forward(ori_input_4, None, ksize, strides, "SAME", [0, 0, 0, 0],
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
              "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

grad_4 = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": ori_shape1,
        "shape": ori_shape1_5, "value": ori_output_value_4, "param_type": "input"}

output_4 = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_shape1,
          "shape": ori_shape1_5, "param_type": "output"}

ut_case.add_case(["Ascend910A"], case={
     "params": [ori_input_4, ori_output_4, grad_4, output_4, ksize, strides, paddings, [0, 0, 0, 0], data_format, False, False]
 })

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    # ut_case.run()
    exit(0)