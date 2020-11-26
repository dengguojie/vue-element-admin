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

import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("max_pool_v3_grad")


def data4Dto5D(input_data, C0=16):
    N, C, H, W = input_data.shape
    input_data = input_data.reshape(N, int(C / C0), C0, H, W).transpose(0, 1, 3, 4, 2)
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

    if data_format == "NHWC":
        N, H, W, C = ori_input.get("ori_shape")
        _, out_h, out_w, _ = grad.get("ori_shape")
        h_index, w_index = 1, 2
    else:
        N, C, H, W = ori_input.get("ori_shape")
        _, _, out_h, out_w = grad.get("ori_shape")
        h_index, w_index = 2, 3

    pooling = (ksize[h_index], ksize[w_index])
    strides = (strides[h_index], strides[w_index])

    if global_pooling:
        pads = (0, 0, 0, 0)
        pooling = (H, W)

    padding_z = np.lib.pad(ori_input, ((0, 0), (0, 0), (pads[0], pads[1]), (pads[2], pads[3])), 'constant',
                           constant_values=0)
    padding_dz = np.zeros_like(padding_z)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    flat_idx = np.argmax(padding_z[n, c,
                                         strides[0] * i:strides[0] * i + pooling[0],
                                         strides[1] * j:strides[1] * j + pooling[1]])
                    h_idx = strides[0] * i + flat_idx // pooling[1]
                    w_idx = strides[1] * j + flat_idx % pooling[1]
                    padding_dz[n, c, h_idx, w_idx] += grad[n, c, i, j]
    res = _remove_padding(padding_dz, pads)
    res = res.astype("float16")

    return res


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


ori_input_ori_shape = (1, 64, 56, 56)
ori_input_shape = (1, 4, 56, 56, 16)
ori_output_ori_shape = (1, 64, 28, 28)
ori_output_shape = (1, 4, 28, 28, 16)
ksize = [1, 1, 3, 3]
strides = [1, 1, 2, 2]
pads = [1, 1, 1, 1]

ori_input_value = np.random.uniform(0.1, 1.0, size=ori_input_ori_shape).astype(np.float16)
ori_input = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": ori_input_ori_shape,
             "shape": ori_input_shape, "value": ori_input_value, "param_type": "input"}

ori_output_value = max_pooling_forward(ori_input, None, ksize, strides, "CALCULATED", pads,
                                       "NC1HWC0", False, False).astype(np.float16)

ori_output = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": ori_output_ori_shape,
              "shape": ori_output_shape, "value": ori_output_value, "param_type": "input"}

grad = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": ori_output_ori_shape,
        "shape": ori_output_shape, "value": ori_output_value, "param_type": "input"}

output = {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": ori_input_ori_shape,
          "shape": ori_input_shape, "param_type": "output"}

# coding cases here
ut_case.add_precision_case("all", {
    "params": [ori_input, ori_output, grad, output, ksize, strides, "CALCULATED", pads, "NCHW", False, False],
    "calc_expect_func": max_pooling_backward,
})
