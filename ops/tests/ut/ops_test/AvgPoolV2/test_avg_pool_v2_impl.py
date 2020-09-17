# # -*- coding:utf-8 -*-
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
test code for avg_pool_v2
"""
import sys
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = BroadcastOpUT("avg_pool_v2")


def NCHW2NC1HWC0(fmi, fmo_shape, precise):
    if precise == "int8":
        fmo = np.zeros((fmo_shape[0], fmo_shape[1], fmo_shape[2], fmo_shape[3], fmo_shape[4]), dtype=np.int8)
    else:
        fmo = np.zeros((fmo_shape[0], fmo_shape[1], fmo_shape[2], fmo_shape[3], fmo_shape[4]), dtype=np.float16)
    for n in range(fmo_shape[0]):
        for c1 in range(fmo_shape[1]):
            for h in range(fmo_shape[2]):
                for w in range(fmo_shape[3]):
                    for c0 in range(fmo_shape[4]):
                        try:
                            fmo[n][c1][h][w][c0] = fmi[n][c1*fmo_shape[4]+c0][h][w]
                        except:
                            fmo[n][c1][h][w][c0] =0
    return fmo


def NC1HWC02NCHW(fmi, fmi_shape, precise, shape_orig):
    if precise == "int8":
        fmo = np.zeros((fmi_shape[0], fmi_shape[1]*fmi_shape[4], fmi_shape[2], fmi_shape[3]), dtype=np.int8)
    else:
        fmo = np.zeros((fmi_shape[0], shape_orig[1], fmi_shape[2], fmi_shape[3]), dtype=np.float16)
    for n in range(fmi_shape[0]):
        for c1 in range(fmi_shape[1]):
            for h in range(fmi_shape[2]):
                for w in range(fmi_shape[3]):
                    for c0 in range(fmi_shape[4]):
                        if c1*fmi_shape[4]+c0 < shape_orig[1]:
                            fmo[n][c1*fmi_shape[4]+c0][h][w] = fmi[n][c1][h][w][c0]
    return fmo


def avg_pool_forward(z, pooling, strides=(1,1), padding=(0,0,0,0), global_pooling=True, ceil_mode=False, exclusive=True):
    N, C, H, W = z.shape
    if global_pooling:
        padding = (0, 0, 0, 0)
        pooling = (H, W)

    padding_z = np.lib.pad(z, ((0,0), (0,0), (padding[0], padding[1]), (padding[2], padding[3])), 'constant', constant_values=0)

    if ceil_mode:
        out_h = (H + padding[0] + padding[1] - pooling[0] + strides[0] - 1) // strides[0] + 1
        out_w = (W + padding[2] + padding[3] - pooling[1] + strides[1] - 1) // strides[1] + 1
    else:
        out_h = (H + padding[0] + padding[1] - pooling[0]) // strides[0] + 1
        out_w = (W + padding[2] + padding[3] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    window = padding_z[n, c, strides[0]*i:strides[0]*i+pooling[0], strides[1]*j:strides[1]*j+pooling[1]]
                    window_size = pooling[0] * pooling[1]
                    nonzero_size = np.count_nonzero(window)
                    mean = np.mean(window)
                    if exclusive:
                        mean = mean * window_size / nonzero_size
                    pool_z[n, c, i, j] = mean
    return pool_z

# [TODO] coding expect function here
def calc_expect_func(input_x, output_z, ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive):
    train_data = NC1HWC02NCHW(input_x['value'], input_x['shape'], "float16", input_x['ori_shape'])
    # print(train_data)
    outs = avg_pool_forward(train_data, (ksize[-2],ksize[-1]), (strides[-2], strides[-1]),
                            pads, global_pooling, ceil_mode, exclusive)
    # print(outs)
    res = NCHW2NC1HWC0(outs, output_z['shape'], "float16")
    return [res, ]


# [TODO] coding cases here
ut_case.add_precision_case(
    "all",
    {"params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1,2048,1,1), "shape": (1,128,1,1,16),
                 "param_type": "input", "addr_type": 0, "valid_shape": (), "slice_offset": (), "L1_workspace_size": -1,
                 "L1_fusion_type": -1, "L1_addr_offset": 0, "total_shape": (), "split_index": 0},
                {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1,2048,1,1), "shape": (1,128,1,1,16),
                 "param_type": "output"},
                (1,1,7,7),
                (1,1,1,1),
                "CALCULATED",
                [0,0,0,0],
                "NCHW",
                True,False,True],
     "calc_expect_func": calc_expect_func,
     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})

