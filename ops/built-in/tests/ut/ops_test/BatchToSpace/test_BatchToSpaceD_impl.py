"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

BatchToSpaceD ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf
ut_case = OpUT("BatchToSpaceD", None, None)

case1 = {"params": [{"shape": (288, 128, 6, 8, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 128, 6, 8, 16),"ori_format": "NHWC"}, #x
                    {"shape": (2, 128, 48, 72, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 128, 48, 72, 16),"ori_format": "NHWC"},
                    12,[[12, 12], [12, 12]],
                    ],
         "case_name": "BatchToSpaceD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (338, 128, 6, 8, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (338, 128, 6, 8, 16),"ori_format": "NHWC"}, #x
                    {"shape": (2, 128, 54, 80, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 128, 54, 80, 16),"ori_format": "NHWC"},
                    13,[[12, 12], [12, 12]],
                    ],
        "case_name": "BatchToSpaceD_2",
        "expect": "success",
        "support_expect": True}

case3 = {"params": [{"shape": (40,2,5,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape":(40,2,5,4,16),"ori_format": "NCHW"}, #x
                    {"shape": (10,2,6,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10,2,6,4,16),"ori_format": "NCHW"},
                    2,[[1, 3], [1, 3]],
                    ],
        "case_name": "BatchToSpaceD_3",
        "expect": "success",
        "support_expect": True}

case4 = {"params": [{"shape": (288, 2, 6, 4000, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 2, 6, 4000, 16),"ori_format": "NCHW"}, #x
                    {"shape": (288, 2, 6, 4000, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 2, 6, 4000, 16),"ori_format": "NCHW"},
                    12, [[-1, -1], [-1, -1]],
                    ],
        "case_name": "BatchToSpaceD_4",
        "expect": RuntimeError,
        "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)

#precision cases
def NCHW2NC1HWC0(fmi, fmo_shape, precise):
    if precise=='int32':
        fmo = np.zeros((fmo_shape[0], fmo_shape[1], fmo_shape[2], fmo_shape[3], fmo_shape[4]), dtype=np.int32)
    else:
        fmo = np.zeros((fmo_shape[0], fmo_shape[1], fmo_shape[2], fmo_shape[3], fmo_shape[4]), dtype=np.float16)
    for n in range(fmo_shape[0]):
        for c1 in range(fmo_shape[1]):
            for h in range(fmo_shape[2]):
                for w in range(fmo_shape[3]):
                    for c0 in range(fmo_shape[4]):
                        fmo[n][c1][h][w][c0] = fmi[n][h][w][c1*fmo_shape[4]+c0]
    return fmo

def compute(input_shape, block_size, input_crops):
    crops_shape = (input_shape[0],
                   input_shape[1],
                   input_shape[2]* block_size - input_crops[0][0] -
                   input_crops[0][1],
                   input_shape[3] * block_size - input_crops[1][0] -
                   input_crops[1][1],
                   input_shape[4])

    output_shape = (input_shape[0] // block_size // block_size,
                    input_shape[1],
                    crops_shape[2],
                    crops_shape[3],
                    input_shape[4])
    return output_shape

def trans_data_to_tf(data_nchwc0):
    out_size = data_nchwc0.shape
    nhwc = np.zeros((out_size[0],out_size[-3], out_size[-2], out_size[1]*out_size[-1]), dtype=data_nchwc0.dtype)

    for i in range(out_size[0]):
        for j in range(out_size[1]):
            for k in range(out_size[-1]):
                for h in range(out_size[-3]):
                    for w in range(out_size[-2]):
                        nhwc[i][h][w][j*out_size[-1] + k] = data_nchwc0[i][j][h][w][k]

    return nhwc

def trans_tf_data_out(data_nhwc):
    in_size = data_nhwc.shape
    if data_nhwc.dtype == "float16":
        c0 = 16
    else:
        c0 = 16

    nchwc0  = np.zeros((in_size[0], in_size[-1] // c0, in_size[-3], in_size[-2], c0), dtype=data_nhwc.dtype)

    for i in range(in_size[0]):
        for j in range(in_size[-1] // c0):
            for k in range(c0):
                for h in range(in_size[-3]):
                    for w in range(in_size[-2]):
                        nchwc0[i][j][h][w][k] = data_nhwc[i][h][w][j*c0+k]
    return nchwc0

def calc_expect_func(x, y, block_size, crops):
    data_input = x['value']
    data_input = trans_data_to_tf(data_input)
    input_data = data_input.shape
    batch_to_space = tf.batch_to_space(data_input, crops, block_size)
    with tf.Session() as sess:
        tensorresult=sess.run(batch_to_space)
    
    return trans_tf_data_out(tensorresult)

ut_case.add_precision_case("all", {"params": [{"shape": (40,2,5,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (40,2,5,4,16),"ori_format": "NCHW", "param_type": "input","value_range":[-10,10]},
                                              {"shape": (10,2,6,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10,2,6,4,16),"ori_format": "NCHW", "param_type": "output"},
                                              2,[[1, 3], [1, 3]]],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
