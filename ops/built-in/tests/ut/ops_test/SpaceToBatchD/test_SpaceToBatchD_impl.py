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

SpaceToBatch ut case
"""
import numpy as np
import tensorflow as tf
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
ut_case = OpUT("SpaceToBatchD", None, None)

case1 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (2, 2, 2, 1), "ori_format": "NHWC"}, #x
                    {"shape": (2, 1, 2, 2, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (2, 2, 2, 1), "ori_format": "NHWC"},
                    2, ((1, 1), (1, 1)),
                    ],
         "case_name": "SpaceToBatchD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (2, 2, 2, 1), "ori_format": "NHWC"}, #x
                    {"shape": (2, 1, 2, 2, 16), "dtype": "int16", "format": "NC1HWC0",
                     "ori_shape": (2, 1, 2, 1), "ori_format": "NHWC"},
                    0, ((1, 1), (1, 1)),
                    ],
         "case_name": "SpaceToBatchD_2",
         "expect": RuntimeError,
         "support_expect": True}

case3 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (2, 2, 2, 1), "ori_format": "NHWC"}, #x
                    {"shape": (2, 1, 2, 2, 16), "dtype": "uint8", "format": "NC1HWC0",
                     "ori_shape": (2, 2, 2, 1), "ori_format": "NHWC"},
                    21, ((1, 1), (1, 1)),
                    ],
         "case_name": "SpaceToBatchD_3",
         "expect": RuntimeError,
         "support_expect": True}

case4 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (2, 2, 2, 1), "ori_format": "NHWC"}, #x
                    {"shape": (2, 1, 2, 2, 16), "dtype": "uint8", "format": "NC1HWC0",
                     "ori_shape": (2, 2, 2, 1), "ori_format": "NHWC"},
                    2, ((-1, 1), (1, 1)),
                    ],
         "case_name": "SpaceToBatch_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "uint64", "format": "NC1HWC0",
                     "ori_shape": (2, 2, 2, 1), "ori_format": "NHWC"}, #x
                    {"shape": (2, 1, 2, 2, 16), "dtype": "uint64", "format": "NC1HWC0",
                     "ori_shape": (2, 2, 2, 1), "ori_format": "NHWC"},
                    2, ((-1, 1), (1, 1)),
                    ],
         "case_name": "SpaceToBatchD_5",
         "expect": RuntimeError,
         "support_expect": True}

case6 = {"params": [{"shape": (2, 2, 2, 16), "dtype": "uint64", "format": "NC1HWC0",
                     "ori_shape": (2, 2, 2, 1), "ori_format": "NHWC"}, #x
                    {"shape": (2, 2, 2, 16), "dtype": "uint64", "format": "NC1HWC0",
                     "ori_shape": (2, 2, 2, 1), "ori_format": "NHWC"},
                    2, ((1, 1), (1, 1)),
                    ],
         "case_name": "SpaceToBatchD_6",
         "expect": RuntimeError,
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case1)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case2)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case3)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case4)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case5)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case6)

def trans_data_to_tf(data_nchwc0):
    """
    trans_data_to_tf
    """
    out_size = data_nchwc0.shape
    nhwc = np.zeros((out_size[0], out_size[-3], out_size[-2], out_size[1]*out_size[-1]), dtype=data_nchwc0.dtype)

    for i in range(out_size[0]):
        for j in range(out_size[1]):
            for k in range(out_size[-1]):
                for h in range(out_size[-3]):
                    for w in range(out_size[-2]):
                        nhwc[i][h][w][j*out_size[-1] + k] = data_nchwc0[i][j][h][w][k]

    return nhwc
def trans_tf_data_out(data_nhwc):
    """
    trans_tf_data_out
    """
    in_size = data_nhwc.shape
    if data_nhwc.dtype == "float16":
        c0 = 16
    else:
        c0 = 16

    nchwc0 = np.zeros((in_size[0], in_size[-1] // c0, in_size[-3], in_size[-2], c0), dtype=data_nhwc.dtype)

    for i in range(in_size[0]):
        for j in range(in_size[-1] // c0):
            for k in range(c0):
                for h in range(in_size[-3]):
                    for w in range(in_size[-2]):
                        nchwc0[i][j][h][w][k] = data_nhwc[i][h][w][j*c0+k]
    return nchwc0

# pylint: disable=unused-argument
def calc_expect_func(x, y, block_size, paddings):
    input_data = x['value']
    input_data = trans_data_to_tf(input_data)
    to_batch = tf.space_to_batch(input_data, np.array(paddings), block_size)
    with tf.Session() as sess:
        out_data = sess.run(to_batch)
    res = trans_tf_data_out(out_data)
    return res

precision_case1 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "float16", "format": "NC1HWC0",
                               "ori_shape": (2, 2, 2, 1), "ori_format": "NHWC", "param_type": "input"}, #x
                              {"shape": (8, 1, 2, 2, 16), "dtype": "float16", "format": "NC1HWC0",
                               "ori_shape": (8, 2, 2, 1), "ori_format": "NHWC", "param_type": "output"},
                              2, ((1, 1), (1, 1))
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_precision_case("Ascend910", precision_case1)
