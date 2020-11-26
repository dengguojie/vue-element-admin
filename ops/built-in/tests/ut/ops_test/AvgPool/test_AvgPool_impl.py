#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import tensorflow as tf

ut_case = OpUT("AvgPool", "impl.avg_pool", "avg_pool")

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

def calc_expect_func(x, filter, bias, y, ksize, strides, padding="VALID", data_format="NHWC", offset_x=0):
    x_data = trans_data_to_tf(x['value'])
    x_holder = tf.placeholder(x["dtype"], shape=x_data.shape)

    y = tf.nn.avg_pool(x_holder, ksize, strides, padding, data_format)
    with tf.Session() as sess:
        result = sess.run(y ,feed_dict={x_holder:x_data})
    result = trans_tf_data_out(result)
    return result

case1 = {"params": [{"shape": (1,2,32,32,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,32,32),"ori_format": "NC1HWC0"},
                    {"shape": (2048,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 16, 4, 2),"ori_format": "FRACTAL_Z"},
                    None,
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 31, 31, 32),"ori_format": "NC1HWC0"},
                    [1,2,2,1], [1,1,1,1], "VALID"],
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (13, 1, 2, 768, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (13, 2, 768, 13),"ori_format": "NC1HWC0"},
                    {"shape": (9, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 16, 3, 3),"ori_format": "FRACTAL_Z"},
                    None,
                    {"shape": (13, 1, 2, 768, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (13, 2, 768, 13),"ori_format": "NC1HWC0"},
                    [1,3,3,1], [1,1,1,1], "SAME"],
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (20, 1, 7, 68, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (20, 7, 68, 3),"ori_format": "NC1HWC0"},
                    {"shape": (4, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 16, 2, 2),"ori_format": "FRACTAL_Z"},
                    None,
                    {"shape": (20, 1, 7, 68, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (20, 7, 68, 3),"ori_format": "NC1HWC0"},
                    [1,2,2,1], [1,1,1,1], "SAME"],
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (10, 7, 5, 33, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10, 5, 33, 110),"ori_format": "NC1HWC0"},
                    {"shape": (28, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 16, 14, 2),"ori_format": "FRACTAL_Z"},
                    None,
                    {"shape": (10, 7, 4, 32, 166), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10, 4, 32, 110),"ori_format": "NC1HWC0"},
                    [1,2,2,1], [1,1,1,1], "VALID"],
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (4, 6, 5, 10, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 5, 10, 89),"ori_format": "NC1HWC0"},
                    {"shape": (54, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 16, 18, 2),"ori_format": "FRACTAL_Z"},
                    None,
                    {"shape": (4, 6, 3, 8, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 3, 8, 89),"ori_format": "NC1HWC0"},
                    [1,3,3,1], [1,1,1,1], "VALID"],
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (1, 8, 32, 32, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, 32, 128),"ori_format": "NC1HWC0",
                    "param_type": "input", "value_range": [1.0, 10.0]},
                    None,
                    None,
                    {"shape": (1, 8, 32, 32, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, 32, 128),"ori_format": "NC1HWC0","param_type": "output"},
                    [1,2,2,1], [1,1,1,1], "SAME"],
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case7 = {"params": [{"shape": (3, 4, 16, 16, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (3, 16, 16, 64),"ori_format": "NC1HWC0",
                    "param_type": "input", "value_range": [1.0, 10.0]},
                    None,
                    None,
                    {"shape": (3, 4, 15, 15, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (3, 15, 15, 64),"ori_format": "NC1HWC0","param_type": "output"},
                    [1,2,2,1], [1,1,1,1], "VALID"],
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case8 = {"params": [{"shape": (4, 8, 10, 10, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 10, 10, 128),"ori_format": "NC1HWC0",
                     "param_type": "input", "value_range": [1.0, 10.0]},
                    None,
                    None,
                    {"shape": (4, 8, 8, 8, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 8, 8, 128),"ori_format": "NC1HWC0","param_type": "output"},
                    [1,3,3,1], [1,1,1,1], "VALID"],
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

ut_case.add_precision_case(["Ascend310", "Ascend710", "Ascend910"], case6)

ut_case.add_precision_case(["Ascend310", "Ascend710", "Ascend910"], case7)

ut_case.add_precision_case(["Ascend310", "Ascend710", "Ascend910"], case8)


if __name__ == '__main__':
    ut_case.run()
    exit(0)
