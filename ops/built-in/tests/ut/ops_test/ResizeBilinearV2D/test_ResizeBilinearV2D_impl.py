#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf

ut_case = OpUT("ResizeBilinearV2D", None, None)

case1 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (2, 2), False, False],
         "case_name": "resize_bilinear_v2_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 2), False, True],
         "case_name": "resize_bilinear_v2_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (25, 2), False, False],
         "case_name": "resize_bilinear_v2_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (1, 1), True, False],
         "case_name": "resize_bilinear_v2_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    (2, 2), False, False],
         "case_name": "resize_bilinear_v2_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (5,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (5,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (5,3,10,10,16), "dtype": "float32", "format": "NHWC", "ori_shape": (5,3,10,10,16),"ori_format": "NHWC"},
                    (10, 10), False, True],
         "case_name": "resize_bilinear_v2_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (2,260,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2, 3, 2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    (257, 10), False, False],
         "case_name": "resize_bilinear_v2_d_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)

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

def calc_expect_func(image, out, size):
    data_in = trans_data_to_tf(image['value'])
    x =  tf.placeholder(image['dtype'], shape=data_in.shape)
    y =  tf.placeholder("int32", shape=(2,))
    z =  tf.image.resize_bilinear(x, y, False)
    
    with tf.Session() as sess:
        res = sess.run(z, feed_dict={x: data_in, y: np.array(size)})
    res = trans_tf_data_out(res)
    return res

ut_case.add_precision_case("all", {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "ND", "ori_shape": (2,3,1,1,16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2,3,2,2,16), "dtype": "float32", "format": "ND", "ori_shape": (2,3,2,2,16),"ori_format": "ND", "param_type": "output"},
                                              (2,2)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })



