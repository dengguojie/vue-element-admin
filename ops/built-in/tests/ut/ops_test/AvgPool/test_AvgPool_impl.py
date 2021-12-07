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

case1 = {"params": [{"shape": (1,2,32,32,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,32,32),"ori_format": "NHWC"},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW"},
                    None,
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 31, 31, 32),"ori_format": "NHWC"},
                    [1,2,2,1], [1,1,1,1], "VALID"],
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (13, 1, 2, 768, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (13, 2, 768, 13),"ori_format": "NHWC"},
                    {"shape": (9, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (13, 1, 3, 3),"ori_format": "NCHW"},
                    None,
                    {"shape": (13, 1, 2, 768, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (13, 2, 768, 13),"ori_format": "NHWC"},
                    [1,3,3,1], [1,1,1,1], "SAME"],
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (20, 1, 7, 68, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (20, 7, 68, 3),"ori_format": "NHWC"},
                    {"shape": (4, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (3, 1, 2, 2),"ori_format": "NCHW"},
                    None,
                    {"shape": (20, 1, 7, 68, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (20, 7, 68, 3),"ori_format": "NHWC"},
                    [1,2,2,1], [1,1,1,1], "SAME"],
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (10, 7, 5, 33, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10, 5, 33, 110),"ori_format": "NHWC"},
                    {"shape": (28, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (110, 1, 2, 2),"ori_format": "NCHW"},
                    None,
                    {"shape": (10, 7, 4, 32, 166), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10, 4, 32, 110),"ori_format": "NHWC"},
                    [1,2,2,1], [1,1,1,1], "VALID"],
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (4, 6, 5, 10, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 5, 10, 89),"ori_format": "NHWC"},
                    {"shape": (54, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (89, 1, 3, 3),"ori_format": "NCHW"},
                    None,
                    {"shape": (4, 6, 3, 8, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 3, 8, 89),"ori_format": "NHWC"},
                    [1,3,3,1], [1,1,1,1], "VALID"],
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (1, 8, 32, 32, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, 32, 128),"ori_format": "NHWC",
                    "param_type": "input", "value_range": [1.0, 10.0]},
                    None,
                    None,
                    {"shape": (1, 8, 32, 32, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, 32, 128),"ori_format": "NHWC","param_type": "output"},
                    [1,2,2,1], [1,1,1,1], "SAME"],
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.1, 0.1)}

#case7 = {"params": [{"shape": (3, 4, 16, 16, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (3, 16, 16, 64),"ori_format": "NHWC",
                   # "param_type": "input", "value_range": [1.0, 10.0]},
                   # None,
                   # None,
                   # {"shape": (3, 4, 15, 15, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (3, 15, 15, 64),"ori_format": "NHWC","param_type": "output"},
                    #[1,2,2,1], [1,1,1,1], "VALID"],
        # "calc_expect_func": calc_expect_func,
       #  "precision_standard": precision_info.PrecisionStandard(0.05, 0.05)}

#case8 = {"params": [{"shape": (4, 8, 10, 10, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 10, 10, 128),"ori_format": "NC1HWC0",
                    # "param_type": "input", "value_range": [1.0, 10.0]},
                   # None,
                   # None,
                   # {"shape": (4, 8, 8, 8, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 8, 8, 128),"ori_format": "NHWC","param_type": "output"},
                   # [1,3,3,1], [1,1,1,1], "VALID"],
       #  "calc_expect_func": calc_expect_func,
        # "precision_standard": precision_info.PrecisionStandard(0.05, 0.05)}
case7 = {"params": [{"shape": (2, 1, 1, 4000, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 1, 4000, 16),"ori_format": "NHWC"},
                    {"shape": (3, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 1, 1, 3),"ori_format": "NCHW"},
                    None,
                    {"shape": (4, 6, 3, 8, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 3, 8, 89),"ori_format": "NHWC"},
                    [1,1,3,1], [1,1,1,1], "VALID"],
         "expect": "success",
         "support_expect": True}
case8 = {"params": [{"shape": (2, 1, 1, 8600, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 1, 8600, 16),"ori_format": "NHWC"},
                    {"shape": (10, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 1, 1, 10),"ori_format": "NCHW"},
                    None,
                    {"shape": (4, 6, 3, 8, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 3, 8, 89),"ori_format": "NHWC"},
                    [1,1,10,1], [1,1,6,1], "SAME"],
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)

# ut_case.add_precision_case(["Ascend310", "Ascend910A"], case6)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case8)

from impl.avg_pool import check_supported
from impl.avg_pool import get_op_support_info
from impl.avg_pool import avg_pool_compute

def test_check_support(test_arg):
    check_supported({"shape": (1, 24, 1, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
    None,None,{"shape": (1, 24, 1, 256 ), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "output"},
    [1,2,2,1],[1,4,4,1],"VALIED","NHWC")
    check_supported({"shape": (1, 24, 1, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
    None,None,{"shape": (1, 24, 1, 256 ), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "output"},
    [1,2,2,1],[1,4,4,1],"VALIED","NCHW")
    check_supported({"shape": (1, 24, 1, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
    None,None,{"shape": (1, 1, 1, 256 ), "dtype": "float16", "format": "ND", "ori_shape": (1, 1, 1, 256),"ori_format": "ND", "param_type": "output"},
    [1,2,2,1],[1,4,4,1],"VALIED","NHWC")
    check_supported({"shape": (1, 24, 1, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
    None,None,{"shape": (1, 3, 3, 256 ), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 3, 256),"ori_format": "ND", "param_type": "output"},
    [1,255,21,1],[1,4,4,1],"VALIED","NHWC")
    
    check_supported({"shape": (1, 1, 100001, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 1, 100001, 256),"ori_format": "ND", "param_type": "input"},
    None,None,{"shape": (1, 24, 1, 256 ), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "output"},
    [1,2,2,1],[1,4,4,1],"VALIED","NCHW")

    check_supported({"shape": (1, 1024, 128, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 1024, 128, 256),"ori_format": "ND", "param_type": "input"},
    None,None,{"shape": (1, 1024, 1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1024, 1, 2),"ori_format": "ND", "param_type": "output"},
    [1,1,90,90],[1,1,90,90],"VALIED","NCHW")
    check_supported({"shape": (1, 128, 256, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 128, 256, 1024),"ori_format": "ND", "param_type": "input"},
    None,None,{"shape": (1, 1, 2, 1024 ), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "output"},
    [1,90,90,1],[1,90,90,1],"VALIED","NHWC")

def test_get_op_support_info(test_arg):
    get_op_support_info({"shape": (1, 16, 24, 24, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 24, 24, 256),"ori_format": "NHWC", "param_type": "input"},
    None,None,{"shape": (1, 16, 24, 24, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 24, 24, 256),"ori_format": "NHWC", "param_type": "output"},
    [1,2,2,1],[1,4,4,1],"VALIED","NHWC")
    get_op_support_info({"shape": (1, 16, 24, 24, 164), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 256, 24, 24),"ori_format": "NCHW", "param_type": "input"},
    None,None,{"shape": (1, 16, 24, 24, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 256, 24, 24),"ori_format": "NCHW", "param_type": "output"},
    [1,2,2,1],[1,4,4,1],"VALIED","NCHW")

def test_avg_pool_compute_001(test_arg):
    from te import tvm
    attr = {"ori_shape": [1, 24, 24, 256]}
    tensor_in = tvm.placeholder((1, 16, 24, 24, 16), name="tensor_in", dtype="float16", attrs=attr)
    avg_pool_compute(
        tensor_in, None, None, {
            "shape": (1, 16, 24, 24, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (1, 24, 24, 256),
            "ori_format": "NHWC",
            "param_type": "output"
        }, [1, 2, 2, 1], [1, 4, 4, 1], "SAME", "NHWC")
    avg_pool_compute(
        tensor_in, None, None, {
            "shape": (1, 16, 24, 24, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (1, 24, 24, 256),
            "ori_format": "NHWC",
            "param_type": "output"
        }, [1, 1, 3, 1], [1, 1, 4, 1], "SAME", "NC1HWC0")
ut_case.add_cust_test_func(test_func=test_avg_pool_compute_001)
ut_case.add_cust_test_func(test_func=test_check_support)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710", "Ascend910A"])
    exit(0)
