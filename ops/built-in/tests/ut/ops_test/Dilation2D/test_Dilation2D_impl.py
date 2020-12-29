#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf
ut_case = OpUT("Dilation2D", "impl.dilation2d", "dilation2d")

case1 = {"params": [{"shape": (10000,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10000,3,3,1),"ori_format": "NHWC"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,1),"ori_format": "NHWC"},
                    {"shape": (10000,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10000,3,3,1),"ori_format": "NHWC"},
                    [1,1,1,1],[1,1,1,1],"SAME",[0,0,0,0],False,"NHWC"],
         "case_name": "Dilation2D_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1,320,17,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,3,3,1),"ori_format": "NHWC"},
                    {"shape": (1,320,5,5,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,3,3,1),"ori_format": "NHWC"},
                    {"shape": (1,320,17,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,3,3,1),"ori_format": "NHWC"},
                    [1,1,1,1], [1,2,2,1], "VALID", [0,0,0,0], False,"NHWC"],
         "case_name": "Dilation2D_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (3,100,9973,14,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,3,3,1),"ori_format": "NHWC"},
                    {"shape": (1,100,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,3,3,1),"ori_format": "NHWC"},
                    {"shape": (3,100,9973,14,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,3,3,1),"ori_format": "NHWC"},
                    [1,1,1,1], [1,1,1,1], "CALCULATED", [0,1,0,1], False,"NHWC"],
         "case_name": "Dilation2D_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (1,1,640,960,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,640,960,1),"ori_format": "NHWC"},
                    {"shape": (1,1,20,20,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,20,20,1),"ori_format": "NHWC"},
                    {"shape": (1,1,640,960,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,3,11,32),"ori_format": "NHWC"},
                    [1,1,1,1], [1,1,1,1], "SAME", [0,0,0,0], False,"NHWC"],
         "case_name": "Dilation2D_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (1,1,196,96,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,640,960,1),"ori_format": "NHWC"},
                    {"shape": (1,1,20,20,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,20,20,1),"ori_format": "NHWC"},
                    {"shape": (1,1,177,77,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,3,11,32),"ori_format": "NHWC"},
                    [1,1,1,1], [1,1,1,1], "CALCULATED", [-1,1,0,1], True,"NHWC"],
         "case_name": "Dilation2D_5",
         "expect": "failed",
         "support_expect": True}

case6 = {"params": [{"shape": (1,320,16,65536,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,640,960,1),"ori_format": "NHWC"},
                    {"shape": (1,320,5,5,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,20,20,1),"ori_format": "NHWC"},
                    {"shape": (1,320,16,65536,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,3,11,32),"ori_format": "NHWC"},
                    [1,1,1,1], [1,1,1,1], "SAME", [0,0,0,0], True,"NHWC"],
         "case_name": "Dilation2D_6",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape": (1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,640,960,1),"ori_format": "NHWC"},
                    {"shape": (1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,20,20,1),"ori_format": "NHWC"},
                    {"shape": (1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,3,11,32),"ori_format": "NHWC"},
                    [1,1,1,1], [1,1,1,1], "SAME", [0,0,0,0], True,"NHWC"],
         "case_name": "Dilation2D_7",
         "expect": "success",
         "support_expect": True}

case8 = {"params": [{"shape": (1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,640,960,1),"ori_format": "NHWC"},
                    {"shape": (1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,20,20,1),"ori_format": "NHWC"},
                    {"shape": (1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,3,11,32),"ori_format": "NHWC"},
                    [1,1,1,1], [1,1,1,1], "CALCULATED", [0,0,0,0], False,"NHWC"],
         "case_name": "Dilation2D_8",
         "expect": "success",
         "support_expect": True}

case9 = {"params": [{"shape": (1,1,196,96,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,640,960,1),"ori_format": "NHWC"},
                    {"shape": (1,1,20,20,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,20,20,1),"ori_format": "NHWC"},
                    {"shape": (1,1,177,77,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,3,11,32),"ori_format": "NHWC"},
                    [1,1,1,1], [1,1,1,1], "CALCULATED", [0,0,0,0], True,"NHWC"],
         "case_name": "Dilation2D_9",
         "expect": "success",
         "support_expect": True}

case10 = {"params": [{"shape": (1,1,6,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,640,960,1),"ori_format": "NHWC"},
                    {"shape": (1,1,5,5,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,20,20,1),"ori_format": "NHWC"},
                    {"shape": (1,1,6,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,3,11,32),"ori_format": "NHWC"},
                    [1,1,1,1], [1,1,1,1], "CALCULATED", [10,0,0,0], False,"NHWC"],
         "case_name": "Dilation2D_10",
         "expect": "failed",
         "support_expect": True}

case11 = {"params": [{"shape": (10,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,640,960,1),"ori_format": "NHWC"},
                    {"shape": (1,1,1,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,20,20,1),"ori_format": "NHWC"},
                    {"shape": (10,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,3,11,32),"ori_format": "NHWC"},
                    [1,2,2,1], [1,1,1,1], "SAME", [0,0,0,0], False,"NHWC"],
         "case_name": "Dilation2D_11",
         "expect": "success",
         "support_expect": True}

case12= {"params": [{"shape": (11,30,10,18,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,640,960,1),"ori_format": "NHWC"},
                    {"shape": (1,30,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,20,20,1),"ori_format": "NHWC"},
                    {"shape": (11,30,10,18,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,3,11,32),"ori_format": "NHWC"},
                    [1,1,1,1], [1,1,1,1], "SAME", [0,0,0,0], False,"NHWC"],
         "case_name": "Dilation2D_12",
         "expect": "success",
         "support_expect": True}

case13= {"params": [{"shape": (1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,640,960,1),"ori_format": "NHWC"},
                    {"shape": (1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,20,20,1),"ori_format": "NHWC"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,3,11,32),"ori_format": "NHWC"},
                    [1,3,3,1], [1,1,1,1], "SAME", [0,0,0,0], False,"NHWC"],
         "case_name": "Dilation2D_13",
         "expect": "success",
         "support_expect": True}

case14= {"params": [{"shape": (1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,640,960,1),"ori_format": "NCHW"},
                    {"shape": (1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,20,20,1),"ori_format": "NCHW"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,3,11,32),"ori_format": "NCHW"},
                    [1,3,3,1], [1,1,1,1], "SAME", [0,0,0,0], False,"NCHW"],
         "case_name": "Dilation2D_14",
         "expect": "failed",
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case11)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case12)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case13)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case14)


def nhwc_data4Dto5D(inputData, channel0=16):
    Ftemp = np.shape(inputData)
    F = [Ftemp[0], np.int(np.ceil(Ftemp[3] * 1.0 / channel0)), Ftemp[1],
         Ftemp[2], channel0]
    outputData = np.zeros(F)
    for N in range(F[0]):
        for C1 in range(F[1]):
            for k in range(channel0):
                if (C1 * channel0 + k < Ftemp[3]):
                    outputData[N, C1, :, :, k] = inputData[N, :, :,
                                                 C1 * channel0 + k]
    return outputData

def nhwc_data5Dto4D(data5D, nhwcDims, channel0=16):
    nc1hwc0Dims = [nhwcDims[0], np.int(np.ceil(nhwcDims[3] * 1.0 / channel0)),
                   nhwcDims[1],
                   nhwcDims[2], channel0]
    outputData = np.zeros(nhwcDims, data5D.dtype)
    for N in range(nc1hwc0Dims[0]):
        for C1 in range(nc1hwc0Dims[1]):
            for k in range(channel0):
                if (C1 * channel0 + k < nhwcDims[3]):
                    outputData[N, :, :, C1 * channel0 + k] = data5D[N, C1, :, :,
                                                             k]
    return outputData

def calc_expect_func(input_x, input_filter, y, strides,rates,padding_mode="SAME",pads=[0,0,0,0],kernel_name="dilation2d"):
    shape_x = input_x.get("value").shape
    shape_filter = input_filter.get("value").shape
    x_4d_shape = [shape_x[0],shape_x[2],shape_x[3],shape_x[1]*shape_x[4]]
    filter_4d_shape = [shape_filter[0],shape_filter[2],shape_filter[3],shape_filter[1]*shape_filter[4]]

    input_tensor = input_x.get("value")
    filter_tensor = input_filter.get("value")
    input_4d = nhwc_data5Dto4D(input_tensor,x_4d_shape)
    filter_4d = nhwc_data5Dto4D(filter_tensor, filter_4d_shape)
    filter_3d = filter_4d.reshape(filter_4d_shape[1:])
    to_dilation = tf.nn.dilation2d(input_4d,filter_3d,strides=strides,rates=rates,padding=padding_mode)
    with tf.compat.v1.Session() as sess:
        tf_out = sess.run(to_dilation)
    out_tensor = nhwc_data4Dto5D(tf_out).astype("float16")
    return out_tensor

def gen_dilation2d_precision_case(shape_x,shape_filter,shape_y,dtype,strides,rates,padding_mode="SAME",pads=[0,0,0,0],kernel_name="dilation2d",expect="success"):
    return {"params": [{"dtype": dtype, "shape": shape_x, "format": "NC1HWC0", "ori_shape": shape_x, "ori_format": "NHWC",
                        "param_type": "input", "value_range": [-10, 10]},
                       {"dtype": dtype, "shape": shape_filter, "format": "NC1HWC0", "ori_shape": shape_filter, "ori_format": "NHWC",
                        "param_type": "input", "value_range": [-10, 10]},
                       {"dtype": dtype, "shape": shape_y, "format": "NC1HWC0", "ori_shape": shape_y, "ori_format": "NHWC",
                        "param_type": "output", "value_range": [-100, 100]},
                       strides,rates,padding_mode,pads],
            "case_name": kernel_name,
            "expect": expect,
            "calc_expect_func": calc_expect_func}

ut_case.add_precision_case("all",
                           gen_dilation2d_precision_case((10,1,3,3,16),(1,1,1,3,16),(10,1,3,3,16),"float16",
                                                         [1,1,1,1],[1,1,1,1],"SAME",[0,0,0,0],"dilation2d_1","success"))
ut_case.add_precision_case("all",
                           gen_dilation2d_precision_case((1,1,25,25,16),(1,1,3,3,16),(1,1,13,13,16),"float16",
                                                         [1,2,2,1],[1,2,2,1],"SAME",[0,0,0,0],"dilation2d_2","success"))
