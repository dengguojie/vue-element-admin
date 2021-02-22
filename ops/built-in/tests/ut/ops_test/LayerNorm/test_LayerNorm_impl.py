#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("LayerNorm", None, None)

case1 = {"params": [{"shape": (24,512,1024), "dtype": "float32", "format": "NCHW", "ori_shape": (24,512,1024),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float32", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float32", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (64,128,1023), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1023),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    -1, -1, 1.0000000116860974e-07],
         "case_name": "layer_norm_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (24,512,1024), "dtype": "float16", "format": "NCHW", "ori_shape": (24,512,1024),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (64,128,1023), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1023),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    -1, -1, 1.0000000116860974e-07],
         "case_name": "layer_norm_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (64,64,8,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64,128,1024),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float32", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float32", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (64,128,1024), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1024),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    -2, -1],
         "case_name": "layer_norm_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (64,64,8,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64,128,1024),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (64,128,1024), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1024),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    -2, -1],
         "case_name": "layer_norm_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (64,2,64,8,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (64,64,8,32),"ori_format": "NHWC"},
                    {"shape": (1,2,1,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (32,),"ori_format": "NCHW"},
                    {"shape": (1,2,1,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (32,),"ori_format": "NCHW"},
                    {"shape": (64,128,1024), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,128,1024),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NCHW"},
                    -1, -1],
         "case_name": "layer_norm_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (64,2,64,8,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,64,8,32),"ori_format": "NHWC"},
                    {"shape": (1,2,1,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (32,),"ori_format": "NCHW"},
                    {"shape": (1,2,1,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (32,),"ori_format": "NCHW"},
                    {"shape": (64,128,1024), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,128,1024),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NCHW"},
                    -1, -1],
         "case_name": "layer_norm_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)


def calc_expect_func(x, gamma, beta, mean, variance, begin_norm_axis):
    input_xArr = x['value']
    input_gammaArr = gamma['value']
    input_betaArr = beta['value']
    dtype = x['dtype']
    shape_x = x['shape']
    index_list = [index for index, _ in enumerate(shape_x)]
    reduce_axis = tuple(index_list[begin_norm_axis:])
    if dtype == "float16":
        input_xArr = input_xArr.astype(np.float32)
        input_gammaArr = input_gammaArr.astype(np.float32)
        input_betaArr = input_betaArr.astype(np.float32)

    mean = np.mean(input_xArr, reduce_axis, np.float32, keepdims=True)
    variance = np.mean(np.square(input_xArr - mean), reduce_axis, np.float32, keepdims=True)
    epsilon = 1e-12
    normalize = (input_xArr - mean) / np.sqrt(variance + epsilon)
    res = input_gammaArr*normalize + input_betaArr

    if dtype == "float16":
        mean = mean.astype(np.float16)
        variance = variance.astype(np.float16)
        res = res.astype(np.float16)
    return res, mean, variance

# ut_case.add_precision_case("all", {"params": [{"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "input"},
#                                               {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "input"},
#                                               {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "input"},
#                                               {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "output"},
#                                               {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "output"},
#                                               {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "output"},
#                                               0, 0
#                                               ],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.1, 0.1)
#                                    })
# ut_case.add_precision_case("all", {"params": [{"shape": (2,2,768), "dtype": "float32", "format": "NCHW", "ori_shape": (2,2,768),"ori_format": "NCHW", "param_type": "input"},
#                                               {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "input"},
#                                               {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "input"},
#                                               {"shape": (2,2,768), "dtype": "float32", "format": "NCHW", "ori_shape": (2,2,768),"ori_format": "NCHW", "param_type": "output"},
#                                               {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "output"},
#                                               {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "output"},
#                                               1, 2
#                                               ],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                    })

