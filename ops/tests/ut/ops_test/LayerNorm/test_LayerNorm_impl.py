#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("LayerNorm", None, None)

case1 = {"params": [{"shape": (64,128,1023), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1023),"ori_format": "NCHW"},
                    {"shape": (1023,), "dtype": "float16", "format": "NCHW", "ori_shape": (1023,),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (64,128,1023), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1023),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    1, 2],
         "case_name": "layer_norm_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (64,128,1024), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1024),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (64,128,1024), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1024),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    1, 2],
         "case_name": "layer_norm_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)


def calc_expect_func(x, gamma, beta, y, mean, variance, begin_norm_axis,begin_params_axis):
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

ut_case.add_precision_case("all", {"params": [{"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "output"},
                                              {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "output"},
                                              {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,),"ori_format": "NCHW", "param_type": "output"},
                                              0, 0
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
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
if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/disk1/ty_mindstudio/.mindstudio/huawei/adk/1.76.T1.0.B010/toolkit/tools/simulator")
