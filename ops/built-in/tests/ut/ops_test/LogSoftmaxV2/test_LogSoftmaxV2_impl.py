#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as NP
ut_case = OpUT("LogSoftmaxV2", None, None)

case1 = {"params": [{"shape": (10,10), "dtype": "float16", "format": "ND", "ori_shape": (10,10),"ori_format": "ND"},
                    {"shape": (10,10), "dtype": "float16", "format": "ND", "ori_shape": (10,10),"ori_format": "ND"},
                    -1],
         "case_name": "LogSoftmaxV2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (5, 9, 6, 11), "dtype": "float32", "format": "ND", "ori_shape": (5, 9, 6, 11),"ori_format": "ND"},
                    {"shape": (5, 9, 6, 11), "dtype": "float32", "format": "ND", "ori_shape": (5, 9, 6, 11),"ori_format": "ND"},
                    (2,3)],
         "case_name": "LogSoftmaxV2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (10, 100, 1000), "dtype": "float16", "format": "ND", "ori_shape": (10, 100, 1000),"ori_format": "ND"},
                    {"shape": (10, 100, 1000), "dtype": "float16", "format": "ND", "ori_shape": (10, 100, 1000),"ori_format": "ND"},
                    3],
         "case_name": "LogSoftmaxV2_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1,1,1), "dtype": "float16", "format": "ND", "ori_shape": (1,1,1),"ori_format": "ND"},
                    {"shape": (1,1,1), "dtype": "float16", "format": "ND", "ori_shape": (1,1,1),"ori_format": "ND"},
                    -1],
         "case_name": "LogSoftmaxV2_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (2, 1, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 2),"ori_format": "ND"},
                    {"shape": (2, 1, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 2),"ori_format": "ND"},
                    (1,2)],
         "case_name": "LogSoftmaxV2_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (8, 81, 25276), "dtype": "float32", "format": "ND", "ori_shape": (8, 81, 25276),"ori_format": "ND"},
                    {"shape": (8, 81, 25276), "dtype": "float32", "format": "ND", "ori_shape": (8, 81, 25276),"ori_format": "ND"},
                    -1],
         "case_name": "LogSoftmaxV2_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend710", "Ascend910A"], case6)

def calc_expect_func(input_x, output, axis):
    inputArr = input_x['value']
    shape = input_x['shape']
    s_type = input_x['dtype']

    if isinstance(axis,tuple):
        pass
    elif axis < 0:
        axis=len(shape) + axis
    shape_list=list(shape)
    shape_list[len(shape)-1]=1


    inputMax=NP.max(inputArr, axis=axis, keepdims=True).astype(NP.float16)
    inputMaxBroadcast=NP.broadcast_to(inputMax, shape)

    inputData=(inputArr-inputMaxBroadcast).astype(NP.float32)

    inputExp=NP.exp(inputData)
    inputSum=NP.sum(inputExp, axis=axis, keepdims=True)
    inputLog=NP.log(inputSum)
    inputLogBroadcast=NP.broadcast_to(inputLog, shape)

    outputArr = (inputData-inputLogBroadcast).astype(s_type)
    return outputArr

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 16, 8, 16), "shape": (16, 16, 8, 16), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 16, 8, 16), "shape": (16, 16, 8, 16), "param_type": "output"},
               3],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 1024), "shape": (2, 1024), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 1024), "shape": (2, 1024), "param_type": "output"},
               0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 2, 4, 32), "shape": (16, 2, 4, 32), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 2, 4, 32), "shape": (16, 2, 4, 32), "param_type": "output"},
               (2,3)],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33), "shape": (11, 33), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33), "shape": (11, 33), "param_type": "output"},
               -1],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
})

def test_get_op_support_info(test_arg):
    from impl.log_softmax_v2 import get_op_support_info
    get_op_support_info(
        {
            "dtype": "float16",
            "format": "ND",
            "ori_format": "ND",
            "ori_shape": (11, 33),
            "shape": (11, 33),
            "param_type": "input"
        }, None, -1, "log_softmax_v2")


ut_case.add_cust_test_func(test_func=test_get_op_support_info)
