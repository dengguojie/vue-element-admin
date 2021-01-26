"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dot ut case
"""
# # -*- coding:utf-8 -*-
import sys
import numpy as np
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("dot")

#pylint: disable=unused-argument
def calc_expect_func(input_x, input_y, output_z):
    res=np.dot(input_x["value"], input_y["value"])
    result = res.reshape((1,))
    return [result, ]

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "output"}],
    "case_name": "test_dot_case_1",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "output"}],
    "case_name": "test_dot_case_2",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "output"}],
    "case_name": "test_dot_case_3",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "output"}],
    "case_name": "test_dot_case_4",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "uint8", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "uint8", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "uint8", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "output"}],
    "case_name": "test_dot_case_5",
    "calc_expect_func": calc_expect_func
})
