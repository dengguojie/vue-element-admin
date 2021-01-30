# # -*- coding:utf-8 -*-
import sys
import numpy as np
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("is_close")

#pylint: disable=unused-argument
def calc_expect_func(input_x, input_y, output_z,rtol=1e-05, atol=1e-08, equal_nan=False):
    res=np.isclose(input_x["value"],input_y["value"],rtol,atol,equal_nan)
    return [res, ]

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"}],
    "case_name": "test_is_close_case_1",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3,), "shape": (3,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3,), "shape": (3,),
                "param_type": "input"},
               {"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (3,), "shape": (3,),
                "param_type": "output"}],
    "case_name": "test_is_close_case_2",
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input" ,"value_range":[0, 10]},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input","value_range":[0, 10]},
               {"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"}],
    "case_name": "test_isclose_case_3",
    "calc_expect_func": calc_expect_func
})