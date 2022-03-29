# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import ElementwiseOpUT
import numpy as np
from op_test_frame.common import op_status

ut_case = ElementwiseOpUT("Mish")

def calc_expect_func(input_x, output_y):
    exp_val = np.exp(input_x["value"])
    add_exp_val = 1 + exp_val
    mul_val = np.multiply(add_exp_val, add_exp_val)
    add_val = 1 + mul_val
    div_val = 2/add_val
    sub_val = 1 - div_val
    outputArr = np.multiply(input_x["value"], sub_val)
    return [outputArr, ]

ut_case.add_elewise_case_simple(["Hi3796CV300ES", "Ascend910"], ["float16"], (1024,))
ut_case.add_elewise_case_simple(["Hi3796CV300ES", "Ascend910"], ["float16"], (256,256))
ut_case.add_elewise_case_simple(["Hi3796CV300ES", "Ascend910"], ["float16"], (1, 2, 608, 608, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300ES", "Ascend910"], ["float16"], (1, 4, 304, 304, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300ES", "Ascend910"], ["float16"], (1, 4, 152, 152, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300ES", "Ascend910"], ["float16"], (1, 8, 76, 76, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300ES", "Ascend910"], ["float16"], (1, 16, 38, 38, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300ES", "Ascend910"], ["float16"], (1, 32, 38, 38, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300ES", "Ascend910"], ["float16"], (1, 64, 19, 19, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300ES"], ["float"], (1024,), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300ES"], ["float"], (256,256), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300ES"], ["float"], (1, 2, 608, 608, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300ES"], ["float"], (1, 4, 304, 304, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300ES"], ["float"], (1, 4, 152, 152, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300ES"], ["float"], (1, 8, 76, 76, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300ES"], ["float"], (1, 16, 38, 38, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300ES"], ["float"], (1, 32, 38, 38, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300ES"], ["float"], (1, 64, 19, 19, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300ES"], ["int32"], (256,256), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300ES"], ["int8"], (256,256), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float16"], (1024,))
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float16"], (256,256))
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float16"], (1, 2, 608, 608, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float16"], (1, 4, 304, 304, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float16"], (1, 4, 152, 152, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float16"], (1, 8, 76, 76, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float16"], (1, 16, 38, 38, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float16"], (1, 32, 38, 38, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float16"], (1, 64, 19, 19, 16))
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float"], (1024,), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float"], (256,256), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float"], (1, 2, 608, 608, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float"], (1, 4, 304, 304, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float"], (1, 4, 152, 152, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float"], (1, 8, 76, 76, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float"], (1, 16, 38, 38, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float"], (1, 32, 38, 38, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["float"], (1, 64, 19, 19, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["int32"], (256,256), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Hi3796CV300CS"], ["int8"], (256,256), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend310"], ["float16"], (1024,))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16"], (256,256))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16"], (1, 2, 608, 608, 16))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16"], (1, 4, 304, 304, 16))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16"], (1, 4, 152, 152, 16))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16"], (1, 8, 76, 76, 16))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16"], (1, 16, 38, 38, 16))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16"], (1, 32, 38, 38, 16))
ut_case.add_elewise_case_simple(["Ascend310"], ["float16"], (1, 64, 19, 19, 16))
ut_case.add_elewise_case_simple(["Ascend310"], ["float"], (1024,), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend310"], ["float"], (256,256), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend310"], ["float"], (1, 2, 608, 608, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend310"], ["float"], (1, 4, 304, 304, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend310"], ["float"], (1, 4, 152, 152, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend310"], ["float"], (1, 8, 76, 76, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend310"], ["float"], (1, 16, 38, 38, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend310"], ["float"], (1, 32, 38, 38, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend310"], ["float"], (1, 64, 19, 19, 16), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend310"], ["int32"], (256,256), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend310"], ["int8"], (256,256), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend710"], ["float16"], (1024,))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16"], (256,256))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16"], (1, 2, 608, 608, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16"], (1, 4, 304, 304, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16"], (1, 4, 152, 152, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16"], (1, 8, 76, 76, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16"], (1, 16, 38, 38, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16"], (1, 32, 38, 38, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16"], (1, 64, 19, 19, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float"], (1024,))
ut_case.add_elewise_case_simple(["Ascend710"], ["float"], (256,256))
ut_case.add_elewise_case_simple(["Ascend710"], ["float"], (1, 2, 608, 608, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float"], (1, 4, 304, 304, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float"], (1, 4, 152, 152, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float"], (1, 8, 76, 76, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float"], (1, 16, 38, 38, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float"], (1, 32, 38, 38, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["float"], (1, 64, 19, 19, 16))
ut_case.add_elewise_case_simple(["Ascend710"], ["int32"], (256,256), expect=op_status.FAILED)
ut_case.add_elewise_case_simple(["Ascend710"], ["int8"], (256,256), expect=op_status.FAILED)
case1 = {
    "params": [
        {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
        {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"}
    ],
    "addition_params": {"impl_mode": "super_performance"},
    "case_name": "mish_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}
ut_case.add_case(["Ascend910A"], case1)