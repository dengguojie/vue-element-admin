#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("DecodeBoundariesTarget", None, None)

def calc_expect_func(inputA, inputB, output):
    data_a = inputA['value']
    data_b = inputB['value']
    a = np.reshape(data_a, (-1, 1, 1))
    b_x1y1x2y2 = np.reshape(data_b, (-1, 2, 2))
    b_x1y1, b_x2y2 = np.split(b_x1y1x2y2, 2, axis=1)
    waha = b_x2y2 - b_x1y1
    xaya = (b_x2y2 + b_x1y1) * 0.5
    xa, ya = np.split(xaya, 2, axis=2)
    w, h = np.split(waha, 2, axis=2)
    c = a*w+xa
    data_c = np.reshape(c, output['shape'])
    return data_c

def gen_precision_case(shape_x, shape_y, dtype):
    return {"params": [{"shape": shape_x, "dtype": dtype, "ori_shape": shape_x, "ori_format": "ND", "format": "ND","param_type":"input"},
                       {"shape": shape_y, "dtype": dtype, "ori_shape": shape_y, "ori_format": "ND", "format": "ND","param_type":"input"},
                       {"shape": shape_x, "dtype": dtype, "ori_shape": shape_x, "ori_format": "ND", "format": "ND","param_type":"output"}],
            "expect": "success",
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

def gen_decode_boundaries_target_case(shape_x, shape_y, dtype, case_name_val):
    return {"params": [{"shape": shape_x, "dtype": dtype, "ori_shape": shape_x, "ori_format": "ND", "format": "ND"},
                       {"shape": shape_y, "dtype": dtype, "ori_shape": shape_y, "ori_format": "ND", "format": "ND"},
                       {"shape": shape_x, "dtype": dtype, "ori_shape": shape_x, "ori_format": "ND", "format": "ND"}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case1 = gen_decode_boundaries_target_case((1, 1), (1, 4), "float16", "decode_boundaries_target_1")
case2 = gen_decode_boundaries_target_case((16, 1), (16, 4), "float16", "decode_boundaries_target_2")
case3 = gen_decode_boundaries_target_case((100, 1), (100, 4), "float16", "decode_boundaries_target_3")

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

ut_case.add_precision_case("Ascend910", gen_precision_case((1, 1), (1, 4), "float16"))
ut_case.add_precision_case("Ascend910", gen_precision_case((3, 1), (3, 4), "float16"))
ut_case.add_precision_case("Ascend910", gen_precision_case((100, 1), (100, 4), "float16"))
ut_case.add_precision_case("Ascend910", gen_precision_case((257, 1), (257, 4), "float16"))

if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/disk1/ty_mindstudio/.mindstudio/huawei/adk/1.76.T1.0.B010/toolkit/tools/simulator")

