# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
import numpy as np

ut_case = OpUT("AnchorResponseFlags", None, None)

shape_1 = (128, 4)
data_type = "float32"
data_format = "ND"
case1 = {
    "params": [
        {"shape": shape_1, "dtype": data_type, "format": data_format, "ori_shape": shape_1, "ori_format": data_format},
        {"shape": [300], "dtype": "uint8", "format": data_format, "ori_shape": [300], "ori_format": data_format},
        [10, 10], [32, 32], 3
    ],
    "case_name": "mmdet_flags_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

shape_2 = (256, 4)
data_type = "float32"
data_format = "ND"
case2 = {
    "params": [
        {"shape": shape_2, "dtype": data_type, "format": data_format, "ori_shape": shape_2, "ori_format": data_format},
        {"shape": [3000], "dtype": "uint8", "format": data_format, "ori_shape": [3000], "ori_format": data_format},
        [10, 10], [32, 32], 30
    ],
    "case_name": "mmdet_flags_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
