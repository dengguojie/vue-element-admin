# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT

ut_case = OpUT("RotatedBoxEncode", "impl.rotated_box_encode", "rotated_box_encode")

ut_case.add_case(["Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "shape": (8, 5, 25601), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "shape": (8, 5, 25601), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "shape": (8, 5, 25601), "param_type": "output"},
                [1.0, 1.0, 1.0, 1.0, 1.0]],
    "case_name": "rotated_box_encode_0",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

ut_case.add_case(["Ascend910A"], {
    "params": [{"dtype": "float16", "format": "ND", "shape": (8, 5, 25601), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "shape": (8, 5, 25601), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "shape": (8, 5, 25601), "param_type": "output"},
                [1.0, 1.0, 1.0, 1.0, 1.0]],
    "case_name": "rotated_box_encode_1",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)