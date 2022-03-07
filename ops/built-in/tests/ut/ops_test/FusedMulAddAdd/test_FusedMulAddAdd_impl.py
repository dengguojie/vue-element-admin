#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("FusedMulAddAdd", None, None)

case1 = {
    "params": [{"shape": (640, 4624), "dtype": "float32", "format": "ND", "ori_shape": (640, 4624), "ori_format": "ND"},
               {"shape": (640, 1), "dtype": "float32", "format": "ND", "ori_shape": (640, 1), "ori_format": "ND"},
               {"shape": (4624,), "dtype": "float32", "format": "ND", "ori_shape": (4624,), "ori_format": "ND"},
               {"shape": (640, 4624), "dtype": "float32", "format": "ND", "ori_shape": (640, 4624), "ori_format": "ND"},
               {"shape": (640, 4624), "dtype": "float32", "format": "ND", "ori_shape": (640, 4624),
                "ori_format": "ND"}],
    "case_name": "FusedMulAddAdd_1",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True}
case2 = {
    "params": [{"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"},
               {"shape": (640, 1), "dtype": "float32", "format": "ND", "ori_shape": (640, 1), "ori_format": "ND"},
               {"shape": (4624,), "dtype": "float32", "format": "ND", "ori_shape": (4624,), "ori_format": "ND"},
               {"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"},
               {"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"}],
    "case_name": "FusedMulAddAdd_2",
    "expect": "success",
    "format_expect": ["FRACTAL_NZ"],
    "support_expect": True}
case3 = {
    "params": [{"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"},
               {"shape": (640,), "dtype": "float32", "format": "ND", "ori_shape": (640,), "ori_format": "ND"},
               {"shape": (4624,), "dtype": "float32", "format": "ND", "ori_shape": (4624,), "ori_format": "ND"},
               {"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"},
               {"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"}],
    "case_name": "FusedMulAddAdd_3",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True}
case4 = {
    "params": [{"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"},
               {"shape": (640, 1), "dtype": "float32", "format": "ND", "ori_shape": (640, 1), "ori_format": "ND"},
               {"shape": (1, 4624), "dtype": "float32", "format": "ND", "ori_shape": (1, 4624), "ori_format": "ND"},
               {"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"},
               {"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"}],
    "case_name": "FusedMulAddAdd_4",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True}
case5 = {
    "params": [{"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"},
               {"shape": (640, 1), "dtype": "float32", "format": "ND", "ori_shape": (640, 1), "ori_format": "ND"},
               {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
               {"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"},
               {"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"}],
    "case_name": "FusedMulAddAdd_5",
    "expect": "success",
    "format_expect": ["FRACTAL_NZ"],
    "support_expect": True}
case6 = {
    "params": [{"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"},
               {"shape": (639, 1), "dtype": "float32", "format": "ND", "ori_shape": (639, 1), "ori_format": "ND"},
               {"shape": (4620,), "dtype": "float32", "format": "ND", "ori_shape": (4620,), "ori_format": "ND"},
               {"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"},
               {"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"}],
    "case_name": "FusedMulAddAdd_6",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True}
case7 = {
    "params": [{"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"},
               {"shape": (640, 1), "dtype": "float32", "format": "ND", "ori_shape": (640, 1), "ori_format": "ND"},
               {"shape": (4624,), "dtype": "float32", "format": "ND", "ori_shape": (4624,), "ori_format": "ND"},
               {"shape": (290, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4640),
                "ori_format": "ND"},
               {"shape": (289, 40, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (640, 4624),
                "ori_format": "ND"}],
    "case_name": "FusedMulAddAdd_7",
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910", "Ascend610"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910", "Ascend610"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910", "Ascend610"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910", "Ascend610"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910", "Ascend610"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910", "Ascend610"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910", "Ascend610"], case7)
