#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("DiagD", None, None)

case1 = {"params": [{"shape": (3,), "dtype": "float32", "format": "ND", "ori_shape": (3,),"ori_format": "ND"},
                    {"shape": (3,3), "dtype": "float32", "format": "ND", "ori_shape": (3,3),"ori_format": "ND"},
                    {"shape": (3,3), "dtype": "float32", "format": "ND", "ori_shape": (3,3),"ori_format": "ND"}],
         "case_name": "diag_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "diag_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32,), "dtype": "float32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32, 32), "dtype": "float32", "format": "ND", "ori_shape": (32, 32),"ori_format": "ND"},
                    {"shape": (32, 32), "dtype": "float32", "format": "ND", "ori_shape": (32, 32),"ori_format": "ND"}],
         "case_name": "diag_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (128, ), "dtype": "float16", "format": "ND", "ori_shape": (128, ),"ori_format": "ND"},
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND"},
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND"}],
         "case_name": "diag_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "diag_d_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
