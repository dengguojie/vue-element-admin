#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("ScatterMax", None, None)

case1 = {"params": [{"shape": (255,8,33), "dtype": "float32", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"},
                    {"shape": (33,), "dtype": "int32", "format": "ND", "ori_shape": (33,),"ori_format": "ND"},
                    {"shape": (33,8,33), "dtype": "float32", "format": "ND", "ori_shape": (33,8,33),"ori_format": "ND"},
                    {"shape": (255,8,33), "dtype": "float32", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"},
                    False],
         "case_name": "scatter_max_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (4,32,32), "dtype": "int32", "format": "ND", "ori_shape": (4,32,32),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,32,32), "dtype": "int32", "format": "ND", "ori_shape": (1,32,32),"ori_format": "ND"},
                    {"shape": (4,32,32), "dtype": "int32", "format": "ND", "ori_shape": (4,32,32),"ori_format": "ND"},
                    False],
         "case_name": "scatter_max_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (4,17,17), "dtype": "float16", "format": "ND", "ori_shape": (4,17,17),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (2,17,17), "dtype": "float16", "format": "ND", "ori_shape": (2,17,17),"ori_format": "ND"},
                    {"shape": (4,17,17), "dtype": "float16", "format": "ND", "ori_shape": (4,17,17),"ori_format": "ND"},
                    False],
         "case_name": "scatter_max_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (255,220,300), "dtype": "float32", "format": "ND", "ori_shape": (255,220,300),"ori_format": "ND"},
                    {"shape": (33,), "dtype": "int32", "format": "ND", "ori_shape": (33,),"ori_format": "ND"},
                    {"shape": (33,220,300), "dtype": "float32", "format": "ND", "ori_shape": (33,220,300),"ori_format": "ND"},
                    {"shape": (255,220,300), "dtype": "float32", "format": "ND", "ori_shape": (255,220,300),"ori_format": "ND"},
                    False],
         "case_name": "scatter_max_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (255,220,300), "dtype": "int32", "format": "ND", "ori_shape": (255,220,300),"ori_format": "ND"},
                    {"shape": (32,), "dtype": "int32", "format": "ND", "ori_shape": (32,),"ori_format": "ND"},
                    {"shape": (32,220,300), "dtype": "int32", "format": "ND", "ori_shape": (32,220,300),"ori_format": "ND"},
                    {"shape": (255,220,300), "dtype": "int32", "format": "ND", "ori_shape": (255,220,300),"ori_format": "ND"},
                    False],
         "case_name": "scatter_max_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (255, 33), "dtype": "int32", "format": "ND", "ori_shape": (255, 33),"ori_format": "ND"},
                    {"shape": (220, 300), "dtype": "int32", "format": "ND", "ori_shape": (220, 300),"ori_format": "ND"},
                    {"shape": (220, 300, 33), "dtype": "int32", "format": "ND", "ori_shape": (220, 300, 33),"ori_format": "ND"},
                    {"shape": (255, 33), "dtype": "int32", "format": "ND", "ori_shape": (255, 33),"ori_format": "ND"},
                    False],
         "case_name": "scatter_max_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)


