#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("ScatterDiv", None, None)

case1 = {"params": [{"shape": (255,8,33), "dtype": "float32", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"},
                    {"shape": (33,), "dtype": "int32", "format": "ND", "ori_shape": (33,),"ori_format": "ND"},
                    {"shape": (33,8,33), "dtype": "float32", "format": "ND", "ori_shape": (33,8,33),"ori_format": "ND"},
                    {"shape": (255,8,33), "dtype": "float32", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"},
                    False],
         "case_name": "scatter_div_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (255,8,33), "dtype": "float16", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"},
                    {"shape": (33,), "dtype": "int32", "format": "ND", "ori_shape": (33,),"ori_format": "ND"},
                    {"shape": (33,8,33), "dtype": "float16", "format": "ND", "ori_shape": (33,8,33),"ori_format": "ND"},
                    {"shape": (255,8,33), "dtype": "float16", "format": "ND", "ori_shape": (255,8,33),"ori_format": "ND"},
                    False],
         "case_name": "scatter_div_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (4,17,17), "dtype": "float16", "format": "ND", "ori_shape": (4,17,17),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (2,17,17), "dtype": "float16", "format": "ND", "ori_shape": (2,17,17),"ori_format": "ND"},
                    {"shape": (4,17,17), "dtype": "float16", "format": "ND", "ori_shape": (4,17,17),"ori_format": "ND"},
                    False],
         "case_name": "scatter_div_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
# case4 = {"params": [{"shape": (255,220,300), "dtype": "float32", "format": "ND", "ori_shape": (255,220,300),"ori_format": "ND"},
                    # {"shape": (33,), "dtype": "int32", "format": "ND", "ori_shape": (33,),"ori_format": "ND"},
                    # {"shape": (33,220,300), "dtype": "float32", "format": "ND", "ori_shape": (33,220,300),"ori_format": "ND"},
                    # {"shape": (255,220,300), "dtype": "float32", "format": "ND", "ori_shape": (255,220,300),"ori_format": "ND"},
                    # False],
         # "case_name": "scatter_div_4",
         # "expect": "success",
         # "format_expect": [],
         # "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)



if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)


