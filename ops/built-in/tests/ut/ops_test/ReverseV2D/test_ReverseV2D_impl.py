#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ReverseV2D", None, None)

case1 = {"params": [{"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"},
                    {"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"},
                    [1,2]],
        "case_name": "ReverseV2D_1",
        "expect": "success",
        "format_expect": [],
        "support_expect": True}
case2 = {"params": [{"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"},
                    {"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"},
                    [0,1]],
        "case_name": "ReverseV2D_2",
        "expect": "success",
        "format_expect": [],
        "support_expect": True}
case3 = {"params": [{"shape": (1, 2, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 16),"ori_format": "ND"},
                    {"shape": (1, 2, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 16),"ori_format": "ND"},
                    [1,2]],
        "case_name": "ReverseV2D_3",
        "expect": "success",
        "format_expect": [],
        "support_expect": True}
case4 = {"params": [{"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    [0,1]],
        "case_name": "ReverseV2D_4",
        "expect": "success",
        "format_expect": [],
        "support_expect": True}
case5 = {"params": [{"shape": (128, 16, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16, 16),"ori_format": "ND"},
                    {"shape": (128, 16, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16, 16),"ori_format": "ND"},
                    [1,2]],
        "case_name": "ReverseV2D_5",
        "expect": "success",
        "format_expect": [],
        "support_expect": True}
case6 = {"params": [{"shape": (1, 16, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 16, 16),"ori_format": "ND"},
                    {"shape": (1, 16, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 16, 16),"ori_format": "ND"},
                    [1,2]],
        "case_name": "ReverseV2D_6",
        "expect": "success",
        "format_expect": [],
        "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case6)

if __name__ == "__main__":
    ut_case.run("Ascend910")



