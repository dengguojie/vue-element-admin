#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("AddRowRanges", None, None)

case1 = {"params": [{"shape": (7, 7), "dtype": "float32", "format": "ND", "ori_shape": (7, 7),"ori_format": "ND"},
                    {"shape": (11, 7), "dtype": "float32", "format": "ND", "ori_shape": (11, 7),"ori_format": "ND"},
                    {"shape": (7, 2), "dtype": "int32", "format": "ND", "ori_shape": (7, 2),"ori_format": "ND"},
                    {"shape": (7, 7), "dtype": "int32", "format": "ND", "ori_shape": (7, 7),"ori_format": "ND"},],
         "case_name": "add_row_ranges_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16, 2049), "dtype": "float32", "format": "ND", "ori_shape": (16, 2049),"ori_format": "ND"},
                    {"shape": (16, 2049), "dtype": "float32", "format": "ND", "ori_shape": (16, 2049),"ori_format": "ND"},
                    {"shape": (16, 2), "dtype": "int32", "format": "ND", "ori_shape": (16, 2),"ori_format": "ND"},
                    {"shape": (16, 2049), "dtype": "int32", "format": "ND", "ori_shape": (16, 2049),"ori_format": "ND"},],
         "case_name": "add_row_ranges_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1024, 32), "dtype": "float32", "format": "ND", "ori_shape": (1024, 32),"ori_format": "ND"},
                    {"shape": (1024, 32), "dtype": "float32", "format": "ND", "ori_shape": (1024, 32),"ori_format": "ND"},
                    {"shape": (1024, 2), "dtype": "int32", "format": "ND", "ori_shape": (1024, 2),"ori_format": "ND"},
                    {"shape": (1024, 32), "dtype": "int32", "format": "ND", "ori_shape": (1024, 32),"ori_format": "ND"},],
         "case_name": "add_row_ranges_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND"},
                    {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND"},
                    {"shape": (16, 2), "dtype": "int32", "format": "ND", "ori_shape": (16, 2),"ori_format": "ND"},
                    {"shape": (16, 16), "dtype": "int32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND"},],
         "case_name": "add_row_ranges_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1600, 160), "dtype": "float32", "format": "ND", "ori_shape": (1600, 160),"ori_format": "ND"},
                    {"shape": (1600, 160), "dtype": "float32", "format": "ND", "ori_shape": (1600, 160),"ori_format": "ND"},
                    {"shape": (1600, 2), "dtype": "int32", "format": "ND", "ori_shape": (1600, 2),"ori_format": "ND"},
                    {"shape": (1600, 160), "dtype": "int32", "format": "ND", "ori_shape": (1600, 160),"ori_format": "ND"},],
         "case_name": "add_row_ranges_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": False}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)


if __name__ == '__main__':
    ut_case.run("Ascend910")
