#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("OneHotD", None, None)

case1 = {"params": [{"shape": (1, ), "dtype": "int32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    2],
         "case_name": "one_hot_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1, ), "dtype": "int32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    16],
         "case_name": "one_hot_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, ), "dtype": "int32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32, ), "dtype": "float32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32, ), "dtype": "float32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32, ), "dtype": "float32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    64],
         "case_name": "one_hot_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (41, ), "dtype": "int32", "format": "ND", "ori_shape": (41, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (41, 16), "dtype": "float16", "format": "ND", "ori_shape": (41,16 ),"ori_format": "ND"},
                    16],
         "case_name": "one_hot_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1536,), "dtype": "int32", "format": "ND", "ori_shape": (1536,),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1536, 200), "dtype": "float16", "format": "ND", "ori_shape": (1536,200),"ori_format": "ND"},
                    200],
         "case_name": "one_hot_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (1539,), "dtype": "int32", "format": "ND", "ori_shape": (1539,),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1539, 201), "dtype": "float16", "format": "ND", "ori_shape": (1539,201),"ori_format": "ND"},
                    201],
         "case_name": "one_hot_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310"], case4)
ut_case.add_case(["Ascend310"], case5)
ut_case.add_case(["Ascend310"], case6)


if __name__ == '__main__':
    ut_case.run("Ascend310")
    exit(0)
