#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("StridedWrite", None, None)

case1 = {"params": [{"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    {"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    1, 1],
         "case_name": "StridedWrite_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1, 4, 3, 4), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4), 'ori_format':"NC1HWC0"},
                    {"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    1, 1],
         "case_name": "StridedWrite_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    {"shape": (1, 4, 3, 4), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4), 'ori_format':"NC1HWC0"},
                    1, 1],
         "case_name": "StridedWrite_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (1, 4, 3, 4, 16), "dtype": "float32", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    {"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    1, 1],
         "case_name": "StridedWrite_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    {"shape": (1, 4, 3, 4, 16), "dtype": "float32", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    0, 1],
         "case_name": "StridedWrite_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    {"shape": (1, 4, 3, 4, 16), "dtype": "int8", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    0, 1],
         "case_name": "StridedWrite_6",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "ND", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"ND"},
                    {"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    0, 1],
         "case_name": "StridedWrite_7",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [{"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    {"shape": (1, 4, 3, 4, 16), "dtype": "float16", "format": "NC1HWC0", 'ori_shape':(1, 4, 3, 4, 16), 'ori_format':"NC1HWC0"},
                    0, 1],
         "case_name": "StridedWrite_8",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
ut_case.add_case("Ascend910", case1)
ut_case.add_case("Ascend910", case2)
ut_case.add_case("Ascend910", case3)
ut_case.add_case("Ascend910", case4)
ut_case.add_case("Ascend910", case5)
ut_case.add_case("Ascend910", case6)
ut_case.add_case("Ascend910", case7)
ut_case.add_case("Ascend910", case8)


if __name__ == '__main__':
    ut_case.run("Ascend910")