#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("SoftmaxV2", "impl.softmax_v2", "op_select_format")

case1 = {"params": [{"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"}],
         "case_name": "softmax_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    [0,1]],
         "case_name": "softmax_v2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    [1,3]],
         "case_name": "softmax_v2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    [1,3]],
         "case_name": "softmax_v2_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    [0,1]],
         "case_name": "softmax_v2_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (16, 16, 1, 16, 16, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    {"shape": (16, 16, 1, 16, 16 ,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    [0]],
         "case_name": "softmax_v2_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (16, 16, 1, 16, 16, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    {"shape": (16, 16, 1, 16, 16 ,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    [-1]],
         "case_name": "softmax_v2_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["all"], case1)
ut_case.add_case(["all"], case2)
ut_case.add_case(["all"], case3)
ut_case.add_case(["all"], case4)
ut_case.add_case(["all"], case5)
ut_case.add_case(["all"], case6)
ut_case.add_case(["all"], case7)

