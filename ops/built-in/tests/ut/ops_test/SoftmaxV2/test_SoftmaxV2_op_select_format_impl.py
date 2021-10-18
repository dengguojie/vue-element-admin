#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("SoftmaxV2", "impl.softmax_v2", "op_select_format")

case1 = {"params": [{"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"}],
         "case_name": "softmax_v2_op_select_format_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    [0,1]],
         "case_name": "softmax_v2_op_select_format_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    [1,3]],
         "case_name": "softmax_v2_op_select_format_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    [1,3]],
         "case_name": "softmax_v2_op_select_format_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    [0,1]],
         "case_name": "softmax_v2_op_select_format_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (16, 16, 1, 16, 16, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    {"shape": (16, 16, 1, 16, 16 ,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    [0]],
         "case_name": "softmax_v2_op_select_format_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (16, 16, 1, 16, 16, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    {"shape": (16, 16, 1, 16, 16 ,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    [-1]],
         "case_name": "softmax_v2_op_select_format_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [{"shape": (16, 16, 4, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16, 50, 50),"ori_format": "ND"},
                    {"shape": (16, 16, 4, 4, 16 ,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16, 50, 50),"ori_format": "ND"},
                    [-1]],
         "case_name": "softmax_v2_op_select_format_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case9 = {"params": [{"shape": (16, 16, 4, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16, 50, 50),"ori_format": "ND"},
                    {"shape": (16, 16, 4, 4, 16 ,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16, 50, 50),"ori_format": "ND"},
                    [-2]],
         "case_name": "softmax_v2_op_select_format_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case10 = {"params": [{"shape": (16,1000000), "dtype": "float16", "format": "ND", "ori_shape": (16,1000000),"ori_format": "ND"},
                    {"shape": (16,1000000), "dtype": "float16", "format": "ND", "ori_shape": (16,1000000),"ori_format": "ND"},
                    [-1]],
         "case_name": "softmax_v2_op_select_format_10",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case11 = {"params": [{"shape": (16,1000), "dtype": "float16", "format": "ND", "ori_shape": (16,1000),"ori_format": "ND"},
                    {"shape": (16,1000), "dtype": "float16", "format": "ND", "ori_shape": (16,1000),"ori_format": "ND"},
                    [-1]],
         "case_name": "softmax_v2_op_select_format_11",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case12 = {"params": [{"shape": (8, 6, 546, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 8732, 81),"ori_format": "ND"},
                    {"shape": (8, 6, 546, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 8732, 81),"ori_format": "ND"},
                    [2]],
         "case_name": "softmax_v2_op_select_format_12",
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
ut_case.add_case(["all"], case8)
ut_case.add_case(["all"], case9)
ut_case.add_case(["all"], case10)
ut_case.add_case(["all"], case11)
from te.platform.cce_conf import te_set_version
te_set_version("Ascend710")
ut_case.add_case(["Ascend710", "Ascend910A"], case12)



if __name__ == '__main__':
    ut_case.run("Ascend710")
