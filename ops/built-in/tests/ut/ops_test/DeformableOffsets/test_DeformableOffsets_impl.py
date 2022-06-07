#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf

ut_case = OpUT("DeformableOffsets", "impl.deformable_offsets", "deformable_offsets")

case1 = {"params": [{"shape": (8, 16, 16, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 16, 16, 64),
                     "ori_format": "NHWC"},
                    {"shape": (8, 16, 16, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 16, 16, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 16, 16, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 16, 16, 27),
                     "ori_format": "NHWC"},
                    {"shape": (8, 48, 48, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 48, 48, 64),
                     "ori_format": "NHWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3], [1, 1, 1, 1], "NHWC", 1, True],
         "case_name": "DeformableOffsets_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (8, 64, 64, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 64, 64, 64),
                     "ori_format": "NHWC"},
                    {"shape": (8, 64, 64, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 64, 64, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (8, 192, 192, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 192, 192, 64),
                     "ori_format": "NHWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3], [1, 1, 1, 1], "NHWC", 1, True],
         "case_name": "DeformableOffsets_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (8, 64, 64, 64), "dtype": "float16", "format": "NHWC", "ori_shape": (8, 64, 64, 64),
                     "ori_format": "NHWC"},
                    {"shape": (8, 64, 64, 27), "dtype": "float16", "format": "NHWC", "ori_shape": (8, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 64, 64, 27), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (8, 192 ,192, 64), "dtype": "float16", "format": "NHWC", "ori_shape": (8, 192, 192, 64),
                     "ori_format": "NHWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3],[1, 1, 1, 1], "NHWC", 1, True],
         "case_name": "DeformableOffsets_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (8, 64, 64, 64), "dtype": "float16", "format": "NHWC", "ori_shape": (8, 64, 64, 64),
                     "ori_format": "NHWC"},
                    {"shape": (8, 64, 64, 27), "dtype": "float16", "format": "NHWC", "ori_shape": (8, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 64, 64, 27), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (8, 192, 192, 64), "dtype": "float16", "format": "NHWC", "ori_shape": (8, 192, 192, 64),
                     "ori_format": "NHWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3], [1, 1, 1, 1], "NHWC", 1, True],
         "case_name": "DeformableOffsets_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (8, 16, 16, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 16, 16, 64),
                     "ori_format": "NHWC"},
                    {"shape": (8, 16, 16, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 16, 16, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 16, 16, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 16, 16, 27),
                     "ori_format": "NHWC"},
                    {"shape": (8, 48, 48, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 48, 48, 64),
                     "ori_format": "NHWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3], [1, 1, 1, 1], "NHWC", 1, True],
         "case_name": "DeformableOffsets_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (16, 80, 80, 128), "dtype": "float32", "format": "NHWC", "ori_shape": (16, 80, 80, 128),
                     "ori_format": "NHWC"},
                    {"shape": (16, 80, 80, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (16, 80, 80, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 80, 80, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 80, 80, 27),
                     "ori_format": "NHWC"},
                    {"shape": (16, 240, 240, 128), "dtype": "float32", "format": "NHWC", "ori_shape": (16, 240, 240, 128),
                     "ori_format": "NHWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3], [1, 1, 1, 1], "NHWC", 1, True],
         "case_name": "DeformableOffsets_6",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape": (1, 152, 152, 256), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 152, 152, 256),
                     "ori_format": "NHWC"},
                    {"shape": (1, 152, 152, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 152, 152, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 152, 152, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 152, 152, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 456, 456, 256), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 456, 456, 256),
                     "ori_format": "NHWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3], [1, 1, 1, 1], "NHWC", 1, True],
         "case_name": "DeformableOffsets_7",
         "expect": "success",
         "support_expect": True}

from impl.deformable_offsets import check_supported


def test_check_support(test_arg):
    print(test_arg)
    check_supported({"shape": (8, 64, 64, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 64, 64, 64),
                     "ori_format": "NHWC"},
                    {"shape": (8, 64, 64, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 64, 64, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (8, 192, 192, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 192, 192, 64),
                     "ori_format": "NHWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3], [1, 1, 1, 1], "NHWC", 1, False)

    check_supported({"shape": (8, 64, 64, 64, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 64, 64, 64),
                     "ori_format": "NHWC"},
                    {"shape": (8, 64, 64, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 64, 64, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (8, 192, 192, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (8, 192, 192, 64),
                     "ori_format": "NHWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3], [1, 1, 1, 1], "NHWC", 1, True)

    check_supported({"shape": (8, 64, 64, 64), "dtype": "float16", "format": "NHWC", "ori_shape": (8, 64, 64, 64),
                     "ori_format": "NHWC"},
                    {"shape": (8, 64, 64, 27), "dtype": "float16", "format": "NHWC", "ori_shape": (8, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 64, 64, 27), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (8, 192, 192, 64), "dtype": "float16", "format": "NHWC", "ori_shape": (8, 192, 192, 64),
                     "ori_format": "NHWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3], [1, 1, 1, 1], "NHWC" , 2, True)


    check_supported({"shape": (8, 64, 64, 64), "dtype": "float16", "format": "NHWC", "ori_shape": (8, 64, 64, 64),
                     "ori_format": "NHWC"},
                    {"shape": (8, 64, 64, 27), "dtype": "float16", "format": "NHWC", "ori_shape": (8, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 64, 64, 27), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 64, 64, 27),
                     "ori_format": "NHWC"},
                    {"shape": (8, 192, 192, 64), "dtype": "float16", "format": "NHWC", "ori_shape": (8, 192, 192, 64),
                     "ori_format": "NHWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3], [1, 1, 1, 1], "NHWC", 100, True)

    check_supported({"shape": (8, 64, 64, 64), "dtype": "float16", "format": "NNWC", "ori_shape": (8, 64, 64, 64),
                     "ori_format": "NNWC"},
                    {"shape": (8, 64, 64, 27), "dtype": "float16", "format": "NNWC", "ori_shape": (8, 64, 64, 27),
                     "ori_format": "NNWC"},
                    {"shape": (1, 64, 64, 27), "dtype": "float16", "format": "NNWC", "ori_shape": (1, 64, 64, 27),
                     "ori_format": "NNWC"},
                    {"shape": (8, 192, 192, 64), "dtype": "float16", "format": "NNWC", "ori_shape": (8, 192, 192, 64),
                     "ori_format": "NNWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3], [1, 1, 1, 1], "NNWC", 100, True)

    check_supported({"shape": (1, 152, 152, 256), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 152, 152, 256),
                     "ori_format": "NHWC"},
                    {"shape": (1, 152, 152, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 152, 152, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 152, 152, 27), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 152, 152, 27),
                     "ori_format": "NHWC"},
                    {"shape": (1, 456, 456, 256), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 456, 456, 256),
                     "ori_format": "NHWC"},
                    [1, 1, 1, 1], [1, 1, 1, 1], [3, 3], [1, 1, 1, 1], "NHWC", 1, True)

ut_case.add_cust_test_func(test_func=test_check_support)
ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
# ut_case.add_case(["Ascend310"], case3)
# ut_case.add_case(["Ascend310P3"], case4)
# ut_case.add_case(["Ascend310P3"], case5)
# ut_case.add_case(["Ascend910"], case6)
ut_case.add_case(["Ascend910A"], case7)
