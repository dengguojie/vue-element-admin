#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPoolWithArgmaxv1", None, "max_pool_with_argmax_v1")

case1 = {"params": [{"shape": (2, 2, 96, 144,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 2, 96, 144, 16), "ori_format": "NC1HWC0"},
                    {"shape": (2, 2, 48, 72, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 2, 48, 72, 16), "ori_format": "NC1HWC0"},
                    {"shape": (13888,), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (13888,), "ori_format": "NC1HWC0"},
                    [1, 2, 2, 1],
                    [1, 2, 2, 1],
                    [1, 1, 1, 1]],
         "case_name": "max_pool_with_arxmax_v1_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (32, 4, 112, 112, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32, 4, 112, 112, 16), "ori_format": "NHWC"},
                    {"shape": (32, 4, 56, 56, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32, 4, 56, 56, 16), "ori_format": "NHWC"},
                    {"shape": (32, 4, 9, 197, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (32, 4, 9, 197, 16), "ori_format": "NHWC"},
                    [1, 3, 3, 1],
                    [1, 2, 2, 1],
                    [1, 1, 1, 1]],
         "case_name": "max_pool_with_arxmax_v1_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2, 4, 672, 672, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 4, 672, 672, 16), "ori_format": "NHWC"},
                    {"shape": (2, 4, 336, 336, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 4, 336, 336, 16), "ori_format": "NHWC"},
                    {"shape": (2, 4, 9, 7057, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (2, 4, 9, 7057, 16), "ori_format": "NHWC"},
                    [1, 3, 3, 1],
                    [1, 2, 2, 1],
                    [1, 1, 1, 1]],
         "case_name": "max_pool_with_arxmax_v1_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2, 2, 5, 5, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 2, 5, 5, 16), "ori_format": "NHWC"},
                    {"shape": (2, 2, 5, 5, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 2, 5, 5, 16), "ori_format": "NHWC"},
                    {"shape": (2, 2, 169, 3, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (2, 2, 169, 3, 16), "ori_format": "NHWC"},
                    [1, 13, 13, 1],
                    [1, 1, 1, 1],
                    [1, 6, 6, 1]],
         "case_name": "max_pool_with_arxmax_v1_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 1, 12, 12, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 12, 12, 16), "ori_format": "NHWC"},
                    {"shape": (1, 1, 12, 12, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 12, 12, 16), "ori_format": "NHWC"},
                    {"shape": (1, 1, 1, 10, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 10, 16), "ori_format": "NHWC"},
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 0, 0, 1]],
         "case_name": "max_pool_with_arxmax_v1_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (16, 4, 120, 1200, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 4, 120, 1200, 16), "ori_format": "NHWC"},
                    {"shape": (16, 4, 9, 2251, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 4, 9, 2251, 16), "ori_format": "NHWC"},
                    {"shape": (16, 4, 60, 600, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (16, 4, 60, 600, 16), "ori_format": "NHWC"},
                    [1, 3, 3, 1],
                    [1, 2, 2, 1],
                    [1, 1, 1, 1]],
         "case_name": "max_pool_with_arxmax_v1_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NHWC"},
                    {"shape": (1, 1, 169, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 169, 2, 16), "ori_format": "NHWC"},
                    {"shape": (1, 1, 1, 1, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NHWC"},
                    [1, 13, 13, 1],
                    [1, 1, 1, 1],
                    [1, 6, 6, 1]],
         "case_name": "max_pool_with_arxmax_v1_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend910A"], case7)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
