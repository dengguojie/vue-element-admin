#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPoolWithArgmaxv2", "impl.dynamic.max_pool_with_argmaxv2", "max_pool_with_argmax_v2")

case1 = {"params": [{"shape": (2,2,35,35,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,35,35,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,35,35,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,35,35,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,1,78,16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (2,2,1,78,16),"ori_format": "NC1HWC0"},
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 0, 0]],
         "case_name": "max_pool_with_arxmax_v2_0",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (2,2,33,33,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,33,33,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,16,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,16,16,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,4,17,16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (2,2,4,17,16),"ori_format": "NC1HWC0"},
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [1, 1, 0, 0]],
         "case_name": "max_pool_with_arxmax_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (2,4,160,201,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,4,160,201,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,4,32,40,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,4,32,40,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,4,16,81,16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (2,4,16,81,16),"ori_format": "NC1HWC0"},
                    [1, 1, 4, 4],
                    [1, 1, 5, 5],
                    [1, 1, 0, 0]],
         "case_name": "max_pool_with_arxmax_v2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (2,4,256,1000,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,4,256,1000,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,4,51,200,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,4,51,200,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,4,25,639,16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (2,4,25,639,16),"ori_format": "NC1HWC0"},
                    [1, 1, 5, 5],
                    [1, 1, 5, 5],
                    [1, 1, 0, 0]],
         "case_name": "max_pool_with_arxmax_v2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend910A"], case4)

if __name__ == '__main__':
    # ut_case.run()
    ut_case.run("Ascend910A")
    exit(0)