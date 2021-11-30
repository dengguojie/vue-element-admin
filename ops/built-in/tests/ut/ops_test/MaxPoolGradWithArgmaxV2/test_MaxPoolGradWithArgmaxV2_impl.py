#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPoolGradWithArgmaxv2", "impl.dynamic.max_pool_grad_with_argmaxv2", "max_pool_grad_with_argmax_v2")

case1 = {"params": [{"shape": (2,2,36,36,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,36,36,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,18,18,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,18,18,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,4,22,16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (2,2,4,22,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,36,36,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,36,36,16),"ori_format": "NC1HWC0"},
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [1, 1, 0, 0]],
         "case_name": "max_pool_grad_with_arxmax_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2,32,36,36,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,32,36,36,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,32,36,36,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,32,36,36,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,32,9,82,16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (2,32,9,82,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,32,36,36,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,32,36,36,16),"ori_format": "NC1HWC0"},
                    [1, 1, 3, 3],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]],
         "case_name": "max_pool_grad_with_arxmax_v2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,2,136,136,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,136,136,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,68,68,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,68,68,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,4,290,16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (2,2,4,290,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,136,136,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,136,136,16),"ori_format": "NC1HWC0"},
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [1, 1, 0, 0]],
         "case_name": "max_pool_grad_with_arxmax_v2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,2,620,620,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,620,620,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,20,20,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,20,20,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,961,26,16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (2,2,961,26,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,620,620,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,620,620,16),"ori_format": "NC1HWC0"},
                    [1, 1, 31, 31],
                    [1, 1, 31, 31],
                    [1, 1, 0, 0]],
         "case_name": "max_pool_grad_with_arxmax_v2_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend710", "Ascend910A"], case4)

if __name__ == '__main__':
    # ut_case.run()
    ut_case.run("Ascend910A")
    exit(0)
