#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPoolGradWithArgmaxv2", None, "max_pool_grad_with_argmax")

case1 = {"params": [{"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    {"shape": (2,2,49,73,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,49,73,16),"ori_format": "NHWC"},
                    {"shape": (13888,), "dtype": "uint16", "format": "NHWC", "ori_shape": (13888,),"ori_format": "NHWC"},
                    {"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    [1, 1, 1, 1]],
         "case_name": "max_pool_grad_with_arxmax_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (32,1,70,514,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,1,70,514,16),"ori_format": "NHWC"},
                    {"shape": (32,1,14,85,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,1,14,85,16),"ori_format": "NHWC"},
                    {"shape": (1576960,), "dtype": "uint16", "format": "NHWC", "ori_shape": (1576960,),"ori_format": "NHWC"},
                    {"shape": (32,1,70,514,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,1,70,514,16),"ori_format": "NHWC"},
                    [1, 5, 8, 1],
                    [1, 5, 6, 1],
                    [1, 1, 1, 1]],
         "case_name": "max_pool_grad_with_arxmax_v2_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend710", "Ascend910A"], case2)

if __name__ == '__main__':
    # ut_case.run()
    ut_case.run(["Ascend710", "Ascend910A"])
    exit(0)
