#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPool3dGradGradD", None, None)

case1 = {"params": [{"shape": (1,1,5,5,16,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,1,5,5,16,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,5,5,16,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,1,5,5,16,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,5,5,16,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,1,5,5,16,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,5,5,16,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,1,5,5,16,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,5,5,16,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,1,5,5,16,16),"ori_format": "NDC1HWC0"},
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]],
         "case_name": "max_pool3d_grad_grad_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == '__main__':
    # ut_case.run()
    ut_case.run("Ascend910")
    exit(0)