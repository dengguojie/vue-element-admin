#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AsinhGrad", None, None)

case1 = {"params": [{"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND"},
                    {"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND"},
                    {"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND"}],
         "case_name": "asinh_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)

if __name__ == '__main__':
    ut_case.run()
    exit(0)