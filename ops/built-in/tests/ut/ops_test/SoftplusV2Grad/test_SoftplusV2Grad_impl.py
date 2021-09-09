#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os


ut_case = OpUT("SoftplusV2Grad", None, None)

case1 = {"params": [{"shape": (1, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 4),"ori_format": "ND",
                     "param_type": "input"},
                    {"shape": (1, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 4),"ori_format": "ND",
                     "param_type": "input"},
                    {"shape": (1, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 4),"ori_format": "ND",
                     "param_type": "output"}, 1.0, 20.0],
         "case_name": "softplus_v2_grad_1",
         "expect": "success"}
case2 = {"params": [{"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND",
                     "param_type": "input"},
                    {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND",
                     "param_type": "input"},
                    {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND",
                     "param_type": "output"}, 2.0, 10.0],
         "case_name": "softplus_v2_grad_2",
         "expect": "success"}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),
                     "ori_format": "ND", "param_type": "input"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),
                     "ori_format": "ND", "param_type": "input"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),
                     "ori_format": "ND", "param_type": "output"}, 10.0, 15.0],
         "case_name": "softplus_v2_grad_3",
         "expect": "success"}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
