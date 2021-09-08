#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os


ut_case = OpUT("SoftplusV2", None, None)

case1 = {"params": [{"shape": (1, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 4),"ori_format": "ND",
                     "param_type": "input"},
                    {"shape": (1, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 4),"ori_format": "ND",
                     "param_type": "output"}, 1.0, 20.0],
         "case_name": "softplus_v2_1",
         "expect": "success"}
case2 = {"params": [{"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND",
                     "param_type": "input"},
                    {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND",
                     "param_type": "output"}, 2.0, 10.0],
         "case_name": "softplus_v2_2",
         "expect": "success"}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),
                     "ori_format": "ND", "param_type": "input"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),
                     "ori_format": "ND", "param_type": "output"}, 10.0, 15.0],
         "case_name": "softplus_v2_3",
         "expect": "success"}
case4 = {"params": [{"shape": (32, 2, 4, 1, 6), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 1, 6),
                     "ori_format": "ND", "param_type": "input"},
                    {"shape": (32, 2, 4, 1, 6), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 1, 6),
                     "ori_format": "ND", "param_type": "output"}, 2.0, 40.0],
         "case_name": "softplus_v2_4",
         "expect": "success"}
case5 = {"params": [{"shape": (12, 2, 4, 3, 1, 6), "dtype": "float16", "format": "ND", "ori_shape": (12, 2, 4, 3, 1, 6),
                     "ori_format": "ND", "param_type": "input"},
                    {"shape": (12, 2, 4, 3, 1, 6), "dtype": "float16", "format": "ND", "ori_shape": (12, 2, 4, 3, 1, 6),
                     "ori_format": "ND", "param_type": "output"}, 5.32152, 18.05113],
         "case_name": "softplus_v2_5",
         "expect": "success"}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
