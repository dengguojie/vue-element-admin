#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
sigmoid_cross_entropy_with_logits_gradv2_test
'''
import os
from op_test_frame.ut import OpUT

ut_case = OpUT("SigmoidCrossEntropyWithLogitsGradV2", None, None)

case1 = {"params": [{"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND"},
                    {"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND"},
                    {"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND"},
                    {"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND"},
                    {"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND"},
                    {"shape": (15, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND"},
                    "mean"
                    ],
         "case_name": "case1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (15, 32, 64), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64),
                     "ori_format": "ND"},
                    {"shape": (15, 32, 64), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64),
                     "ori_format": "ND"},
                    {"shape": (15, 32, 64), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64),
                     "ori_format": "ND"},
                    {"shape": (15, 32, 64), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64),
                     "ori_format": "ND"},
                    None,
                    {"shape": (15, 32, 64), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64),
                     "ori_format": "ND"},
                    "none"
                    ],
         "case_name": "case2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (15, 32, 64, 128), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64, 128),
                     "ori_format": "ND"},
                    {"shape": (15, 32, 64, 128), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64, 128),
                     "ori_format": "ND"},
                    {"shape": (15, 32, 64, 128), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64, 128),
                     "ori_format": "ND"},
                    None,
                    {"shape": (15, 32, 64, 128), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64, 128),
                     "ori_format": "ND"},
                    {"shape": (15, 32, 64, 128), "dtype": "float32", "format": "ND", "ori_shape": (15, 32, 64, 128),
                     "ori_format": "ND"},
                    "sum"
                    ],
         "case_name": "case3",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
