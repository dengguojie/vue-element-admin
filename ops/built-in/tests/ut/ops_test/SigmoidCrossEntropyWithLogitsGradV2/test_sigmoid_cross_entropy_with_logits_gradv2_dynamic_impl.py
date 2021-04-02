#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
test_SigmoidCrossEntropyWithLogitsGradV2_dynamic
'''
from op_test_frame.ut import OpUT
ut_case = OpUT("SigmoidCrossEntropyWithLogitsGradV2",
               "impl.dynamic.sigmoid_cross_entropy_with_logits_grad_v2",
               "sigmoid_cross_entropy_with_logits_grad_v2")

case1 = {"params": [{"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    "mean"
                    ],
         "case_name": "case1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    "mean"
                    ],
         "case_name": "case2",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
