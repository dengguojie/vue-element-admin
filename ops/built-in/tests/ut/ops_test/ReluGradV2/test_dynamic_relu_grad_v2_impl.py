#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
ut_test_dynamic_relu_grad_v2
'''

from op_test_frame.ut import OpUT

ut_case = OpUT("ReluGradV2", "impl.dynamic.relu_grad_v2", "relu_grad_v2")

case1 = {"params": [{"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "bool", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]}],
         "case_name": "relu_grad_v2_float16",
         "expect": "success"}
case2 = {"params": [{"shape": (1, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "bool", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]}],
         "case_name": "relu_grad_v2_float32",
         "expect": "success"}

ut_case.add_case("all", case1)
ut_case.add_case("all", case2)


if __name__ == "__main__":
    ut_case.run("Ascend910A")
