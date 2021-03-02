#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
ut_test_dynamic_relu_v2
'''
from op_test_frame.ut import OpUT
import te

ut_case = OpUT("ReluV2", "impl.dynamic.relu_v2", "relu_v2")

case1 = {"params": [{"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "NC1HWC0", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "NC1HWC0", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "uint8", "format": "ND", "ori_shape": (1, 1),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]}],
         "case_name": "relu_v2_float16",
         "expect": "success"}

case2 = {"params": [{"shape": (1, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "NC1HWC0", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "NC1HWC0", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "uint8", "format": "ND", "ori_shape": (1, 1),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]}],
         "case_name": "relu_v2_float32",
         "expect": "success"}

case3 = {"params": [{"shape": (1, -1), "dtype": "int8", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "NC1HWC0", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "int8", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "NC1HWC0", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "uint8", "format": "ND", "ori_shape": (1, 1),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]}],
         "case_name": "relu_v2_int8",
         "expect": "success"}

ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)


if __name__ == "__main__":
    with te.op.dynamic():
        ut_case.run("Ascend910A")
