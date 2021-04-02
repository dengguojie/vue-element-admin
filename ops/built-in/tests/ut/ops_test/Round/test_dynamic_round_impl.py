#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
ut_test_dynamic_rint
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("Round", "impl.dynamic.round", "round")

case1 = {"params": [{"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (16, 16),
                     "ori_format": "ND", "range": [(1, None), (1, None)]},
                    {"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (16, 16),
                     "ori_format": "ND", "range": [(1, None), (1, None)]}],
         "case_name": "round_float16",
         "expect": "success"}

case2 = {"params": [{"shape": (1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 16, 32),
                     "ori_format": "ND", "range": [(1, 1), (1, None), (1, None)]},
                    {"shape": (1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 16, 32),
                     "ori_format": "ND", "range": [(1, 1), (1, None), (1, None)]}],
         "case_name": "round_float32",
         "expect": "success"}

case3 = {"params": [{"shape": (1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (1, 16, 32),
                     "ori_format": "ND", "range": [(1, 1), (1, None), (1, None)]},
                    {"shape": (1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (1, 16, 32),
                     "ori_format": "ND", "range": [(1, 1), (1, None), (1, None)]}],
         "case_name": "round_int32",
         "expect": "success"}

ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)

if __name__ == "__main__":
    ut_case.run(["Ascend310", "Ascend910A"])
