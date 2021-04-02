#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
ut_test_dynamic_apply_add_sign_d
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("ApplyAddSignD",
               "impl.dynamic.apply_add_sign_d",
               "apply_add_sign_d")

case1 = {"params": [{"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (1, 16),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (1, 16),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (1, 16),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (1, 16),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (1, 16),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]}],
         "case_name": "dynamic_apply_add_sign_d_001",
         "expect": "success"}


ut_case.add_case("Ascend910A", case1)


if __name__ == "__main__":
    ut_case.run(["Ascend910A"])
