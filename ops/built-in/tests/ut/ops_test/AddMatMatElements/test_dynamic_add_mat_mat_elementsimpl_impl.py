#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
test_AddMatMatElements_impl
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("AddMatMatElements",
               "impl.dynamic.add_mat_mat_elements",
               "add_mat_mat_elements")

case1 = {"params": [{"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 2)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 2)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND", "range": [(1, 100)]},
                    ],
         "case_name": "case1",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
