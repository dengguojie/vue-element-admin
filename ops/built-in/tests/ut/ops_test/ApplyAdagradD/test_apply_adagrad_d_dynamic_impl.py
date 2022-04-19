#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
test_applyadagradd_impl
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("ApplyAdagradD",
               "impl.dynamic.apply_adagrad_d",
               "apply_adagrad_d")

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
                    True
                    ],
         "case_name": "case1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
                    True
                    ],
         "case_name": "case1",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
