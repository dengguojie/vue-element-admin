#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
test_AddMatMatElements_impl
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("AddMatMatElements", None, None)

case1 = {"params": [{"shape": (1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"},
                    ],
         "case_name": "apply_adagrad_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
