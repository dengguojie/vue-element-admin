#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dynamic Topk ut case
"""
import te
from op_test_frame.ut import OpUT


ut_case = OpUT("TopkD", "impl.top_k_d", "top_k_d")

case1 = {"params": [{"shape": (8, 32), "dtype": "float16", "ori_shape": (8, 32), "format": "ND", "ori_format": "ND", "range": ((1, 880), (1, 48))},
                    {"shape": (8192,), "dtype": "float16", "format": "ND", "ori_shape": (8192,),"ori_format": "ND", "range": ((8192,8192),)},
                    {"shape": (8,8), "dtype": "float16", "format": "ND", "ori_shape": (8,8),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    {"shape": (8,8), "dtype": "int32", "format": "ND", "ori_shape": (8,8),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    8,True,-1,True],
         "case_name": "TopkD_k_8",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (8, 5000), "dtype": "float16", "ori_shape": (8, 5000), "format": "ND", "ori_format": "ND", "range": ((1, 880), (1, 48))},
                    {"shape": (8192,), "dtype": "float16", "format": "ND", "ori_shape": (8192,),"ori_format": "ND", "range": ((8192,8192),)},
                    {"shape": (8,6), "dtype": "float16", "format": "ND", "ori_shape": (8,6),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    {"shape": (8,6), "dtype": "int32", "format": "ND", "ori_shape": (8,6),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    6,True,-1,True],
         "case_name": "TopkD_k_6",
         "expect": "RuntimeError",
         "support_expect": False}
# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend310","Ascend610", "Ascend710","Ascend920A"], case1)
ut_case.add_case(["Ascend910A","Ascend310","Ascend610", "Ascend710","Ascend920A"], case3)

def test_static_1951(test_arg):
    from te.platform.cce_conf import te_set_version
    from impl.top_k_d import top_k_d
    te_set_version("Ascend610", "VectorCore")
    top_k_d({"shape": (8, 32), "format": "ND", "dtype": "float16", "ori_shape": (8, 32), "ori_format": "ND"},
          {"shape": (8192, ), "format": "ND", "dtype": "float16", "ori_shape": (8192, ), "ori_format": "ND"},
          {"shape": (8, 8), "format": "ND", "dtype": "float16", "ori_shape": (8, 8), "ori_format": "ND"},
          {"shape": (8, 8), "format": "ND", "dtype": "int32", "ori_shape": (8, 8), "ori_format": "ND"},
          8, True, -1, True)
    te_set_version(test_arg)
ut_case.add_cust_test_func(test_func=test_static_1951)

if __name__ == '__main__':
    ut_case.run(["Ascend910A","Ascend310","Ascend610", "Ascend710","Ascend920A"])
    exit(0)
