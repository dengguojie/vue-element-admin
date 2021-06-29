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


ut_case = OpUT("TopkD", "impl.dynamic.top_k_d", "top_k_d")

case1 = {"params": [{"shape": (-1, -1), "dtype": "float16", "ori_shape": (880, 48), "format": "ND", "ori_format": "ND", "range": ((1, 880), (1, 48))},
                    {"shape": (8192,), "dtype": "float16", "format": "ND", "ori_shape": (8192,),"ori_format": "ND", "range": ((8192,8192),)},
                    {"shape": (880,16), "dtype": "float16", "format": "ND", "ori_shape": (880,16),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    {"shape": (880,16), "dtype": "int32", "format": "ND", "ori_shape": (880,16),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    16,True,-1,True],
         "case_name": "TopkD_1",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend310","Ascend710", "Ascend920A"], case1)


def test_1981(test_arg):
    from te.platform.cce_conf import te_set_version
    from impl.top_k import top_k
    te_set_version("Ascend920A", "VectorCore")
    top_k({"shape": (100000, ), "format": "ND", "dtype": "float16", "ori_shape": (100000, ), "ori_format": "ND"},
          {"shape": (100000, ), "format": "ND", "dtype": "float16", "ori_shape": (100000, ), "ori_format": "ND"},
          {"shape": (10, ), "format": "ND", "dtype": "float16", "ori_shape": (10, ), "ori_format": "ND"},
          {"shape": (10, ), "format": "ND", "dtype": "int32", "ori_shape": (10, ), "ori_format": "ND"},
          10, False, -1, True)

ut_case.add_cust_test_func(test_func=test_1981)

if __name__ == '__main__':
    ut_case.run(["Ascend910A","Ascend310","Ascend710", "Ascend920A"])
    exit(0)
