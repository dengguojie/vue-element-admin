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
from op_test_frame.ut import OpUT


ut_case = OpUT("TopKV2D", "impl.top_k_v2_d", "check_supported")

case1 = {"params": [{"shape": (1, 16), "dtype": "float16", "ori_shape": (1, 16), "format": "ND", "ori_format": "ND", "range": ((1, 1), (16, 16))},
                    {"shape": (8192,), "dtype": "float16", "format": "ND", "ori_shape": (8192,),"ori_format": "ND", "range": ((8192,8192),)},
                    {"shape": (1,-1), "dtype": "float16", "format": "ND", "ori_shape": (1,-1),"ori_format": "ND", "range": ((1, 1), (1, 16))},
                    {"shape": (1,-1), "dtype": "int32", "format": "ND", "ori_shape": (1,-1),"ori_format": "ND", "range": ((1, 1), (1, 16))},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":((1,1),)},
                    True,-1,True],
         "case_name": "TopKV2D_1",
         "expect": "success",
         "support_expect": True}


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
