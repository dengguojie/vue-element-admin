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
                    {"shape": (4096*2,), "dtype": "int32", "format": "ND", "ori_shape": (4096*2,),"ori_format": "ND", "range": (4096*2,)},
                    {"shape": (880,16), "dtype": "float32", "format": "ND", "ori_shape": (880,16),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    {"shape": (880,16), "dtype": "int32", "format": "ND", "ori_shape": (880,16),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    16,True,-1,True],
         "case_name": "TopkD_1",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
