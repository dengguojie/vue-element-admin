#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the2
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dynamic Topk ut case
"""
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("DynSeqOuter", "impl.dynamic.dyn_seq_outer", "dyn_seq_outer")

ut_case.add_case(["Ascend910A","Ascend710","Ascend310"], {
    "params": [{"shape": (-1, -1), "dtype": "float32",
                "ori_shape": (41, 512),
                "format": "ND", "ori_format": "ND",
                "range": [(1, 51200), (1, 51200)]},
               {"shape": (-1, -1), "dtype": "float32",
                "ori_shape": (27, 512),
                "format": "ND", "ori_format": "ND",
                "range": [(1, 51200), (1, 51200)]},
               {"shape": (-1,), "dtype": "int32",
                "ori_shape": (8,),
                "format": "ND", "ori_format": "ND",
                "range": [(8, 51200)]},
               {"shape": (-1,), "dtype": "int32",
                "ori_shape": (8,),
                "format": "ND", "ori_format": "ND",
                "range": [(8, 51200)]},
               {"shape": (-1, -1), "dtype": "float32",
               "ori_shape": (141, 512),
               "format": "ND", "ori_format": "ND",
                "range": [(1, 51200), (1, 51200)]}],
    "case_name": "test_1",
    "expect": "success",
    "support_expect": True})
