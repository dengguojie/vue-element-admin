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

ut_case = OpUT("MovingSumWithSigmoid", "impl.dynamic.moving_sum_with_sigmoid", "moving_sum_with_sigmoid")

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"shape": (-1,), "dtype": "float32",
                "ori_shape": (51200,),
                "format": "ND", "ori_format": "ND",
                "range": [(1, 51200)]},
               {"shape": (-1,), "dtype": "float32",
                "ori_shape": (51200,),
                "format": "ND", "ori_format": "ND",
                "range": [(1, 51200)]},
               {"shape": (1,), "dtype": "int32",
                "ori_shape": (1,),
                "format": "ND", "ori_format": "ND"}, 
               {"shape": (-1,), "dtype": "float32",
               "ori_shape": (51200,),
               "format": "ND", "ori_format": "ND",
                "range": [(1, 51200)]},
                512],
    "case_name": "test_1",
    "expect": "success",
    "support_expect": True})

ut_case.add_case(["Ascend910A","Ascend710"], {
    "params": [{"shape": (-1,), "dtype": "float32",
                "ori_shape": (2500,),
                "format": "ND", "ori_format": "ND",
                "range": [(1, 2500)]},
               {"shape": (-1,), "dtype": "float32",
               "ori_shape": (2500,),
               "format": "ND", "ori_format": "ND",
                "range": [(1, 2500)]},
               {"shape": (1,), "dtype": "int32",
               "ori_shape": (1,),
               "format": "ND", "ori_format": "ND"}, 
               {"shape": (-1,), "dtype": "float32",
               "ori_shape": (2500,),
               "format": "ND", "ori_format": "ND",
                "range": [(1, 2500)]},
                250],
    "case_name": "test_2",
    "expect": "success",
    "support_expect": True})