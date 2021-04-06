#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dynamic InTopk ut case
"""
import te
from op_test_frame.ut import OpUT


ut_case = OpUT("InTopk", "impl.dynamic.in_top_k", "in_top_k")

case1 = {"params": [{"shape": (-1, -1), "dtype": "float32", "ori_shape": (880, 48), "format": "ND", "ori_format": "ND", "range": ((1, 880), (1, 48))},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (880,),"ori_format": "ND", "range": ((1, 880), )},
                    {"shape": (-1,), "dtype": "uint8", "format": "ND", "ori_shape": (880,),"ori_format": "ND", "range": ((1, 880), )},
                    16],
         "case_name": "InTopk_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1), "dtype": "float32", "ori_shape": (98026, 42), "format": "ND", "ori_format": "ND", "range": ((1, 98026), (1, 42))},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (98026,),"ori_format": "ND", "range": ((1, 98026), )},
                    {"shape": (-1,), "dtype": "uint8", "format": "ND", "ori_shape": (98026,),"ori_format": "ND", "range": ((1, 98026), )},
                    16],
         "case_name": "InTopk_2",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
