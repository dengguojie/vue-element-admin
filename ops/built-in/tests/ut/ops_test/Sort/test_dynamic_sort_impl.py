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

ut_case = OpUT("Sort", "impl.dynamic.sort", "sort")

case1 = {"params": [
    {"shape": (-1, -1, -1, -1), "dtype": "float16", "ori_shape": (10, 10, 10, 32), "format": "ND", "ori_format": "ND",
     "range": ((1, 10), (1, 10), (1, 10), (1, 32))},
    {"shape": (10, 10, 10, 32), "dtype": "float16", "format": "ND", "ori_shape": (10, 10, 10, 32), "ori_format": "ND",
     "range": ((1, 10), (1, 10), (1, 10), (1, 32))},
    {"shape": (10, 10, 10, 32), "dtype": "int32", "format": "ND", "ori_shape": (10, 10, 10, 32), "ori_format": "ND",
     "range": ((1, 10), (1, 10), (1, 10), (1, 32))},
    3, False],
    "case_name": "Sort_1",
    "expect": "success",
    "support_expect": True}

ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910", "Ascend310", "Ascend710"])
    exit(0)
