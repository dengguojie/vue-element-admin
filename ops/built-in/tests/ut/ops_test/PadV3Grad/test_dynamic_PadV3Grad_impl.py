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

Dynamic PadV3Grad ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("PadV3Grad", "impl.dynamic.pad_v3_grad", "pad_v3_grad")

case1 = {
    "params": [
        {"shape": (-1, -1, -1, -1), "dtype": "float16", "ori_shape": (-1, -1, -1, -1),
        "format": "ND", "ori_format": "ND", "range": ((-1, 1), (-1, 1), (-1, 1), (-1, 1))},
        {"shape": (-1), "dtype": "int32", "ori_shape": (-1),
        "format": "ND", "ori_format": "ND", "range": ((1, -1))},
        {"shape": (-1, -1, -1, -1), "dtype": "float16", "ori_shape": (-1, -1, -1, -1),
        "format": "ND", "ori_format": "ND", "range": ((1, -1), (1, -1), (-1, -1), (-1, -1))},
        "reflect"
    ],
    "case_name": "dynamic_pad_v3_grad_01",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1, -1, -1, -1), "dtype": "float16", "ori_shape": (-1, -1, -1, -1),
        "format": "ND", "ori_format": "ND", "range": ((-1, 1), (-1, 1), (-1, 1), (-1, 1))},
        {"shape": (-1), "dtype": "int32", "ori_shape": (-1),
        "format": "ND", "ori_format": "ND", "range": ((1, -1))},
        {"shape": (-1, -1, -1, -1), "dtype": "float16", "ori_shape": (-1, -1, -1, -1),
        "format": "ND", "ori_format": "ND", "range": ((1, -1), (1, -1), (-1, -1), (-1, -1))},
        "edge"
    ],
    "case_name": "dynamic_pad_v3_grad_02",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend310","Ascend710","Ascend910A"], case1)
ut_case.add_case(["Ascend310","Ascend710","Ascend910A"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend310","Ascend710","Ascend910A")
