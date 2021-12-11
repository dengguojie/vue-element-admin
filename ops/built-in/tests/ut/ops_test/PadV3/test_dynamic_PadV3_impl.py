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

Dynamic PadV3 ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("PadV3", "impl.dynamic.pad_v3", "pad_v3")

case1 = {
    "params": [
        {"shape": (-1, -1, -1, -1), "dtype": "float16", "ori_shape": (-1, -1, -1, -1),
        "format": "ND", "ori_format": "ND", "range": ((-1, 1), (-1, 1), (-1, 1), (-1, 1))},
        {"shape": (-1), "dtype": "int32", "ori_shape": (-1),
        "format": "ND", "ori_format": "ND", "range": ((1, -1))},
        {"shape": (1), "dtype": "float16", "ori_shape": (1),
        "format": "ND", "ori_format": "ND", "range": ((1, 1))},
        {"shape": (-1, -1, -1, -1), "dtype": "float16", "ori_shape": (-1, -1, -1, -1),
        "format": "ND", "ori_format": "ND", "range": ((1, -1), (1, -1), (-1, -1), (-1, -1))},
        "constant"
    ],
    "case_name": "dynamic_pad_v3_01",
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
        {"shape": (1), "dtype": "float16", "ori_shape": (1),
        "format": "ND", "ori_format": "ND", "range": ((1, 1))},
        {"shape": (-1, -1, -1, -1), "dtype": "float16", "ori_shape": (-1, -1, -1, -1),
        "format": "ND", "ori_format": "ND", "range": ((1, -1), (1, -1), (-1, -1), (-1, -1))},
        "reflect"
    ],
    "case_name": "dynamic_pad_v3_02",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case3 = {
    "params": [
        {"shape": (-1, -1, -1, -1), "dtype": "float16", "ori_shape": (-1, -1, -1, -1),
        "format": "ND", "ori_format": "ND", "range": ((-1, 1), (-1, 1), (-1, 1), (-1, 1))},
        {"shape": (-1), "dtype": "int32", "ori_shape": (-1),
        "format": "ND", "ori_format": "ND", "range": ((1, -1))},
        {"shape": (1), "dtype": "float16", "ori_shape": (1),
        "format": "ND", "ori_format": "ND", "range": ((1, 1))},
        {"shape": (-1, -1, -1, -1), "dtype": "float16", "ori_shape": (-1, -1, -1, -1),
        "format": "ND", "ori_format": "ND", "range": ((1, -1), (1, -1), (-1, -1), (-1, -1))},
        "edge"
    ],
    "case_name": "dynamic_pad_v3_03",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case4 = {
    "params": [
        {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "ori_shape": (-1, -1, -1, -1),
        "format": "NC1HWC0", "ori_format": "ND", "range": ((-1, 1), (-1, 1), (-1, 1), (-1, 1), (16, 16))},
        {"shape": (-1), "dtype": "int32", "ori_shape": (-1),
        "format": "ND", "ori_format": "ND", "range": ((1, -1))},
        {"shape": (1), "dtype": "float16", "ori_shape": (1),
        "format": "ND", "ori_format": "ND", "range": ((1, 1))},
        {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "ori_shape": (-1, -1, -1, -1),
        "format": "NC1HWC0", "ori_format": "ND", "range": ((1, -1), (1, -1), (-1, -1), (-1, -1), (16, 16))},
        "constant"
    ],
    "case_name": "dynamic_pad_v3_04",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend310","Ascend710","Ascend910"], case1)
ut_case.add_case(["Ascend310","Ascend710","Ascend910"], case2)
ut_case.add_case(["Ascend310","Ascend710","Ascend910"], case3)
ut_case.add_case(["Ascend310","Ascend710","Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
