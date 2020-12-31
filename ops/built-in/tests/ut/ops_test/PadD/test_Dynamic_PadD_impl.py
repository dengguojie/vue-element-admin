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

Dynamic PadD ut case
"""
import te
from op_test_frame.ut import OpUT


ut_case = OpUT("PadD", "impl.dynamic.pad_d", "pad_d")


def gen_dynamic_pad_d_case(dict_x, dict_y, paddings, kernel_name_val, expect):
    return {"params": [dict_x, dict_y, paddings],
            "case_name": kernel_name_val,
            "expect": expect,
            "support_expect": True}

ut_case.add_case(["Ascend910A"],
                 gen_dynamic_pad_d_case(
                     {"shape": (-1, 2, 1), "dtype": "float16", "ori_shape": (-1, 2, 1),
                      "format": "ND", "ori_format": "ND", "range": ((1, 38), (2, 2), (1, 1))},
                     {"shape": (-1, 2, 509680), "dtype": "float16", "ori_shape": (-1, 2, 509680),
                      "format": "ND", "ori_format": "ND", "range": ((1, 38), (2, 2), (509680, 509680))},
                     ((0,0), (0,0), (0,509679)),
                     "dynamic_pad_d_01", "success"))
if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910A")
    exit(0)
