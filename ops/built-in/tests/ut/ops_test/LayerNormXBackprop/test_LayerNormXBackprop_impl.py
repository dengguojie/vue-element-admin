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

Dynamic layer_norm_x_backprop ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("LayerNormXBackprop", None, None)

case1 = {"params": [{"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"}],
         "case_name": "layer_norm_x_backprop_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"}],
         "case_name": "layer_norm_x_backprop_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)