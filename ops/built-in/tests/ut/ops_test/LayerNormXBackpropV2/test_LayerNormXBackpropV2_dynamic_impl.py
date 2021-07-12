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

ut_case = OpUT("LayerNormXBackpropV2", "impl.dynamic.layer_norm_x_backprop_v2", "layer_norm_x_backprop_v2")

case1 = {"params": [{"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (512,), "dtype": "float32", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "support_expect": True}
case2 = {"params": [{"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (512,), "dtype": "float16", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "support_expect": True}
case3 = {"params": [{"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (512,), "dtype": "float32", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "support_expect": True}
case4 = {"params": [{"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (512,), "dtype": "float16", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "support_expect": True}

def test_generalization(args):
    from impl.dynamic.layer_norm_x_backprop_v2 import layer_norm_x_backprop_generalization
    layer_norm_x_backprop_generalization(
        {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
        {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
        {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
        {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
        {"shape": (512,), "dtype": "float16", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
        {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
        {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
        None, None)


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_cust_test_func(test_func=test_generalization)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
