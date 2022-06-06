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
         "case_name": "LayerNormXBackpropV2_01",
         "support_expect": True}
case2 = {"params": [{"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (512,), "dtype": "float16", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "case_name": "LayerNormXBackpropV2_02",
         "support_expect": True}
case3 = {"params": [{"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (512,), "dtype": "float32", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "case_name": "LayerNormXBackpropV2_03",
         "support_expect": True}
case4 = {"params": [{"shape": (-1, -1, 30), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 30),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 30), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 30),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (30,), "dtype": "float16", "format": "ND", "ori_shape": (30,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 30), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 30),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 30), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 30),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "case_name": "LayerNormXBackpropV2_04",
         "support_expect": True}
case5 = {"params": [{"shape": (64, 114, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, -1, -1, 16),"ori_format": "FRACTAL_NZ", "range": ((0, 1), (0, 1), (0, 1), (0, 1))},
                    {"shape": (64, 114, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, -1, -1, 16),"ori_format": "FRACTAL_NZ","range": ((0, 1), (0, 1), (0, 1), (0, 1))},
                    {"shape": (1824, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1),"ori_format": "ND", "range": ((0, 5), (0, 5))},
                    {"shape": (1824, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1),"ori_format": "ND", "range": ((0, 5), (0, 5))},
                    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (1024,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (64, 1824, 16), "dtype": "float16", "format": "ND", "ori_shape": (64, 1824, 16),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (64, 1824, 16), "dtype": "float32", "format": "ND", "ori_shape": (64, 1824, 16),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "case_name": "LayerNormXBackpropV2_05",
         "support_expect": True}
case6 = {"params": [{"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "case_name": "LayerNormXBackpropV2_06",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
